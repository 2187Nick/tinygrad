import unittest
import functools
import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates
from tinygrad.dtype import AddrSpace
from tinygrad.runtime.autogen.amd.rdna3.ins import *
import tinygrad.runtime.autogen.amd.rdna3.ins as r3
import tinygrad.runtime.autogen.amd.rdna4.ins as r4
from tinygrad.renderer.amd.dsl import s, v
from test.amd.helpers import TARGET_TO_ARCH
from extra.gemm.amd_asm_matmul import Kernel

def custom_add_one(A:UOp) -> UOp:
  A = A.flatten()
  assert dtypes.is_float(A.dtype.base), f"buffer dtype must be float32, got {A.dtype}"
  threads = UOp.special(A.size, "lidx0")
  insts = [
    s_load_b64(s[0:1], s[0:1], soffset=NULL),
    s_waitcnt_lgkmcnt(sdst=NULL, simm16=0),
    v_lshlrev_b32_e32(v[0], 2, v[0]), # element offset
    global_load_b32(v[1], v[0], saddr=s[0:1]),
    s_waitcnt_vmcnt(sdst=NULL, simm16=0),
    v_mov_b32_e32(v[2], 1.0),
    v_add_f32_e32(v[1], v[1], v[2]),
    global_store_b32(addr=v[0], data=v[1], saddr=s[0:1]),
    s_endpgm(),
  ]
  sink = UOp.sink(A.base, threads, arg=KernelInfo(f"custom_add_one_{A.size}", estimates=Estimates(ops=A.size, mem=A.size*4*2)))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD"), UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))

def custom_add_var(A:UOp, B:UOp) -> UOp:
  A,B = A.flatten(), B.flatten()
  assert A.dtype.base == dtypes.uint32, f"buffer dtype must be uint32, got {A.dtype}"
  threads = UOp.special(A.size, "lidx0")
  var = UOp.variable("var", 0, 10)
  insts = [
    s_load_b128(s[4:7], s[0:1]),
    s_load_b32(s[8], s[0:1], offset=0x10), # all threads load the same variable
    s_waitcnt_lgkmcnt(sdst=NULL, simm16=0),
    v_lshlrev_b32_e32(v[0], 2, v[0]), # element offset, different per thread
    global_load_b32(v[1], v[0], saddr=s[6:7]),
    s_waitcnt_vmcnt(sdst=NULL, simm16=0),
    v_add_nc_u32_e32(v[1], s[8], v[1]),
    global_store_b32(addr=v[0], data=v[1], saddr=s[4:5]),
    s_endpgm(),
  ]
  sink = UOp.sink(A.base, B.base, var, threads, arg=KernelInfo(f"custom_add_var_{A.size}"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD"), UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))

def custom_wave_sync(A:UOp, arch:str) -> UOp:
  # 4 waves across 1024 WG — enough to saturate a SIMD with many concurrent WGs
  # s_sleep yields the SIMD so waves from different WGs interleave, causing barrier packet reordering
  threads = UOp.special(128, "lidx0")
  wg = UOp.special(1024, "gidx0")
  insts = []
  for _ in range(4):
    insts.append(s_sleep(4))
    insts += [s_barrier()] if arch == "rdna3" else [r4.s_barrier_signal(), r4.s_barrier_wait()]
    insts += [s_nop(0)]*4
  insts.append(s_endpgm())
  sink = UOp.sink(A.base, threads, wg, arg=KernelInfo("custom_wave_sync"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD"), UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))

def custom_lds_sync(A:UOp, arch:str) -> UOp:
  A = A.flatten()
  num_threads = A.shape[0]
  threads = UOp.special(num_threads, "lidx0")
  wg = UOp.special(1, "gidx0")
  lds = UOp(Ops.DEFINE_LOCAL, dtypes.uint8.ptr(size=512, addrspace=AddrSpace.LOCAL), (), 'lds')  # 128 * 4 bytes
  isa = r4 if arch == "rdna4" else r3
  wait_kmcnt = [isa.s_wait_kmcnt(simm16=0)] if arch == "rdna4" else [isa.s_waitcnt_lgkmcnt(sdst=NULL, simm16=0)]
  wait_dscnt = [isa.s_wait_dscnt(simm16=0)] if arch == "rdna4" else [isa.s_waitcnt_lgkmcnt(sdst=NULL, simm16=0)]
  barrier = [isa.s_barrier_signal(ssrc0=-1), isa.s_barrier_wait(simm16=-1)] if arch == "rdna4" else [isa.s_barrier()]
  global_store = [isa.global_store_b32(vaddr=v[6:7], saddr=s[0:1], vsrc=v[5])] if arch == "rdna4" \
      else [isa.global_store_b32(addr=v[6], data=v[5], saddr=s[0:1])]
  insts = [
    isa.s_load_b64(s[0:1], s[0:1], soffset=NULL),
    *wait_kmcnt,
    isa.v_lshlrev_b32_e32(v[1], 2, v[0]),
    # lds[thread_idx] = thread_idx
    isa.ds_store_b32(addr=v[1], data0=v[0]),
    *wait_dscnt,
    *barrier,
    # out[threaed_idx] = thread_idx == num_threads ? -1 : lds[thread_idx + 1]
    isa.v_add_nc_u32_e32(v[2], 4, v[1]),
    isa.v_cmp_gt_u32_e32(num_threads-1, v[0]),
    isa.ds_load_b32(vdst=v[3], addr=v[2]),
    *wait_dscnt,
    isa.v_mov_b32_e32(v[4], -1),
    isa.v_cndmask_b32_e32(v[5], v[4], v[3]),
    isa.v_lshlrev_b32_e32(v[6], 2, v[0]),
    *global_store,
    isa.s_endpgm(),
  ]
  sink = UOp.sink(A.base, lds, threads, wg, arg=KernelInfo("custom_lds_sync"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD"), UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))

def custom_handwritten(A:UOp, arch:str) -> UOp:
  A = A.flatten()
  threads = UOp.special(128, "lidx0")
  wg = UOp.special(256, "gidx0")
  lds = UOp(Ops.DEFINE_LOCAL, dtypes.uint8.ptr(size=512, addrspace=AddrSpace.LOCAL), (), 'lds')  # 128 * 4 bytes
  k = Kernel(arch)
  k.emit(r4.s_nop(0))
  k.emit(r4.v_mov_b32_e32(v[1], 4))
  def emit_alt():
    for i in range(2):
      k.emit(r4.v_mov_b32_e32(v[20+i], 4.0))
      k.emit(r4.v_rcp_f32_e32(v[22+i], v[20+i]))
      k.emit(r4.s_mov_b32(s[20+i], i))
      k.emit(r4.s_mul_i32(s[14+i], s[12+i], 32))
  def emit_wmma():
    for _ in range(2):
      k.emit(r4.v_wmma_f32_16x16x16_f16(v[0:7], v[8:11], v[8:11], 1))
  k.label("start")
  k.emit(s_mov_b32(s[1], 10))
  k.label("loop")
  # wmma should've overlapped here if it was a different unit?
  for _ in range(2):
    emit_wmma()
    emit_alt()
  for _ in range(8): k.emit(s_nop(1))
  k.emit(s_add_u32(s[1], s[1], -1))
  k.emit(s_cmp_eq_i32(s[1], 0))
  k.emit(s_cbranch_scc0(), target="loop")
  k.emit(r4.s_endpgm())
  insts = k.finalize()
  sink = UOp.sink(A.base, threads, wg, lds, arg=KernelInfo("custom_handwritten"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD"), UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))

def custom_data_deps(A:UOp, arch:str) -> UOp:
  A = A.flatten()
  threads = UOp.special(A.size, "lidx0")
  k = Kernel(arch)
  k.emit(s_load_b64(s[0:1], s[0:1], soffset=NULL))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))
  k.emit(v_lshlrev_b32_e32(v[0], 2, v[0]))
  k.emit(global_load_b32(v[1], v[0], saddr=s[0:1]))
  k.emit(s_waitcnt_vmcnt(sdst=NULL, simm16=0))
  k.emit(v_add_f32_e32(v[1], 1.0, v[1]))
  k.emit(global_store_b32(addr=v[0], data=v[1], saddr=s[0:1]))
  k.emit(s_endpgm())
  insts = k.finalize()
  sink = UOp.sink(A.base, threads, arg=KernelInfo("custom_data_deps"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD"), UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))

def custom_probe_sgpr_cmps(A:UOp, arch:str) -> UOp:
  """Probe: 4 v_cmp_e64 back-to-back writing s[0..3], then 4 v_cndmask reading them.
  Run TWICE with a long trans-pipeline gap between. Compare stall patterns."""
  A = A.flatten()
  threads = UOp.special(A.size, "lidx0")
  k = Kernel(arch)
  k.emit(s_load_b64(s[0:1], s[0:1], soffset=NULL))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))
  k.emit(v_lshlrev_b32_e32(v[0], 2, v[0]))
  k.emit(global_load_b32(v[1], v[0], saddr=s[0:1]))
  k.emit(s_waitcnt_vmcnt(sdst=NULL, simm16=0))
  # Prep v[2..5] to compare
  k.emit(v_mov_b32_e32(v[2], 1.0))
  k.emit(v_mov_b32_e32(v[3], 2.0))
  k.emit(v_mov_b32_e32(v[4], 3.0))
  k.emit(v_mov_b32_e32(v[5], 4.0))
  # Block A: 4 v_cmps back-to-back (use s[4..6] to avoid kernel args s[0:1])
  k.emit(v_cmp_gt_f32_e32(0.5, v[2]))                  # VCC
  k.emit(v_cmp_gt_f32_e64(s[4], 0.5, v[3]))            # s[4]
  k.emit(v_cmp_gt_f32_e64(s[5], 0.5, v[4]))            # s[5] — watch this
  k.emit(v_cmp_gt_f32_e64(s[6], 0.5, v[5]))            # s[6]
  # 4 v_cndmask
  k.emit(v_cndmask_b32_e32(v[6], 0.0, v[3]))           # VCC implicit (e32)
  k.emit(v_cndmask_b32_e64(v[7], 0.0, v[3], s[4]))
  k.emit(v_cndmask_b32_e64(v[8], 0.0, v[3], s[5]))
  k.emit(v_cndmask_b32_e64(v[9], 0.0, v[3], s[6]))
  # Long trans pipeline to drain state
  k.emit(v_exp_f32_e32(v[10], v[2]))
  k.emit(v_log_f32_e32(v[10], v[10]))
  k.emit(v_sqrt_f32_e32(v[10], v[10]))
  k.emit(s_waitcnt(simm16=0))
  k.emit(s_nop(15))  # force drain
  k.emit(s_nop(15))
  k.emit(s_nop(15))
  # Block B: IDENTICAL sequence - compare stalls
  k.emit(v_cmp_gt_f32_e32(0.5, v[2]))
  k.emit(v_cmp_gt_f32_e64(s[4], 0.5, v[3]))
  k.emit(v_cmp_gt_f32_e64(s[5], 0.5, v[4]))
  k.emit(v_cmp_gt_f32_e64(s[6], 0.5, v[5]))
  k.emit(v_cndmask_b32_e32(v[11], 0.0, v[3]))
  k.emit(v_cndmask_b32_e64(v[12], 0.0, v[3], s[4]))
  k.emit(v_cndmask_b32_e64(v[13], 0.0, v[3], s[5]))
  k.emit(v_cndmask_b32_e64(v[14], 0.0, v[3], s[6]))
  # Store something to keep compiler happy
  k.emit(v_add_f32_e32(v[1], v[6], v[1]))
  k.emit(global_store_b32(addr=v[0], data=v[1], saddr=s[0:1]))
  k.emit(s_endpgm())
  insts = k.finalize()
  sink = UOp.sink(A.base, threads, arg=KernelInfo("custom_probe_sgpr_cmps"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD"), UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))

def custom_probe_cmp_chain(A:UOp, arch:str) -> UOp:
  """Probe: 8 v_cmp_e64 in a row writing s[0..7]. Check where stalls appear."""
  A = A.flatten()
  threads = UOp.special(A.size, "lidx0")
  k = Kernel(arch)
  k.emit(s_load_b64(s[0:1], s[0:1], soffset=NULL))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))
  k.emit(v_lshlrev_b32_e32(v[0], 2, v[0]))
  k.emit(global_load_b32(v[1], v[0], saddr=s[0:1]))
  k.emit(s_waitcnt_vmcnt(sdst=NULL, simm16=0))
  for i in range(8): k.emit(v_mov_b32_e32(v[2+i], float(i+1)))
  # 8 v_cmps back-to-back writing s[4..11] (avoid s[0:1] which holds kernel args)
  for i in range(8): k.emit(v_cmp_gt_f32_e64(s[4+i], 0.5, v[2+i]))
  # nop drain
  k.emit(s_nop(15))
  k.emit(v_add_f32_e32(v[1], 1.0, v[1]))
  k.emit(global_store_b32(addr=v[0], data=v[1], saddr=s[0:1]))
  k.emit(s_endpgm())
  insts = k.finalize()
  sink = UOp.sink(A.base, threads, arg=KernelInfo("custom_probe_cmp_chain"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD"), UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))

def custom_probe_branch_cost(A:UOp, arch:str) -> UOp:
  """Probe: branches of various types, to measure branch issue cost vs retire delay.
  Use forward branches that are never taken (scc=0 + scc1 branch, exec stays set)."""
  A = A.flatten()
  threads = UOp.special(A.size, "lidx0")
  k = Kernel(arch)
  k.emit(s_load_b64(s[0:1], s[0:1], soffset=NULL))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))
  k.emit(v_lshlrev_b32_e32(v[0], 2, v[0]))
  k.emit(global_load_b32(v[1], v[0], saddr=s[0:1]))
  k.emit(s_waitcnt_vmcnt(sdst=NULL, simm16=0))
  # SCC-based branch NOT TAKEN (scc=0, branch on scc1)
  k.emit(s_mov_b32(s[4], 0))
  k.emit(s_cmp_eq_i32(s[4], 1))      # sets SCC=0
  k.emit(s_cbranch_scc1(), target="end")
  k.emit(v_mov_b32_e32(v[10], 1.0))
  k.emit(v_mov_b32_e32(v[11], 2.0))
  # Second SCC branch
  k.emit(s_cmp_eq_i32(s[4], 1))
  k.emit(s_cbranch_scc1(), target="end")
  k.emit(v_mov_b32_e32(v[12], 3.0))
  k.emit(v_add_f32_e32(v[1], v[10], v[1]))
  k.emit(global_store_b32(addr=v[0], data=v[1], saddr=s[0:1]))
  k.label("end")
  k.emit(s_endpgm())
  insts = k.finalize()
  sink = UOp.sink(A.base, threads, arg=KernelInfo("custom_probe_branch_cost"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD"), UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))

def custom_probe_vmem_chain(A:UOp, arch:str) -> UOp:
  """Probe: back-to-back global_stores to probe VMEM issue port serialization."""
  A = A.flatten()
  threads = UOp.special(A.size, "lidx0")
  k = Kernel(arch)
  k.emit(s_load_b64(s[0:1], s[0:1], soffset=NULL))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))
  k.emit(v_lshlrev_b32_e32(v[0], 2, v[0]))
  k.emit(v_mov_b32_e32(v[1], 1.0))
  k.emit(v_mov_b32_e32(v[2], 2.0))
  k.emit(v_mov_b32_e32(v[3], 3.0))
  k.emit(v_mov_b32_e32(v[4], 4.0))
  k.emit(s_nop(10))  # drain VALU
  k.emit(global_store_b32(addr=v[0], data=v[1], saddr=s[0:1]))
  k.emit(global_store_b32(addr=v[0], data=v[2], saddr=s[0:1]))
  k.emit(global_store_b32(addr=v[0], data=v[3], saddr=s[0:1]))
  k.emit(global_store_b32(addr=v[0], data=v[4], saddr=s[0:1]))
  k.emit(s_endpgm())
  insts = k.finalize()
  sink = UOp.sink(A.base, threads, arg=KernelInfo("custom_probe_vmem_chain"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD"), UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))

# ──────────────────────────────────────────────────────────────────────────────
# Parametric probes (2026-04-18) — isolate the hypotheses from MISMATCH_ANALYSIS.md
# Each probe takes only A so it slots into rigorous_hw_test.py the same way as
# the existing custom_probe_* kernels.
# ──────────────────────────────────────────────────────────────────────────────

def _probe_cold_start(A:UOp, arch:str, n:int, name:str) -> UOp:
  """D1: no global_load. After s_waitcnt both waves exit within ~2cy and fight
  the CU's VALU issue port. Wave 1's first VALU should slip by ~min(N, burst)."""
  A = A.flatten()
  threads = UOp.special(A.size, "lidx0")
  k = Kernel(arch)
  k.emit(s_load_b64(s[0:1], s[0:1], soffset=NULL))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))
  # N back-to-back VALUs on freshly-allocated VGPRs — no VMEM, no LDS.
  for i in range(n): k.emit(v_mov_b32_e32(v[2+i], float(i+1)))
  k.emit(v_add_f32_e32(v[1], 1.0, v[2]))
  k.emit(v_lshlrev_b32_e32(v[0], 2, v[0]))
  k.emit(global_store_b32(addr=v[0], data=v[1], saddr=s[0:1]))
  k.emit(s_endpgm())
  insts = k.finalize()
  sink = UOp.sink(A.base, threads, arg=KernelInfo(name))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD"), UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))

def custom_probe_cold_start_n2(A:UOp, arch:str) -> UOp: return _probe_cold_start(A, arch, 2, "probe_cold_start_n2")
def custom_probe_cold_start_n4(A:UOp, arch:str) -> UOp: return _probe_cold_start(A, arch, 4, "probe_cold_start_n4")
def custom_probe_cold_start_n8(A:UOp, arch:str) -> UOp: return _probe_cold_start(A, arch, 8, "probe_cold_start_n8")

def _probe_nop_chain(A:UOp, arch:str, n:int, name:str) -> UOp:
  """B1: N consecutive s_nop(15) then a VALU. Expect first (n-1) nops = 16cy,
  last nop = 20cy (pipeline resume). Contrast n=1 (isolated) vs n≥2 (chain)."""
  A = A.flatten()
  threads = UOp.special(A.size, "lidx0")
  k = Kernel(arch)
  k.emit(s_load_b64(s[0:1], s[0:1], soffset=NULL))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))
  k.emit(v_lshlrev_b32_e32(v[0], 2, v[0]))
  k.emit(global_load_b32(v[1], v[0], saddr=s[0:1]))
  k.emit(s_waitcnt_vmcnt(sdst=NULL, simm16=0))
  # N nops in a chain
  for _ in range(n): k.emit(s_nop(15))
  # Follow-up VALU so we can measure the last-nop stamp
  k.emit(v_add_f32_e32(v[1], 1.0, v[1]))
  k.emit(global_store_b32(addr=v[0], data=v[1], saddr=s[0:1]))
  k.emit(s_endpgm())
  insts = k.finalize()
  sink = UOp.sink(A.base, threads, arg=KernelInfo(name))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD"), UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))

def custom_probe_nop_chain_n1(A:UOp, arch:str) -> UOp: return _probe_nop_chain(A, arch, 1, "probe_nop_chain_n1")
def custom_probe_nop_chain_n3(A:UOp, arch:str) -> UOp: return _probe_nop_chain(A, arch, 3, "probe_nop_chain_n3")
def custom_probe_nop_chain_n5(A:UOp, arch:str) -> UOp: return _probe_nop_chain(A, arch, 5, "probe_nop_chain_n5")

def _probe_store_bypass(A:UOp, arch:str, preload:bool, name:str) -> UOp:
  """A4 (A1 vs A2): solo VALU→store with/without a prior global_load to warm
  the VMEM scoreboard. Expect cold = 21cy, warm = 17cy for the store dt."""
  A = A.flatten()
  threads = UOp.special(A.size, "lidx0")
  k = Kernel(arch)
  k.emit(s_load_b64(s[0:1], s[0:1], soffset=NULL))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))
  k.emit(v_lshlrev_b32_e32(v[0], 2, v[0]))
  if preload:
    k.emit(global_load_b32(v[2], v[0], saddr=s[0:1]))
    k.emit(s_waitcnt_vmcnt(sdst=NULL, simm16=0))
  k.emit(v_mov_b32_e32(v[1], 1.0))
  k.emit(v_add_f32_e32(v[1], 1.0, v[1]))  # dt=1
  k.emit(global_store_b32(addr=v[0], data=v[1], saddr=s[0:1]))  # measure this dt
  k.emit(s_endpgm())
  insts = k.finalize()
  sink = UOp.sink(A.base, threads, arg=KernelInfo(name))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD"), UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))

def custom_probe_store_cold(A:UOp, arch:str) -> UOp: return _probe_store_bypass(A, arch, False, "probe_store_cold")
def custom_probe_store_warm(A:UOp, arch:str) -> UOp: return _probe_store_bypass(A, arch, True,  "probe_store_warm")

def _probe_trans_pair(A:UOp, arch:str, spacing:int, name:str) -> UOp:
  """F3: wave-pair TRANS-pipe interlock. Both waves run the same trans chain;
  compare wave-0 vs wave-1 dt for v_log. spacing = #v_adds between v_exp/v_log."""
  A = A.flatten()
  threads = UOp.special(A.size, "lidx0")
  k = Kernel(arch)
  k.emit(s_load_b64(s[0:1], s[0:1], soffset=NULL))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))
  k.emit(v_lshlrev_b32_e32(v[0], 2, v[0]))
  k.emit(global_load_b32(v[1], v[0], saddr=s[0:1]))
  k.emit(s_waitcnt_vmcnt(sdst=NULL, simm16=0))
  k.emit(v_mov_b32_e32(v[2], 1.0))
  k.emit(v_exp_f32_e32(v[2], v[2]))
  for i in range(spacing): k.emit(v_add_f32_e32(v[3+i], 1.0, v[3+i]))
  k.emit(v_log_f32_e32(v[2], v[2]))  # measure this dt — HW serializes vs sibling wave
  k.emit(s_waitcnt(simm16=0))
  k.emit(v_add_f32_e32(v[1], v[1], v[2]))
  k.emit(global_store_b32(addr=v[0], data=v[1], saddr=s[0:1]))
  k.emit(s_endpgm())
  insts = k.finalize()
  sink = UOp.sink(A.base, threads, arg=KernelInfo(name))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD"), UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))

def custom_probe_trans_pair_tight(A:UOp, arch:str)  -> UOp: return _probe_trans_pair(A, arch, 0, "probe_trans_pair_tight")
def custom_probe_trans_pair_spaced(A:UOp, arch:str) -> UOp: return _probe_trans_pair(A, arch, 4, "probe_trans_pair_spaced")

def _probe_scalar_beat(A:UOp, arch:str, phase:int, name:str) -> UOp:
  """E1: scalar-pipe 4-beat phase. Insert `phase` s_nop(0) before s_cmp → s_cbranch
  NOT TAKEN. Expect the cbranch dt to oscillate with period 4 as phase varies."""
  A = A.flatten()
  threads = UOp.special(A.size, "lidx0")
  k = Kernel(arch)
  k.emit(s_load_b64(s[0:1], s[0:1], soffset=NULL))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))
  k.emit(v_lshlrev_b32_e32(v[0], 2, v[0]))
  k.emit(global_load_b32(v[1], v[0], saddr=s[0:1]))
  k.emit(s_waitcnt_vmcnt(sdst=NULL, simm16=0))
  k.emit(s_mov_b32(s[4], 0))
  for _ in range(phase): k.emit(s_nop(0))
  k.emit(s_cmp_eq_i32(s[4], 1))      # SCC = 0
  k.emit(s_cbranch_scc1(), target="end")
  k.emit(v_add_f32_e32(v[1], 1.0, v[1]))
  k.emit(global_store_b32(addr=v[0], data=v[1], saddr=s[0:1]))
  k.label("end")
  k.emit(s_endpgm())
  insts = k.finalize()
  sink = UOp.sink(A.base, threads, arg=KernelInfo(name))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD"), UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))

def custom_probe_scalar_beat_p0(A:UOp, arch:str) -> UOp: return _probe_scalar_beat(A, arch, 0, "probe_scalar_beat_p0")
def custom_probe_scalar_beat_p1(A:UOp, arch:str) -> UOp: return _probe_scalar_beat(A, arch, 1, "probe_scalar_beat_p1")
def custom_probe_scalar_beat_p2(A:UOp, arch:str) -> UOp: return _probe_scalar_beat(A, arch, 2, "probe_scalar_beat_p2")
def custom_probe_scalar_beat_p3(A:UOp, arch:str) -> UOp: return _probe_scalar_beat(A, arch, 3, "probe_scalar_beat_p3")

def _probe_vopd(A:UOp, arch:str, mode:str, name:str) -> UOp:
  """C1: VOPD spacing. 'chain' = 4 back-to-back VOPDs; 'split' = 2+waitcnt+2;
  'nodep' = 4 writing disjoint registers. VOPD uses r3.VOPDOp enum."""
  A = A.flatten()
  threads = UOp.special(A.size, "lidx0")
  k = Kernel(arch)
  k.emit(s_load_b64(s[0:1], s[0:1], soffset=NULL))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))
  k.emit(v_lshlrev_b32_e32(v[0], 2, v[0]))
  k.emit(global_load_b32(v[1], v[0], saddr=s[0:1]))
  k.emit(s_waitcnt_vmcnt(sdst=NULL, simm16=0))
  # Initialize VGPRs
  for i in range(8): k.emit(v_mov_b32_e32(v[4+i], float(i+1)))
  def vopd(x, y): return r3.VOPD(opx=r3.VOPDOp.V_DUAL_MUL_F32, opy=r3.VOPDOp.V_DUAL_MUL_F32,
                                 vdstx=v[x],       srcx0=v[x+2], vsrcx1=v[x+4],
                                 vdsty=v[y],       srcy0=v[y+2], vsrcy1=v[y+4])
  if mode == "chain":
    k.emit(vopd(4, 5)); k.emit(vopd(4, 5)); k.emit(vopd(4, 5)); k.emit(vopd(4, 5))
  elif mode == "split":
    k.emit(vopd(4, 5)); k.emit(vopd(4, 5))
    k.emit(s_waitcnt_depctr(simm16=0xfffe))
    k.emit(vopd(4, 5)); k.emit(vopd(4, 5))
  elif mode == "nodep":
    k.emit(vopd(4, 5)); k.emit(vopd(6, 7)); k.emit(vopd(8, 9)); k.emit(vopd(10, 11))
  k.emit(s_waitcnt(simm16=0))
  k.emit(v_add_f32_e32(v[1], v[4], v[1]))
  k.emit(global_store_b32(addr=v[0], data=v[1], saddr=s[0:1]))
  k.emit(s_endpgm())
  insts = k.finalize()
  sink = UOp.sink(A.base, threads, arg=KernelInfo(name))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD"), UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))

def custom_probe_vopd_chain(A:UOp, arch:str) -> UOp: return _probe_vopd(A, arch, "chain", "probe_vopd_chain")
def custom_probe_vopd_split(A:UOp, arch:str) -> UOp: return _probe_vopd(A, arch, "split", "probe_vopd_split")
def custom_probe_vopd_nodep(A:UOp, arch:str) -> UOp: return _probe_vopd(A, arch, "nodep", "probe_vopd_nodep")

@unittest.skipUnless(Device.DEFAULT == "AMD", "requires AMD device")
class TestCustomKernel(unittest.TestCase):
  def setUp(self): self.arch = TARGET_TO_ARCH[Device["AMD"].arch]

  def test_simple(self):
    if self.arch != "rdna3": self.skipTest("only rdna3")
    a = Tensor.full((16, 16), 1.).contiguous().realize()
    a = Tensor.custom_kernel(a, fxn=custom_add_one)[0]
    ei = a.schedule()[-1].lower()
    self.assertEqual(ei.prg.estimates.ops, a.numel())
    self.assertEqual(ei.prg.estimates.mem, a.nbytes()*2)
    ei.run()
    self.assertTrue((a.numpy() == 2.).all())

  def test_variable(self):
    if self.arch != "rdna3": self.skipTest("only rdna3")
    b = Tensor.full((16, 16), 1, dtype=dtypes.uint32).contiguous().realize()
    a = Tensor.zeros_like(b).contiguous().realize()
    a = Tensor.custom_kernel(a, b, fxn=custom_add_var)[0]
    ei = a.schedule()[-1].lower()
    for i in range(4):
      ei.run({"var":i})
      self.assertTrue((a.numpy() == 1+i).all())

  def test_lds_sync(self):
    if self.arch not in ("rdna3", "rdna4"): self.skipTest("only rdna3/rdna4")
    a = Tensor.empty(128, dtype=dtypes.int32).contiguous().realize()
    a = Tensor.custom_kernel(a, fxn=functools.partial(custom_lds_sync, arch=self.arch))[0]
    a.realize()
    ref = Tensor.arange(1, 129, dtype=dtypes.int32)
    ref[127] = -1
    self.assertListEqual(a.tolist(), ref.tolist())

  def test_handwritten(self):
    if self.arch != "rdna4": self.skipTest("only tested on rdna4")
    a = Tensor.empty(1024, dtype=dtypes.int32).contiguous().realize()
    a = Tensor.custom_kernel(a, fxn=functools.partial(custom_handwritten, arch=self.arch))[0]
    a.realize()

  def test_data_deps(self):
    if self.arch != "rdna3": self.skipTest("only tested on rdna3")
    a = Tensor(np.full(32, 5.0, dtype=np.float32)).realize()
    a = Tensor.custom_kernel(a, fxn=functools.partial(custom_data_deps, arch=self.arch))[0]
    a.realize()
    self.assertTrue((a.numpy() == 6.0).all())

if __name__ == "__main__":
  unittest.main()
