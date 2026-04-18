#!/usr/bin/env python3
"""Batch A microbenchmarks — families A.7 / A.8 / A.9 / A.10 (36 kernels).

Targets the RDNA3 cycle-accurate emulator bounty. Each kernel isolates ONE
HW timing knob so a mismatch versus the captured 7900 XTX traces pins
an unambiguous diagnosis.

Families:
  A.7 — s_nop by N and predecessor                       (16 kernels)
  A.8 — s_waitcnt variants                               ( 6 kernels)
  A.9 — LDS opcodes (ds_load / ds_store)                 ( 8 kernels)
  A.10 — global_load / global_store                      ( 6 kernels)

Naming convention is mb_<family>_<shape>, matching MICROBENCH_TAXONOMY.md
sections A.7-A.10.

Every kernel uses the standard microbench prologue and epilogue unless a
prologue override is required:
  - A.9 LDS kernels allocate 1 KB LDS via `extra={"lds_size": 1024}`, which
    makes `_build_program_uop` add an Ops.DEFINE_LOCAL to the sink.
  - A.10 "_isolated" variants skip the standard `global_load_b32` prologue
    so the timing captures a cold global_store_b32.
"""
from __future__ import annotations

from tinygrad.renderer.amd.dsl import s, v, NULL  # noqa: F401
from tinygrad.runtime.autogen.amd.rdna3.ins import (
  s_nop, s_mov_b32,
  s_waitcnt, s_waitcnt_vmcnt, s_waitcnt_lgkmcnt, s_waitcnt_depctr,
  s_load_b64, v_lshlrev_b32_e32,
  v_add_f32_e32, v_mov_b32_e32,
  ds_store_b32, ds_store_b64, ds_store_b128,
  ds_load_b32, ds_load_b64, ds_load_b128,
  global_load_b32, global_load_b64, global_load_b128,
  global_store_b32, global_store_b64, global_store_b128,
  s_endpgm,
)

from extra.sqtt.rgp.microbench import microbench, register, MicroBench, Kernel

# ── Shared helpers ──────────────────────────────────────────────────────────

def _prologue_no_load(k: "Kernel") -> None:
  """Prologue that skips the standard global_load_b32 pre-warm.

  After this runs:
    - s[0:1] = output buffer base address
    - v[0]   = element offset (4*thread_idx)
    - v[1]   is NOT loaded (used as a plain scratch VGPR)
  """
  k.emit(s_load_b64(s[0:1], s[0:1], soffset=NULL))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))
  k.emit(v_lshlrev_b32_e32(v[0], 2, v[0]))
  # seed v[1] so the store has a real value without an actual global_load
  k.emit(v_mov_b32_e32(v[1], 1.0))

def _prologue_lds(k: "Kernel") -> None:
  """Standard prologue + seed a per-thread LDS index into v[2].

  v[2] = v[0] (byte offset = 4*thread_idx) is a valid LDS address inside the
  1024-byte LDS slab (we only need 4 B/thread for small LDS probes).
  After this:
    - s[0:1], v[0], v[1] as per standard prologue
    - v[2]   = LDS byte offset (4*thread_idx)
  """
  k.emit(s_load_b64(s[0:1], s[0:1], soffset=NULL))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))
  k.emit(v_lshlrev_b32_e32(v[0], 2, v[0]))
  k.emit(global_load_b32(v[1], v[0], saddr=s[0:1]))
  k.emit(s_waitcnt_vmcnt(sdst=NULL, simm16=0))
  # reuse v[0] as the LDS byte offset too -- it is already 4*tid
  k.emit(v_mov_b32_e32(v[2], v[0]))

def _seed_v_many(k: "Kernel", start: int, count: int) -> None:
  """Seed v[start..start+count-1] with independent literal floats."""
  for i in range(count):
    k.emit(v_mov_b32_e32(v[start+i], float(i+1)))

# ═════════════════════════════════════════════════════════════════════════════
# A.7 — s_nop by N and predecessor (16 kernels)
# ═════════════════════════════════════════════════════════════════════════════
# The standard prologue leaves v[1] warm via global_load + s_waitcnt_vmcnt.
# The "_after_valu" variants inject an extra v_add to establish a fresh VALU
# predecessor before the s_nop, decoupling from the waitcnt_vmcnt stamp.

@microbench(name="mb_snop_0_after_valu", category="s_nop")
def _mb_snop_0_after_valu(k: "Kernel") -> None:
  k.emit(v_add_f32_e32(v[1], 1.0, v[1]))
  k.emit(s_nop(0))
  k.emit(v_add_f32_e32(v[1], 1.0, v[1]))

@microbench(name="mb_snop_5_after_valu", category="s_nop")
def _mb_snop_5_after_valu(k: "Kernel") -> None:
  k.emit(v_add_f32_e32(v[1], 1.0, v[1]))
  k.emit(s_nop(5))
  k.emit(v_add_f32_e32(v[1], 1.0, v[1]))

@microbench(name="mb_snop_10_after_valu", category="s_nop")
def _mb_snop_10_after_valu(k: "Kernel") -> None:
  k.emit(v_add_f32_e32(v[1], 1.0, v[1]))
  k.emit(s_nop(10))
  k.emit(v_add_f32_e32(v[1], 1.0, v[1]))

@microbench(name="mb_snop_15_after_valu", category="s_nop")
def _mb_snop_15_after_valu(k: "Kernel") -> None:
  k.emit(v_add_f32_e32(v[1], 1.0, v[1]))
  k.emit(s_nop(15))
  k.emit(v_add_f32_e32(v[1], 1.0, v[1]))

@microbench(name="mb_snop_0_after_scalar", category="s_nop")
def _mb_snop_0_after_scalar(k: "Kernel") -> None:
  k.emit(s_mov_b32(s[4], 0))
  k.emit(s_nop(0))
  k.emit(v_add_f32_e32(v[1], 1.0, v[1]))

@microbench(name="mb_snop_15_after_scalar", category="s_nop")
def _mb_snop_15_after_scalar(k: "Kernel") -> None:
  k.emit(s_mov_b32(s[4], 0))
  k.emit(s_nop(15))
  k.emit(v_add_f32_e32(v[1], 1.0, v[1]))

@microbench(name="mb_snop_15_after_waitcnt_vmcnt", category="s_nop")
def _mb_snop_15_after_waitcnt_vmcnt(k: "Kernel") -> None:
  # Second global_load so we have a fresh vmcnt drain right before the nop.
  k.emit(global_load_b32(v[2], v[0], saddr=s[0:1]))
  k.emit(s_waitcnt_vmcnt(sdst=NULL, simm16=0))
  k.emit(s_nop(15))
  k.emit(v_add_f32_e32(v[1], 1.0, v[1]))

@microbench(name="mb_snop_15_after_waitcnt_lgkmcnt", category="s_nop")
def _mb_snop_15_after_waitcnt_lgkmcnt(k: "Kernel") -> None:
  # Scalar load to bump lgkmcnt, then drain it, then the nop.
  k.emit(s_load_b64(s[4:5], s[0:1], soffset=NULL))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))
  k.emit(s_nop(15))
  k.emit(v_add_f32_e32(v[1], 1.0, v[1]))

@microbench(name="mb_snop_15_after_waitcnt_empty", category="s_nop")
def _mb_snop_15_after_waitcnt_empty(k: "Kernel") -> None:
  # Drain everything, then an s_waitcnt(0) barrier with no pending ops, then nop.
  k.emit(s_waitcnt(simm16=0))
  k.emit(s_nop(15))
  k.emit(v_add_f32_e32(v[1], 1.0, v[1]))

@microbench(name="mb_snop_15_after_depctr", category="s_nop")
def _mb_snop_15_after_depctr(k: "Kernel") -> None:
  k.emit(v_add_f32_e32(v[1], 1.0, v[1]))
  k.emit(s_waitcnt_depctr(simm16=0xffff))
  k.emit(s_nop(15))
  k.emit(v_add_f32_e32(v[1], 1.0, v[1]))

def _chain_body(n: int):
  def _body(k: "Kernel") -> None:
    k.emit(v_add_f32_e32(v[1], 1.0, v[1]))
    for _ in range(n):
      k.emit(s_nop(15))
    k.emit(v_add_f32_e32(v[1], 1.0, v[1]))
  return _body

microbench(name="mb_snop_15_chain_n2", category="s_nop")(_chain_body(2))
microbench(name="mb_snop_15_chain_n3", category="s_nop")(_chain_body(3))
microbench(name="mb_snop_15_chain_n5", category="s_nop")(_chain_body(5))
microbench(name="mb_snop_15_chain_n8", category="s_nop")(_chain_body(8))

@microbench(name="mb_snop_15_chain_after_vmcnt_n3", category="s_nop")
def _mb_snop_15_chain_after_vmcnt_n3(k: "Kernel") -> None:
  # vmcnt drain → 3 s_nop(15) → v_add. Per PROBE_FINDINGS B, first nop is 20
  # (post-drain), middle nops 16, last nop 20 (before VALU).
  k.emit(global_load_b32(v[2], v[0], saddr=s[0:1]))
  k.emit(s_waitcnt_vmcnt(sdst=NULL, simm16=0))
  for _ in range(3):
    k.emit(s_nop(15))
  k.emit(v_add_f32_e32(v[1], 1.0, v[1]))

@microbench(name="mb_snop_mixed_values", category="s_nop")
def _mb_snop_mixed_values(k: "Kernel") -> None:
  # s_nop(0); s_nop(15); s_nop(5); v_add — tests predecessor dependence
  # within a mixed-N chain (expected dts: 1, 16, 6, 1).
  k.emit(v_add_f32_e32(v[1], 1.0, v[1]))
  k.emit(s_nop(0))
  k.emit(s_nop(15))
  k.emit(s_nop(5))
  k.emit(v_add_f32_e32(v[1], 1.0, v[1]))

# ═════════════════════════════════════════════════════════════════════════════
# A.8 — s_waitcnt variants (6 kernels)
# ═════════════════════════════════════════════════════════════════════════════

@microbench(name="mb_waitcnt_vmcnt_null", category="waitcnt",
            prologue=_prologue_no_load)
def _mb_waitcnt_vmcnt_null(k: "Kernel") -> None:
  # Real global_load + s_waitcnt_vmcnt(0) — measure the full DRAM-drain cost.
  k.emit(global_load_b32(v[2], v[0], saddr=s[0:1]))
  k.emit(s_waitcnt_vmcnt(sdst=NULL, simm16=0))
  k.emit(v_add_f32_e32(v[1], 1.0, v[1]))

@microbench(name="mb_waitcnt_vmcnt_nonzero", category="waitcnt",
            prologue=_prologue_no_load)
def _mb_waitcnt_vmcnt_nonzero(k: "Kernel") -> None:
  # 8 outstanding global_loads, then s_waitcnt_vmcnt(7) — partial drain
  # (wait until only 7 are still in flight, i.e. 1 completes).
  for i in range(8):
    k.emit(global_load_b32(v[2+i], v[0], saddr=s[0:1]))
  k.emit(s_waitcnt_vmcnt(sdst=NULL, simm16=7))
  k.emit(v_add_f32_e32(v[1], 1.0, v[1]))
  # Drain remaining loads before epilogue so v[2..9] retire cleanly.
  k.emit(s_waitcnt_vmcnt(sdst=NULL, simm16=0))

@microbench(name="mb_waitcnt_lgkmcnt_null_smem", category="waitcnt",
            prologue=_prologue_no_load)
def _mb_waitcnt_lgkmcnt_null_smem(k: "Kernel") -> None:
  # s_load_b64 into a scratch pair, then drain lgkmcnt. Measures SMEM latency.
  k.emit(s_load_b64(s[4:5], s[0:1], soffset=NULL))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))
  k.emit(v_add_f32_e32(v[1], 1.0, v[1]))

@microbench(name="mb_waitcnt_lgkmcnt_null_lds", category="waitcnt",
            prologue=_prologue_lds, lds_size=1024)
def _mb_waitcnt_lgkmcnt_null_lds(k: "Kernel") -> None:
  # Write LDS then read it back, drain lgkmcnt.
  k.emit(ds_store_b32(addr=v[2], data0=v[1]))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))
  k.emit(ds_load_b32(vdst=v[3], addr=v[2]))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))
  k.emit(v_add_f32_e32(v[1], 1.0, v[3]))

@microbench(name="mb_waitcnt_depctr_4095", category="waitcnt")
def _mb_waitcnt_depctr_4095(k: "Kernel") -> None:
  # v_add; s_waitcnt_depctr(0xffff == 4095 non-default bits clear); v_add
  # Measures the depctr reset cost (C1 mismatch).
  k.emit(v_add_f32_e32(v[1], 1.0, v[1]))
  k.emit(s_waitcnt_depctr(simm16=0xffff))
  k.emit(v_add_f32_e32(v[1], 1.0, v[1]))

@microbench(name="mb_waitcnt_empty_barrier", category="waitcnt")
def _mb_waitcnt_empty_barrier(k: "Kernel") -> None:
  # s_waitcnt(0) with nothing pending — baseline barrier cost.
  k.emit(v_add_f32_e32(v[1], 1.0, v[1]))
  k.emit(s_waitcnt(simm16=0))
  k.emit(v_add_f32_e32(v[1], 1.0, v[1]))

# ═════════════════════════════════════════════════════════════════════════════
# A.9 — LDS opcodes (8 kernels)
# ═════════════════════════════════════════════════════════════════════════════
# All LDS kernels allocate a 1 KB LDS slab via extra["lds_size"].
# v[2] holds 4*tid as the LDS byte offset (seeded by _prologue_lds).
# b64 and b128 stores reuse v[3..6] as the payload; loads write into v[4..7].

@microbench(name="mb_lds_store_b32_n1", category="lds",
            prologue=_prologue_lds, lds_size=1024)
def _mb_lds_store_b32_n1(k: "Kernel") -> None:
  k.emit(ds_store_b32(addr=v[2], data0=v[1]))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))

@microbench(name="mb_lds_store_b64_n1", category="lds",
            prologue=_prologue_lds, lds_size=1024)
def _mb_lds_store_b64_n1(k: "Kernel") -> None:
  # Need two contiguous VGPRs for b64 payload; seed them first.
  k.emit(v_mov_b32_e32(v[3], 1.0))
  k.emit(v_mov_b32_e32(v[4], 2.0))
  k.emit(ds_store_b64(addr=v[2], data0=v[3:4]))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))

@microbench(name="mb_lds_store_b128_n1", category="lds",
            prologue=_prologue_lds, lds_size=1024)
def _mb_lds_store_b128_n1(k: "Kernel") -> None:
  # ds_store_b128 takes 4 contiguous VGPRs as the payload.
  _seed_v_many(k, 3, 4)
  k.emit(ds_store_b128(addr=v[2], data0=v[3:6]))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))

@microbench(name="mb_lds_load_b32_n1", category="lds",
            prologue=_prologue_lds, lds_size=1024)
def _mb_lds_load_b32_n1(k: "Kernel") -> None:
  # Pre-populate LDS so the load reads a known value.
  k.emit(ds_store_b32(addr=v[2], data0=v[1]))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))
  k.emit(ds_load_b32(vdst=v[4], addr=v[2]))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))
  k.emit(v_add_f32_e32(v[1], 1.0, v[4]))

@microbench(name="mb_lds_load_b64_n1", category="lds",
            prologue=_prologue_lds, lds_size=1024)
def _mb_lds_load_b64_n1(k: "Kernel") -> None:
  k.emit(v_mov_b32_e32(v[3], 1.0))
  k.emit(v_mov_b32_e32(v[4], 2.0))
  k.emit(ds_store_b64(addr=v[2], data0=v[3:4]))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))
  k.emit(ds_load_b64(vdst=v[4:5], addr=v[2]))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))
  k.emit(v_add_f32_e32(v[1], 1.0, v[4]))

@microbench(name="mb_lds_load_b128_n1", category="lds",
            prologue=_prologue_lds, lds_size=1024)
def _mb_lds_load_b128_n1(k: "Kernel") -> None:
  _seed_v_many(k, 3, 4)
  k.emit(ds_store_b128(addr=v[2], data0=v[3:6]))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))
  k.emit(ds_load_b128(vdst=v[7:10], addr=v[2]))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))
  k.emit(v_add_f32_e32(v[1], 1.0, v[7]))

@microbench(name="mb_lds_store_then_valu_forward", category="lds",
            prologue=_prologue_lds, lds_size=1024)
def _mb_lds_store_then_valu_forward(k: "Kernel") -> None:
  # After ds_store_b32 v[1], v_add reads v[1] right away — probes the
  # LDS_WR_FORWARD path (the store doesn't block the VALU from v[1]).
  k.emit(ds_store_b32(addr=v[2], data0=v[1]))
  k.emit(v_add_f32_e32(v[1], 1.0, v[1]))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))

@microbench(name="mb_lds_load_then_valu_forward", category="lds",
            prologue=_prologue_lds, lds_size=1024)
def _mb_lds_load_then_valu_forward(k: "Kernel") -> None:
  # Pre-seed LDS, then ds_load_b32 into v[4], waitcnt, then v_add consumes
  # v[4] — probes LDS_RD_FORWARD (cost of feeding a fresh LDS load into VALU).
  k.emit(ds_store_b32(addr=v[2], data0=v[1]))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))
  k.emit(ds_load_b32(vdst=v[4], addr=v[2]))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))
  k.emit(v_add_f32_e32(v[1], 1.0, v[4]))

# ═════════════════════════════════════════════════════════════════════════════
# A.10 — global_load / global_store (6 kernels)
# ═════════════════════════════════════════════════════════════════════════════

@microbench(name="mb_vmem_store_b32_isolated", category="vmem",
            prologue=_prologue_no_load)
def _mb_vmem_store_b32_isolated(k: "Kernel") -> None:
  # No prior vmem load — store is "cold". v_add produces v[1], then store.
  k.emit(v_add_f32_e32(v[1], 1.0, v[1]))
  k.emit(global_store_b32(addr=v[0], data=v[1], saddr=s[0:1]))
  # waitcnt so epilogue's own store sees a drained state.
  k.emit(s_waitcnt_vmcnt(sdst=NULL, simm16=0))

@microbench(name="mb_vmem_store_b32_after_load", category="vmem")
def _mb_vmem_store_b32_after_load(k: "Kernel") -> None:
  # The standard prologue already did global_load + wait; v[1] is warm.
  # Body: produce new v[1] then store. Tests the A1/A2 17-vs-21 bimodality.
  k.emit(v_add_f32_e32(v[1], 1.0, v[1]))
  k.emit(global_store_b32(addr=v[0], data=v[1], saddr=s[0:1]))
  k.emit(s_waitcnt_vmcnt(sdst=NULL, simm16=0))

@microbench(name="mb_vmem_store_b32_chain_n4", category="vmem")
def _mb_vmem_store_b32_chain_n4(k: "Kernel") -> None:
  # 4 stores back-to-back with paired VALU producers to different VGPRs.
  # Tests the "A chain hypothesis": first store 21, subsequent stores 1.
  _seed_v_many(k, 10, 4)
  for i in range(4):
    k.emit(v_add_f32_e32(v[10+i], 1.0, v[10+i]))
    k.emit(global_store_b32(addr=v[0], data=v[10+i], saddr=s[0:1]))
  k.emit(s_waitcnt_vmcnt(sdst=NULL, simm16=0))

@microbench(name="mb_vmem_store_b32_spaced_nopK", category="vmem")
def _mb_vmem_store_b32_spaced_nopK(k: "Kernel") -> None:
  # K=8 nops between the producer VALU and the store — spacing curve
  # (expected dt = max(21-K, 4) per taxonomy).
  k.emit(v_add_f32_e32(v[1], 1.0, v[1]))
  for _ in range(8):
    k.emit(s_nop(0))
  k.emit(global_store_b32(addr=v[0], data=v[1], saddr=s[0:1]))
  k.emit(s_waitcnt_vmcnt(sdst=NULL, simm16=0))

@microbench(name="mb_vmem_store_b64_isolated", category="vmem",
            prologue=_prologue_no_load)
def _mb_vmem_store_b64_isolated(k: "Kernel") -> None:
  # Seed v[1:2] as a 64-bit payload.
  k.emit(v_mov_b32_e32(v[1], 1.0))
  k.emit(v_mov_b32_e32(v[2], 2.0))
  k.emit(global_store_b64(addr=v[0], data=v[1:2], saddr=s[0:1]))
  k.emit(s_waitcnt_vmcnt(sdst=NULL, simm16=0))

@microbench(name="mb_vmem_store_b128_isolated", category="vmem",
            prologue=_prologue_no_load)
def _mb_vmem_store_b128_isolated(k: "Kernel") -> None:
  # b128 needs 4 contiguous VGPRs as data; seed v[1..4] then store.
  _seed_v_many(k, 1, 4)
  k.emit(global_store_b128(addr=v[0], data=v[1:4], saddr=s[0:1]))
  k.emit(s_waitcnt_vmcnt(sdst=NULL, simm16=0))
