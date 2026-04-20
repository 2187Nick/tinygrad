# RDNA3 emulator v2 - compiles pcode to UOps executed via tinygrad CPU backend
# Each instruction is compiled to a kernel that operates on buffers:
#   arg=0: sgpr - sgpr[0-127], inline constants[128-255], PC_LO=256, PC_HI=257, SCC=258, SCRATCH_STRIDE=259
#   arg=1: vgpr - vgpr[reg * 32 + lane]
#   arg=2: vmem - base address 0, INDEX offsets directly to host memory
#   arg=3: lds - local data share
#   arg=4: scratch - per-lane scratch memory
from __future__ import annotations
import ctypes, functools, os, re, platform, subprocess, tempfile
from typing import Callable

# Set/restore DAZ+FTZ (denormals-are-zero + flush-to-zero) to match RDNA3 default float mode
# x86: MXCSR bits DAZ(6)+FTZ(15), ARM64: FPCR bit FZ(24)
# Only applied during emulator execution, restored afterward to avoid breaking hypothesis tests
@functools.cache
def _get_ftz_lib():
  machine = platform.machine()
  if machine in ('x86_64', 'AMD64'):
    src = b'''
unsigned int get_fpcr(void){unsigned int m;__asm__ __volatile__("stmxcsr %0":"=m"(m));return m;}
void set_fpcr(unsigned int m){__asm__ __volatile__("ldmxcsr %0"::"m"(m));}
'''
    ftz_bits = 0x8040  # DAZ (bit 6) + FTZ (bit 15)
  elif machine in ('arm64', 'aarch64'):
    src = b'''
unsigned int get_fpcr(void){unsigned long long v;__asm__ __volatile__("mrs %0,fpcr":"=r"(v));return(unsigned int)v;}
void set_fpcr(unsigned int m){unsigned long long v=m;__asm__ __volatile__("msr fpcr,%0"::"r"(v));}
'''
    ftz_bits = 1 << 24  # FZ (bit 24)
  else: return None, 0
  try:
    with tempfile.NamedTemporaryFile(suffix='.so', delete=False) as f:
      subprocess.check_output(['clang', '-shared', '-O2', '-x', 'c', '-', '-o', f.name], input=src)
      lib = ctypes.CDLL(f.name)
      lib.get_fpcr.restype = ctypes.c_uint32
      lib.set_fpcr.argtypes = [ctypes.c_uint32]
      return lib, ftz_bits
  except Exception: return None, 0

class _MXCSRContext:
  """Context manager to set DAZ+FTZ during emulator execution and restore afterward."""
  __slots__ = ('_saved',)
  def __enter__(self):
    lib, ftz_bits = _get_ftz_lib()
    if lib is None: return self
    self._saved = lib.get_fpcr()
    lib.set_fpcr(self._saved | ftz_bits)
    return self
  def __exit__(self, *args):
    lib, _ = _get_ftz_lib()
    if lib is None or not hasattr(self, '_saved'): return
    lib.set_fpcr(self._saved)

from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.device import Buffer, BufferSpec
from tinygrad.runtime.autogen import hsa
from tinygrad.helpers import Context, DEBUG, PROFILE, colored
from tinygrad.engine.realize import get_runner

from tinygrad.renderer.amd import decode_inst
from tinygrad.runtime.autogen.amd.rdna3.str_pcode import PCODE as PCODE_RDNA3
from tinygrad.runtime.autogen.amd.rdna4.str_pcode import PCODE as PCODE_RDNA4
from tinygrad.runtime.autogen.amd.cdna.str_pcode import PCODE as PCODE_CDNA
from tinygrad.runtime.autogen.amd.rdna3 import ins as ir3
from tinygrad.runtime.autogen.amd.rdna4 import ins as ir4
from tinygrad.runtime.autogen.amd.cdna import ins as irc
from tinygrad.renderer.amd.dsl import VCC_LO, EXEC_LO, SCC, ttmp
from tinygrad.runtime.autogen.amd.common import Fmt, OpType
from test.mockgpu.amd.pcode import parse_block, _FUNCS, _set_bits, _val_to_bits

MASK32 = 0xFFFFFFFF

# ═══════════════════════════════════════════════════════════════════════════════
# SQTT TRACE COLLECTION
# ═══════════════════════════════════════════════════════════════════════════════

# Global trace storage: populated by run_asm as raw SQTT blobs, consumed by amdgpu.py
sqtt_traces: list[bytes] = []
# Cycle counts per kernel: populated alongside sqtt_traces, consumed by beam search POC
sqtt_cycle_counts: list[int] = []

# Encoder primitives
from tinygrad.renderer.amd.sqtt import (
  _build_decode_tables, PACKET_TYPES_RDNA3, LAYOUT_HEADER, WAVESTART, WAVEEND,
  INST, IMMEDIATE, VALUINST, TS_DELTA_SHORT, TS_DELTA_S5_W3, REG, SNAPSHOT, TS_WAVE_STATE, InstOp)

_NIB_COUNTS: dict = {cls: nc for _, (cls, nc, *_) in _build_decode_tables(PACKET_TYPES_RDNA3)[0].items()}

def _encode_raw(pkt_cls, **kwargs) -> tuple[int, int]:
  raw = pkt_cls.encoding.default
  for k, v in kwargs.items(): raw = pkt_cls.__dict__[k].set(raw, v)
  return raw, _NIB_COUNTS[pkt_cls]

def _emit_nibbles(nibbles: list[int], pkt_cls, **kwargs):
  raw, nc = _encode_raw(pkt_cls, **kwargs)
  for i in range(nc): nibbles.append((raw >> (i * 4)) & 0xF)

def _emit_with_delta(nibbles: list[int], pkt_cls, delta: int, **kwargs):
  # WAVESTART has 2-bit delta (max 3), all others have 3-bit delta (max 7)
  max_d = 3 if pkt_cls is WAVESTART else 7
  if delta > max_d:
    excess = delta - max_d
    delta = max_d
    while excess >= 8:
      take = min(excess, 23)
      _emit_nibbles(nibbles, TS_DELTA_SHORT, delta=take - 8)
      excess -= take
    if excess > 0: _emit_nibbles(nibbles, TS_DELTA_S5_W3, delta=excess)
  _emit_nibbles(nibbles, pkt_cls, delta=delta, **kwargs)

def _nibbles_to_bytes(nibbles: list[int]) -> bytes:
  result = bytearray()
  for i in range(0, len(nibbles), 2): result.append(nibbles[i] | ((nibbles[i + 1] if i + 1 < len(nibbles) else 0) << 4))
  return bytes(result)

# ═══════════════════════════════════════════════════════════════════════════════
# SQTT TIMING MODEL — Cycle-accurate SQ scheduling simulation for RDNA3
# ═══════════════════════════════════════════════════════════════════════════════

# Debug trace: when SQTT_DEBUG=1, collect per-instruction diagnostic info accessible via _sqtt_debug_log
_SQTT_DEBUG = int(os.environ.get("SQTT_DEBUG", "0"))
_sqtt_debug_log: list[dict] = []  # populated by _simulate_sq_timing when _SQTT_DEBUG=1

# SIMD-arbiter shadow telemetry. When SIMD_ARB_SHADOW=1, each call to
# `_simulate_sq_timing` appends one dict summarizing how often the arbiter's
# per-SIMD VALU port_avail would have forced a stall beyond the current
# heuristics' output. Read by analysis scripts; never changes behaviour.
# When SIMD_ARB_SHADOW=2, additionally records one dict PER VALU issue into
# `_simd_arb_shadow_events` so analysis scripts can correlate shadow-mode
# queue-pressure events against MODAL mismatches in rigorous_hw_test.py.
# Analysis scripts align shadow events to emu_traces by (wave_id, stamp_cycle).
_SIMD_ARB_SHADOW = int(os.environ.get("SIMD_ARB_SHADOW", "0"))
_simd_arb_shadow_log: list[dict] = []
_simd_arb_shadow_events: list[dict] = []

# Latency constants (in SQ clock cycles) — tuned against GFX1100 SQTT traces.
# Source of truth: `test/mockgpu/amd/sq_timing/constants.py::TimingConstants`.
# These module-level `_XXX` aliases are kept for backward compatibility with
# existing references in `_simulate_sq_timing`; every value matches
# CONST.<FIELD> exactly (see EMU_REWRITE_DESIGN.md §3, §5 Step 1).
from test.mockgpu.amd.sq_timing.constants import CONST
from test.mockgpu.amd.sq_timing.ib_fetch import IbFetch
from test.mockgpu.amd.sq_timing.lds import LdsPipe
from test.mockgpu.amd.sq_timing.scalar import ScalarPipe
from test.mockgpu.amd.sq_timing.sgpr import SgprScoreboard
from test.mockgpu.amd.sq_timing.simd_arbiter import SimdArbiter
from test.mockgpu.amd.sq_timing.trans import TransPipe
from test.mockgpu.amd.sq_timing.valu import VAluPipe
from test.mockgpu.amd.sq_timing.vmem import VmemPipe
_LDS_RD_LATENCY = CONST.LDS_RD_LATENCY
_LDS_WR_LATENCY = CONST.LDS_WR_LATENCY
_SMEM_LATENCY = CONST.SMEM_LATENCY
_VMEM_LATENCY = CONST.VMEM_LATENCY
_BARRIER_FROM_LAST = CONST.BARRIER_FROM_LAST
_LDS_SERVICE_COST = CONST.LDS_SERVICE_COST
_VALU_DS_WR_FORWARD = CONST.VALU_DS_WR_FORWARD
_VALU_DS_RD_FORWARD = CONST.VALU_DS_RD_FORWARD
_VALU_VMEM_WR_FORWARD = CONST.VALU_VMEM_WR_FORWARD
_VALU_VMEM_WR_BYPASS = CONST.VALU_VMEM_WR_BYPASS
_VALU_VMEM_ADDR_FORWARD = CONST.VALU_VMEM_ADDR_FORWARD
_VALU_VMEM_RD_FORWARD = CONST.VALU_VMEM_RD_FORWARD
_VALU_VMEM_RD_BYPASS = CONST.VALU_VMEM_RD_BYPASS
_VMEM_DRAIN_CYCLES = CONST.VMEM_DRAIN_CYCLES
_VMEM_EXEC_MIN = CONST.VMEM_EXEC_MIN
_TRANS_PIPELINE_LATENCY = CONST.TRANS_PIPELINE_LATENCY
_TRANS_PIPELINE_LATENCY_SQRT = CONST.TRANS_PIPELINE_LATENCY_SQRT
_SGPR_LATENCY = CONST.SGPR_LATENCY
_CNDMASK_SGPR_LATENCY = CONST.CNDMASK_SGPR_LATENCY
_CMP_LIT_WB_LATENCY = CONST.CMP_LIT_WB_LATENCY
_SGPR_COMMIT_GAP = CONST.SGPR_COMMIT_GAP
_WAVESTART_GAP = CONST.WAVESTART_GAP
_FIRST_INST_GAP = CONST.FIRST_INST_GAP
# S_DELAY_ALU INSTID → base stall cycles for single-wave (0=NO_DEP, 1-4=VALU_DEP, 5-7=TRANS32_DEP, 8=FMA_ACCUM, 9-11=SALU_CYCLE)
# VALU deps (1-4): RDNA3 VALU pipeline = 5 cycles. With N waves round-robin hides (N-1) cycles, so stall = max(0, base - (N-1)).
# TRANS32 deps (5-7): 0 — trans ALU runs in parallel with VALU. Pipeline occupancy enforced by trans_pipe_avail. Register deps by s_waitcnt_depctr.
_INSTID_BASE_STALLS = (0, 4, 3, 2, 1, 0, 0, 0, 1, 1, 2, 3)
_TRANS_PIPE_CYCLES = CONST.TRANS_PIPE_CYCLES
_VOPD_PIPE_CYCLES = CONST.VOPD_PIPE_CYCLES
_EXEC_WRITE_LATENCY = CONST.EXEC_WRITE_LATENCY
_LDS_B128_EXTRA = CONST.LDS_B128_EXTRA
_LDS_B128_VGPR_STAGGER = CONST.LDS_B128_VGPR_STAGGER
_LDS_B128_RD_SERVICE = CONST.LDS_B128_RD_SERVICE
def _instid_stall(instid: int, n_waves: int) -> int:
  base = _INSTID_BASE_STALLS[min(instid, 11)]
  return max(0, base - (n_waves - 1)) if 1 <= instid <= 4 else base

# Per-instruction SQ issue cost: multi-cycle ops occupy the execution unit for N cycles.
# Suffix number in InstOp name = issue cost — but ONLY for VALU ops. Memory op suffixes
# (SGMEM_WR_2, FLAT_WR_3, LDS_WR_2, etc.) are opcode markers, not cycle counts.
_INSTOP_ISSUE_COST: dict[InstOp, int] = {}
_COST_SUFFIX_PREFIXES = ('VALU',)  # only parse suffix as cost for VALU* ops
for _op in InstOp:
  _parts = _op.name.rsplit('_', 1)
  if (len(_parts) == 2 and _parts[1].isdigit() and int(_parts[1]) > 1
      and _op.name.startswith(_COST_SUFFIX_PREFIXES)):
    _INSTOP_ISSUE_COST[_op] = int(_parts[1])
del _op, _parts

def _get_issue_cost(pkt_cls, kwargs) -> int:
  if pkt_cls is INST and 'op' in kwargs:
    op = kwargs['op']
    # Trans instructions (VALUT_4) have 1-cycle SQ issue cost — the 4-cycle pipeline is enforced by trans_pipe_avail
    if op == InstOp.VALUT_4: return 1
    # Scalar branch TAKEN: SQTT records 3cy before next instruction (HW: layernorm [15])
    # Scalar branch NOT-TAKEN: SQTT records 10cy on the branch's own token (HW: probe_branch_cost)
    if op == InstOp.JUMP: return 3
    if op == InstOp.JUMP_NO: return 3
    return _INSTOP_ISSUE_COST.get(op, 1)
  return 1

def _vmem_wr_issue(deadline: int, store_vgprs: int, selffwd_vgprs: int, vgprs_written: int,
                   cndmask_cluster_vgprs: int = 0) -> int:
  """Compute VMEM store issue deadline based on VALU forwarding pattern."""
  width_extra = max(0, store_vgprs - 1)
  if selffwd_vgprs >= store_vgprs and store_vgprs > 1:
    return deadline - 1  # pipeline batch: all data VGPRs from consecutive self-fwd VALUs
  if cndmask_cluster_vgprs >= store_vgprs and store_vgprs > 1:
    return deadline  # cndmask chain (cmp-interleaved) feeds store cleanly — no width_extra/scatter
  if vgprs_written >= store_vgprs:
    return deadline + width_extra  # all data VGPRs from consecutive non-VOPC VALUs
  return deadline + width_extra + 1  # scattered writes (VOPCs broke chain) — extra gather cycle

def _simulate_sq_timing(wave_events: dict[int, list]) -> list[tuple[int, int, type, dict]]:
  # RDNA3 SQ round-robin scheduling: one non-zero-cost instruction per cycle per SIMD
  wave_ids = sorted(wave_events.keys())
  n = len(wave_ids)
  if not n: return []
  if _SQTT_DEBUG: _sqtt_debug_log.clear()

  # Pre-pass: for each wave, for each instruction index, compute the length of the
  # containing "same-reg RAW chain". A chain runs while each consecutive VALU reads
  # a VGPR written by the immediately-preceding VALU (no gap VGPRs). Used to detect
  # LONG chains (HW wave 1+ stall on chain depth ≥ 6 saturates the issue queue)
  # vs SHORT chains (all waves bypass). See mb_f1_valu_fmac_n16 vs mb_valu_add_n4.
  # `raw_chain_L[i][pc]` stores the raw segment length (parallel to `long_raw_chain`),
  # used by the slot-0 exemption in the wave-credit rule.
  long_raw_chain: list[list[bool]] = []
  raw_chain_L: list[list[int]] = []
  wave_valu_count: list[int] = []
  for wid in wave_ids:
    events = wave_events[wid]
    seg_len = [0] * len(events)
    p = 0
    while p < len(events):
      _, _, cat_p, ex_p = events[p]
      if cat_p != 'valu' or not isinstance(ex_p, tuple) or len(ex_p) < 5:
        p += 1; continue
      vw_p = ex_p[3] if len(ex_p) > 4 else ()
      end = p + 1
      prev_w = set(vw_p or ())
      while end < len(events):
        _, _, cat_e, ex_e = events[end]
        if cat_e != 'valu' or not isinstance(ex_e, tuple) or len(ex_e) < 5: break
        vw_e = ex_e[3] if len(ex_e) > 4 else ()
        vr_e = ex_e[4] if len(ex_e) > 4 else ()
        if not (prev_w & set(vr_e or ())): break
        prev_w = set(vw_e or ())
        end += 1
      L = end - p
      for q in range(p, end): seg_len[q] = L
      p = end if end > p else p + 1
    long_raw_chain.append([L >= 6 for L in seg_len])
    raw_chain_L.append(list(seg_len))
    wave_valu_count.append(sum(1 for e in events if e[2] == 'valu'))

  # Per-wave state
  pc = [0] * n
  ready = [0] * n
  at_barrier = [False] * n
  barrier_issue = [0] * n
  wave_done = [False] * n
  dly: list[list[int]] = [[0, -1, 0, -1, 0] for _ in range(n)]  # S_DELAY_ALU state
  valu_ds_wr_deadline = [0] * n   # VALU→DS_WR forwarding
  valu_ds_rd_deadline = [0] * n   # VALU→DS_RD forwarding
  last_vmem_wr_issue = [0] * n    # last VMEM_WR issue cycle (for back-to-back store pipe-reuse)
  # Per-wave VMEM pipeline tracker (EMU_REWRITE_DESIGN §1.6 / §5 Step 4).
  # Owns valu_vmem_wr_deadline / wr_set_time / wr_slow_ext / rd_deadline /
  # vmem_drain_deadline / vm_pend. `_vmem_wr_bypass_active` (cross-wave) still
  # lives in emu.py for this step — it will move to a PeerSnapshot in Step 5+.
  vmem = [VmemPipe(CONST) for _ in range(n)]
  # Per-wave trans ALU pipeline tracker (EMU_REWRITE_DESIGN §1.3 / §5 Step 5a).
  # Owns trans_pipe_avail, scalar_after_trans_ready, trans_vgpr_ready, and the
  # valu_pend queue used by s_waitcnt_depctr. Pure state holder — logic stays
  # in emu.py for this step.
  trans = [TransPipe(CONST) for _ in range(n)]
  # Per-wave SGPR scoreboard (EMU_REWRITE_DESIGN §1.5 / §5 Step 5b). Owns
  # sgpr_write_time, sgpr_cmp_lit_read_ready/last_commit/hist, smem_sgpr_ready.
  # Pure state holder — logic (read_stall, pending_nonvcc_drain) stays in emu.py.
  sgpr = [SgprScoreboard(CONST) for _ in range(n)]
  # Per-wave VALU pipeline tracker (EMU_REWRITE_DESIGN §1.2 / §5 Step 5c). Owns
  # valu_issue_hist, vopd_pipe_avail, last_vopd_issue, bank_vopd_write_time,
  # consecutive_single_valu, consecutive_selffwd_vgprs, consecutive_vgprs_written,
  # vgpr_ready, vgpr_slow_fresh_until, vgpr_write_time. Pure state holder.
  valu = [VAluPipe(CONST) for _ in range(n)]
  has_delay_alu = [False] * n     # per-wave: has any s_delay_alu been seen (controls VGPR scoreboard activation)
  # Per-wave ScalarPipe (EMU_REWRITE_DESIGN §1.4 / §5 Step 6). Owns scc_write_time
  # and exec_write_time. Pure state holder — SALU/cbranch/s_nop cost helpers stay
  # in emu.py for now; Step 7 will land HW-confirmed constant fixes and may then
  # refactor those helpers onto ScalarPipe.
  scal = [ScalarPipe(CONST) for _ in range(n)]
  # Per-wave IB-fetch tracker (EMU_REWRITE_DESIGN.md §1.8 / §5 Step 2). Owns last_drain_stamp and
  # had_drain_nop. Reset at the top of each `_drain_zero_cost(i)` call to preserve scalar-per-call
  # semantics of the pre-refactor code (pure refactor — zero behaviour change).
  ib = [IbFetch(CONST) for _ in range(n)]
  # CU-shared LDS pipeline + per-wave lgkm queue (EMU_REWRITE_DESIGN §1.7 / §5 Step 3).
  # Owns cu_lds_available, cu_lds_last_was_write, cu_lds_rd_available, lgkm_pend[*].
  # Pure refactor — b128 VGPR-stagger logic stays in this file (VGPR state belongs
  # to VAluPipe / Step 5); LdsPipe only reports `is_serialized`.
  lds = LdsPipe(n, CONST)
  burst_wave = -1             # VALU burst: wave in consecutive VALU sequence
  burst_exclusive = False     # True when burst started with no other wave recently active (prevents false bursts during interleaving)
  prev_issue_cycle = -1       # global SIMD scheduling: last non-zero-cost issue cycle
  prev_wave = -1              # last wave that issued a non-zero-cost instruction
  # Shared-SIMD VALU issue port (GPUOpen / Seb-V): when waves share a SIMD, VALU issues serialize 1 wave/cycle.
  # For n==2 workgroups, HW typically places both waves on the same SIMD (16 wave slots, small wg fits).
  simd_valu_avail = 0
  # CU-shared SIMD VALU arbiter (shadow-wire — Step 2 of the arbiter refactor).
  # State is always updated on VALU issue; `port_avail` is never read back to mutate
  # issue_cycle, so there is zero behaviour change. When SIMD_ARB_SHADOW=1 the
  # would-bump telemetry is accumulated and appended to `_simd_arb_shadow_log`.
  arbiter = SimdArbiter(n, CONST)
  _arb_would_stall_cy = 0
  _arb_would_stall_count = 0

  timed: list[tuple[int, int, type, dict]] = []

  def _vmem_wr_bypass_active(i: int) -> bool:
    """Check if inter-wave VMEM forwarding bypass applies for wave i's store.
    Bypass fires when the SQ has overlapping work: either another wave's active VMEM drain
    provides pipeline overlap, or another wave is schedulable before our reduced deadline.

    NOTE: reads peer-wave VmemPipe instances via the `vmem` list. This function
    stays here (not on VmemPipe) for Step 4 — moving it requires a PeerSnapshot
    that also exposes pc/at_barrier/wave_done/ready (design §4.3, Step 5+)."""
    if vmem[i].wr_set_time == 0: return False
    reduced = vmem[i].wr_deadline - _VALU_VMEM_WR_BYPASS
    for j in range(n):
      if j == i or at_barrier[j]: continue
      # Active VMEM drain from a DONE wave provides pipeline overlap (wave j finished & queued VMEM)
      if wave_done[j] and vmem[j].drain_deadline >= reduced: return True
      if wave_done[j]: continue
      if pc[j] >= len(wave_events[wave_ids[j]]): continue
      _, _, jcat, _ = wave_events[wave_ids[j]][pc[j]]
      if jcat == 'waveend': continue
      j_eff = ready[j]
      if jcat == 'vmem_wr': j_eff = max(j_eff, vmem[j].wr_deadline)
      elif jcat == 'ds_wr': j_eff = max(j_eff, valu_ds_wr_deadline[j])
      elif jcat == 'ds_rd': j_eff = max(j_eff, valu_ds_rd_deadline[j])
      elif jcat == 'vmem_rd': j_eff = max(j_eff, vmem[j].rd_deadline)
      if j_eff <= reduced: return True
    return False

  def _vmem_rd_bypass_active(i: int) -> bool:
    """Inter-wave VMEM read forwarding bypass (22→18 cycles).
    HW: 2-wave probe_branch_cost keeps 22cy forward; 16+ wave microbenches drop to 18cy.
    The SQ overlaps address-setup with another wave's work — fires when the simulation
    has ≥4 waves running (the threshold that distinguishes small probes from many-wave
    kernels)."""
    if vmem[i].rd_deadline == 0: return False
    if n < 4: return False
    # Any peer wave that isn't barrier-blocked or done participates
    for j in range(n):
      if j == i or at_barrier[j] or wave_done[j]: continue
      if pc[j] >= len(wave_events[wave_ids[j]]): continue
      return True
    return False

  # Phase 1: Emit WAVESTART events at fixed offsets
  for i in range(n):
    wid = wave_ids[i]
    events = wave_events[wid]
    if events and events[0][2] == 'wavestart':
      ws_time = 1 + i * _WAVESTART_GAP
      timed.append((ws_time, wid, events[0][0], events[0][1]))
      pc[i] = 1
      ready[i] = ws_time + _FIRST_INST_GAP
    else:
      ready[i] = 1

  def _drain_zero_cost(i: int):
    nonlocal burst_wave, burst_exclusive
    wid = wave_ids[i]
    events = wave_events[wid]
    # Track the stamp of the last drain event (waitcnt/depctr/nop).
    # When nop follows a drain event, it starts at that stamp (no 1-cycle gap).
    # When nop follows a non-drain event (VALU/etc), normal gap applies.
    # Per EMU_REWRITE_DESIGN §1.8 / §5 Step 2: state lives on per-wave IbFetch;
    # reset per-call preserves the pre-refactor scalar-per-call semantics.
    ib[i].reset_drain()
    ib[i].clear_nop_chain()  # nop preceded by drain → needs IB resume +1 after loop
    while pc[i] < len(events):
      pkt_cls, kwargs, cat, extra = events[pc[i]]
      if cat == 'delay_alu':
        simm16 = extra
        ds = dly[i]
        ds[1] = ds[0] + 1
        ds[2] = simm16 & 0xF  # raw instid0 (resolved to stall at application time)
        instid1 = (simm16 >> 7) & 0xF
        ds[3] = ds[0] + ((simm16 >> 4) & 0x7) + 1 if instid1 else -1
        ds[4] = instid1  # raw instid1
        pc[i] += 1
        continue
      if cat == 'waitcnt':
        simm16 = extra
        lgkm_th = (simm16 >> 4) & 0x3f
        vm_th = (simm16 >> 10) & 0x3f
        _initial_ready = ready[i]
        stall_until = ready[i]
        # Trans→scalar stall: scalar path waits for trans pipeline to clear
        stall_until = max(stall_until, trans[i].scalar_ready)
        # lgkmcnt: stall until at most lgkm_th ops are still pending
        sl = sorted(lds.lgkm_pending(i))
        if lgkm_th < len(sl):
          stall_until = max(stall_until, sl[len(sl) - lgkm_th - 1])
        # vmcnt: stall until at most vm_th ops are still pending
        sv = sorted(vmem[i].vm_pending())
        if vm_th < len(sv):
          stall_until = max(stall_until, sv[len(sv) - vm_th - 1])
        # Empty/no-stall s_waitcnt still costs ~3cy on HW (scalar-pipe decode + SQ handoff).
        # mb_waitcnt_empty_barrier measures dt=3 from prev VALU to empty s_waitcnt.
        # Apply only when neither lgkm nor vm actually stalled (and trans pipe clean).
        if stall_until == _initial_ready:
          stall_until = _initial_ready + 2  # +3cy total from prev dispatch (ready adds +1 after)
        timed.append((stall_until, wid, pkt_cls, kwargs))
        ready[i] = stall_until + 1  # waitcnt occupies the stall_until cycle; next issue at +1
        ib[i].set_drain(stall_until)  # nop after waitcnt starts at stall_until (no +1 gap)
        scal[i].mark_drain()  # first s_cbranch_scc after waitcnt drain pays +1cy (probe_branch_cost [7])
        # First cndmask chain after vmcnt drain pays SGPR bank port pressure on 3rd-SGPR-cndmask.
        # probe_sgpr_cmps [16] (1st chain) = {2, 5}; [31] (2nd chain, after s_nop drain) = {1, 1}.
        if vm_th < len(sv): sgpr[i].mark_first_cndmask_chain_pending()
        # A drain-effective waitcnt absorbs the post-depctr cmp_lit phase-offset (+3cy) —
        # chain-2 in exp_chain [53-57] has a 719cy vmcnt wait and HW A[0] is at
        # write_time+3 (no +3 offset left). `phase_shift_armed` stays set so the chain still
        # uses GAP=1 once cmp_lit reactivates it.
        if stall_until > _initial_ready:
          sgpr[i].set_next_cmp_lit_phase_offset(0)
        pc[i] += 1
        # Prune completed ops
        lds.prune(i, stall_until)
        vmem[i].prune(stall_until)
        sgpr[i].replace_smem_ready({r: c for r, c in sgpr[i].smem_ready_map().items() if c > stall_until})
        continue
      if cat == 'depctr':
        # s_waitcnt_depctr: stall until pending multi-cycle VALU (transcendental) ops complete
        _initial_ready = ready[i]
        stall_until = ready[i]
        if trans[i].valu_pending():
          stall_until = max(stall_until, max(trans[i].valu_pending()))
        # Empty/no-stall s_waitcnt_depctr still costs ~3cy on HW (scalar-pipe decode overhead).
        # mb_waitcnt_depctr_4095 measures dt=3 from prev VALU when there's no trans pending.
        if stall_until == _initial_ready:
          stall_until = _initial_ready + 2
        timed.append((stall_until, wid, pkt_cls, kwargs))
        ready[i] = stall_until + 1
        ib[i].set_drain(stall_until)
        pc[i] += 1
        trans[i].prune_valu_pend(stall_until)
        # depctr drains the LIT v_cmp completion buffer — subsequent chains start fresh.
        # Phase shift: the next cmp_lit chain's A[n] gets +3cy offset (scalar pipe phase
        # left in a different state by depctr). HW exp_chain [33-36] shows the first
        # cndmask after a depctr-drained cmp chain pays ~3cy more than the [12-15] variant
        # where the cmp chain followed a VALU prefix.
        sgpr[i].clear_cmp_lit_hist()
        sgpr[i].set_cmp_lit_last_commit(0)
        sgpr[i].clear_cmp_lit_read_ready()
        sgpr[i].set_next_cmp_lit_phase_offset(3)
        # Arm the phase-shifted chain flag. Survives subsequent waitcnt drain
        # (drain clears the +3 offset but leaves GAP=1 applicable — HW exp_chain
        # chain-2 [53-57] after 719cy vmcnt wait).
        sgpr[i].set_phase_shift_armed(True)
        continue
      if cat == 'nop':
        # s_nop(N): stalls IB for N+1 cycles.
        nop_cycles = (extra + 1) if extra is not None else 1
        # After drain event: nop starts at drain stamp (no gap). After non-drain: normal gap.
        if ib[i].last_drain_stamp >= 0:
          nop_start = max(ib[i].last_drain_stamp, vmem[i].drain_deadline)
        else:
          nop_start = max(ready[i], vmem[i].drain_deadline)
        # SGPR drain: s_nop stalls until all pending SGPR write-backs complete
        if sgpr[i].write_time_map():
          sgpr_drain = max(t + _SGPR_LATENCY + 1 for t in sgpr[i].write_time_map().values())
          nop_start = max(nop_start, sgpr_drain)
        if nop_cycles > 1:
          # Peek at next inst to detect "last nop in a chain" (next is NOT a nop) — HW adds +4cy
          # to the last nop's stamp in a drain-path chain (probe_sgpr_cmps [21-23] unanimous both waves).
          _next_pc = pc[i] + 1
          _is_last_nop_in_chain = True
          if _next_pc < len(wave_events[wid]):
            _, _, _next_cat, _ = wave_events[wid][_next_pc]
            if _next_cat == 'nop': _is_last_nop_in_chain = False
          if ib[i].last_drain_stamp >= 0:
            # After drain event: stamp = start + stall_cycles (no overhead); last-in-chain pays +4.
            # If the chain originated from a VALU/SALU (not a real drain), skip the +4 —
            # HW mb_snop_mixed_values [8] shows nop(5) last-in-chain after VALU+nop(0)+nop(15)
            # runs 6cy (not 10cy). probe_sgpr_cmps chain starts from waitcnt → +4 still applies.
            _last_nop_extra = (4 if _is_last_nop_in_chain and not ib[i].chain_is_valu_origin else 0)
            nop_stamp = nop_start + nop_cycles + _last_nop_extra
            ib[i].mark_nop_in_chain()
          else:
            # After non-drain event (VALU/etc): stamp includes +1 overhead
            nop_stamp = nop_start + nop_cycles + 1
            ib[i].mark_nop_in_chain()  # next VALU stamps 1cy after nop (IB resume), same as drain path
          timed.append((nop_stamp, wid, pkt_cls, kwargs))
          ready[i] = nop_stamp
          ib[i].set_drain(nop_stamp)  # chain: next nop starts at this stamp
          # ISA team answer2.md (Bonus): s_nop counts toward all cycle-based forwarding windows.
          _residual = nop_stamp + 5
          vmem[i].cap_wr_deadline(_residual)
          vmem[i].cap_rd_deadline(_residual)
          _wt_cap = _residual - _VALU_VMEM_ADDR_FORWARD
          _vwt = valu[i].vgpr_write_time_map()
          for _r in list(_vwt.keys()):
            if _vwt[_r] > _wt_cap: _vwt[_r] = _wt_cap
          # Prune completed SGPR writes: writes that have fully drained during the nop
          sgpr[i].replace_write_time({r: t for r, t in sgpr[i].write_time_map().items() if t + _SGPR_LATENCY >= nop_stamp})
          # Long nops invalidate burst state: the scheduling gap makes pre-nop burst stale
          if nop_cycles >= 16: burst_wave = -1; burst_exclusive = False
        else:
          # Single-cycle s_nop(0): stamp at nop_start, propagate drain so
          # consecutive s_nop(0)s each advance by 1cy (HW mb_e2_store_after_longnop
          # and mb_e3_extra_waitcnt_nop16_vmov: 16 back-to-back s_nop(0) unanimous
          # dt=1 across all 16 waves; without set_drain each nop reused the
          # previous drain stamp and all stamped at the same cycle).
          nop_stamp = max(nop_start, ready[i])
          # HW stamps s_nop(0) +2cy later when it directly follows a real
          # instruction (not a drain event): mb_snop_0_after_valu and
          # mb_snop_0_after_scalar show HW=3 EMU=1 uniformly across all 16 waves.
          # Only apply when last_drain_stamp == -1 (no prior waitcnt/depctr drain).
          _valu_origin = False
          if ib[i].last_drain_stamp < 0 and pc[i] > 0:
            _prev_cat = wave_events[wid][pc[i] - 1][2]
            if _prev_cat in ('valu', 'salu', 'vmem_wr', 'vmem_rd', 'ds_wr', 'ds_rd'):
              nop_stamp += 2
              _valu_origin = True
          timed.append((nop_stamp, wid, pkt_cls, kwargs))
          ready[i] = nop_stamp + nop_cycles
          ib[i].set_drain(nop_stamp + nop_cycles, from_valu_nop=_valu_origin)
        pc[i] += 1
        continue
      if cat == 'immediate':
        timed.append((ready[i], wid, pkt_cls, kwargs))
        pc[i] += 1
        continue
      if cat == 'clause':
        # S_CLAUSE marks clause start; HW records it 1cy before next instruction (plus [0]→[1])
        timed.append((ready[i], wid, pkt_cls, kwargs))
        ready[i] += 1
        pc[i] += 1
        continue
      break  # hit a non-zero-cost event
    # After draining: nop preceded by drain event needs 1 cycle for IB resume
    # (HW validated: v_cmp/v_add after drain nop chain always has delta=1, not 0)
    ready[i] += ib[i].resume_penalty()

  # Phase 2: Round-robin instruction scheduling
  clock = min(ready) if ready else 1
  rr = 0  # round-robin starting index
  max_iters = sum(len(e) for e in wave_events.values()) * n * 10 + 1000

  for _ in range(max_iters):
    if all(wave_done): break

    # Drain zero-cost events for all active waves
    for i in range(n):
      if not wave_done[i] and not at_barrier[i]: _drain_zero_cost(i)

    # VALU burst: consecutive VALU from same wave gets priority only when burst started exclusively
    # (other wave was stalled). When both waves are interleaving, no burst priority (HW interleaves).
    best, best_cycle = -1, 1 << 62
    if burst_wave >= 0 and burst_exclusive and not wave_done[burst_wave] and not at_barrier[burst_wave] \
        and pc[burst_wave] < len(wave_events[wave_ids[burst_wave]]):
      _, _, bcat, _ = wave_events[wave_ids[burst_wave]][pc[burst_wave]]
      if bcat == 'valu':
        best = burst_wave
        best_cycle = ready[burst_wave]
    if best == -1:
      # Burst ending: other waves must wait until after burst's last issue cycle (SIMD serialization)
      if burst_wave >= 0 and burst_exclusive and prev_issue_cycle >= 0:
        for j in range(n):
          if j != burst_wave: ready[j] = max(ready[j], prev_issue_cycle + 1)
      burst_wave = -1
      burst_exclusive = False
    # Fallback: earliest effective-ready cycle, round-robin tiebreak
    if best == -1:
      for offset in range(n):
        i = (rr + offset) % n
        if wave_done[i] or at_barrier[i]: continue
        if pc[i] >= len(wave_events[wave_ids[i]]): continue
        # VALU forwarding stall
        eff_ready = ready[i]
        _, _, next_cat, next_extra = wave_events[wave_ids[i]][pc[i]]
        if next_cat == 'ds_wr': eff_ready = max(eff_ready, valu_ds_wr_deadline[i])
        elif next_cat == 'ds_rd': eff_ready = max(eff_ready, valu_ds_rd_deadline[i])
        elif next_cat == 'vmem_wr':
          ne_bytes = next_extra[0] if isinstance(next_extra, tuple) else next_extra
          store_vgprs = max(1, (ne_bytes or 4) // 4) if ne_bytes else 1
          wr_eff = _vmem_wr_issue(vmem[i].wr_deadline, store_vgprs,
                                                     valu[i].consecutive_selffwd_vgprs, valu[i].consecutive_vgprs_written,
                                                     valu[i].cndmask_cluster_vgprs)
          if store_vgprs == 1 and _vmem_wr_bypass_active(i):
            wr_eff -= _VALU_VMEM_WR_BYPASS
          eff_ready = max(eff_ready, wr_eff)
          if isinstance(next_extra, tuple) and next_extra[1] is not None:
            addr_wt = valu[i].vgpr_write_time_map().get(next_extra[1], 0)
            if addr_wt > 0: eff_ready = max(eff_ready, addr_wt + _VALU_VMEM_ADDR_FORWARD)
        elif next_cat == 'vmem_rd': eff_ready = max(eff_ready, vmem[i].rd_deadline)
        if eff_ready < best_cycle:
          best_cycle = eff_ready
          best = i

    if best == -1:
      # All waves blocked — check barrier release
      barrier_waves = [i for i in range(n) if at_barrier[i]]
      if not barrier_waves: break
      non_done_non_barrier = any(not wave_done[i] and not at_barrier[i] and pc[i] < len(wave_events[wave_ids[i]]) for i in range(n))
      if non_done_non_barrier: break
      release_cycle = max(barrier_issue[i] for i in barrier_waves) + _BARRIER_FROM_LAST
      for idx, i in enumerate(sorted(barrier_waves)):
        at_barrier[i] = False
        # +2cy post-barrier pipeline refill; 2cy RR stagger per wave (GFX1100 observation)
        ready[i] = release_cycle + idx * 2 + 2
      continue

    i = best
    wid = wave_ids[i]
    pkt_cls, kwargs, cat, extra = wave_events[wid][pc[i]]
    # RDNA3: waves run on independent SIMDs — per-wave issue is not serialized by other waves' clock.
    # Shared resources (LDS port, barrier) have their own trackers (cu_lds_available, at_barrier).
    issue_cycle = ready[i]

    # Extract VALU register info from extra tuple: (is_vopc, sgpr_writes, sgpr_reads, vgpr_writes, vgpr_reads, is_vopd, cond_sgpr, is_vopd_lit, is_cmp_lit, is_cndmask)
    is_vopd_lit = False
    is_cmp_lit = False
    is_cndmask = False
    if isinstance(extra, tuple) and len(extra) == 10:
      is_vopc, sgpr_w_regs, sgpr_r_regs, vgpr_w_regs, vgpr_r_regs, is_vopd, cond_sgpr, is_vopd_lit, is_cmp_lit, is_cndmask = extra
    elif isinstance(extra, tuple) and len(extra) == 9:
      is_vopc, sgpr_w_regs, sgpr_r_regs, vgpr_w_regs, vgpr_r_regs, is_vopd, cond_sgpr, is_vopd_lit, is_cmp_lit = extra
    elif isinstance(extra, tuple) and len(extra) == 8:
      is_vopc, sgpr_w_regs, sgpr_r_regs, vgpr_w_regs, vgpr_r_regs, is_vopd, cond_sgpr, is_vopd_lit = extra
    elif isinstance(extra, tuple) and len(extra) == 7:
      is_vopc, sgpr_w_regs, sgpr_r_regs, vgpr_w_regs, vgpr_r_regs, is_vopd, cond_sgpr = extra
    elif isinstance(extra, tuple) and len(extra) == 6:
      is_vopc, sgpr_w_regs, sgpr_r_regs, vgpr_w_regs, vgpr_r_regs, is_vopd = extra
      cond_sgpr = -1
    elif isinstance(extra, tuple) and len(extra) == 5:
      is_vopc, sgpr_w_regs, sgpr_r_regs, vgpr_w_regs, vgpr_r_regs = extra
      is_vopd, cond_sgpr = False, -1
    elif isinstance(extra, tuple) and len(extra) == 3:
      is_vopc, sgpr_w_regs, sgpr_r_regs = extra
      vgpr_w_regs, vgpr_r_regs, is_vopd, cond_sgpr = (), (), False, -1
    else:
      sgpr_w_regs, sgpr_r_regs, vgpr_w_regs, vgpr_r_regs, is_vopd, cond_sgpr = (), (), (), (), False, -1

    # Apply S_DELAY_ALU stall (time-based for VALU_DEP, fixed for TRANS32_DEP/SALU)
    ds = dly[i]
    ds[0] += 1  # count packet-emitting instructions for INSTSKIP tracking
    def _resolve_delay(instid):
      # VALU_DEP (1-4): time-based stall using actual VALU issue history
      if 1 <= instid <= 4:
        vh = valu[i].issue_hist()
        if instid <= len(vh): return max(0, vh[-instid] + 5 - issue_cycle)
        return 0  # not enough history — no real dependency
      # All other deps (TRANS32_DEP, SALU_CYCLE): use calibrated fixed stalls
      return _instid_stall(instid, n)
    stall = 0
    if ds[1] == ds[0]:
      has_delay_alu[i] = True
      stall = max(stall, _resolve_delay(ds[2]))
      ds[1] = -1
    if ds[3] == ds[0]:
      has_delay_alu[i] = True
      stall = max(stall, _resolve_delay(ds[4]))
      ds[3] = -1
    issue_cycle += stall

    # SGPR write-to-read dependency stall: HW enforces 4-cycle latency without requiring S_DELAY_ALU hints
    # v_cndmask reading non-VCC condition SGPR has enhanced 6-cycle latency + drain for pending SGPR write-backs
    if cat == 'valu' and sgpr_r_regs:
      _is_cndmask_nonvcc = cond_sgpr >= 0 and cond_sgpr != 106  # non-VCC condition read (v_cndmask_b32_e64)
      _cmp_lit_rr = sgpr[i].cmp_lit_read_ready_map()
      _swt = sgpr[i].write_time_map()
      _smem_rr = sgpr[i].smem_ready_map()
      # VOPD MOV-only reads SGPRs via a late operand-gather stage — effective write-to-read
      # latency is ~2cy (HW mb_vopd_dualmov_sgpr_{pair,chain_n4}), not the 4cy standard latency.
      _is_vopd_mov_only = kwargs.get('vopd_mov_only', False)
      _mov_only_latency = 2
      # Immediate-predecessor bypass: when a cndmask (or other VALU) reads an SGPR written
      # by the very previous instruction (issued at ready[i]-1), HW forwards at +1cy rather
      # than the full SGPR_LATENCY. HW mb_vcmp_interleave_cndmask shows cmp→cndmask pairs
      # (same SGPR) at dt=1 each. Applies only to non-LIT cmp writes (LIT uses the
      # completion buffer path) and when we're not in a phase-shifted chain.
      _pre_stall_issue = issue_cycle
      for r in sgpr_r_regs:
        # VCC (r=106) bypasses the LIT completion buffer — HW exp_chain [56] cndmask_b32_e64
        # reading VCC_LO after a VCC-writing v_cmp_lit uses standard SGPR latency (~+4cy),
        # not A[VCC] = I+6cy. Applies only to explicit VCC reads (r==106).
        if r != 106 and r in _cmp_lit_rr:  # LIT v_cmp writer: completion-buffer A[n] overrides standard latency
          issue_cycle = max(issue_cycle, _cmp_lit_rr[r])
        elif r in _swt:
          if _is_cndmask_nonvcc and r == cond_sgpr: lat = _CNDMASK_SGPR_LATENCY
          elif _is_vopd_mov_only: lat = _mov_only_latency
          else: lat = _SGPR_LATENCY
          # Immediate-predecessor bypass: if the SGPR was written by the inst at pre_stall-1
          # (i.e. directly the previous dispatch), HW uses 1cy bypass.
          if (_swt[r] == _pre_stall_issue - 1 and not sgpr[i].in_phase_shifted_chain
              and r not in _cmp_lit_rr):
            lat = min(lat, 1)
          issue_cycle = max(issue_cycle, _swt[r] + lat)
        if r in _smem_rr:
          issue_cycle = max(issue_cycle, _smem_rr[r])
      # v_cndmask condition drain: disabled — HW probe_sgpr_cmps shows gap=4 sufficient without drain
      # 3rd+ SGPR-reading cndmask in a chain reading vanilla (non-cmp_lit) SGPR pays +1cy.
      # HW probe_sgpr_cmps [16] w0=2 w1=5 vs EMU=1. Chain-producing cmps lack LIT so no
      # completion buffer applies, and the 3rd _e64 cndmask hits SGPR bank port contention.
      # Gated on (a) not-phase-shifted (exp_chain [33-36] phase_shifted 4-chain stays 1cy),
      # (b) first chain after vmcnt drain (probe_sgpr_cmps [31] in 2nd chain, after s_nop drain,
      # is clean — bank state has settled). Streak only counts cndmasks with cond_sgpr set (_e64);
      # _e32 reads VCC implicitly (not encoded) → doesn't bump streak, so gate is `>= 2`.
      if (_is_cndmask_nonvcc and sgpr[i].cndmask_streak >= 2 and
          not sgpr[i].in_phase_shifted_chain and sgpr[i].first_cndmask_chain_pending):
        if any(r != 106 and r not in _cmp_lit_rr and r in _swt for r in sgpr_r_regs):
          issue_cycle += 1
          sgpr[i].consume_first_cndmask_chain()

    # EXEC write-to-read dependency: v_cmpx writes EXEC, s_cbranch_execz/nz must wait for EXEC propagation
    # SCC write-to-read dependency: s_cmp writes SCC, s_cbranch_scc0/scc1 must wait 1cy for SCC propagation
    if cat == 'branch' and isinstance(extra, tuple):
      reads_exec, reads_scc = extra
      if reads_exec:
        issue_cycle = max(issue_cycle, scal[i].exec_write_time + _EXEC_WRITE_LATENCY)
      if reads_scc:
        issue_cycle = max(issue_cycle, scal[i].scc_write_time + 2)
        # First s_cbranch_scc after a waitcnt/depctr/nop drain: per-wave arbitration split.
        # HW probe_branch_cost [7] shows W0=8, W1=10 (bimodal). Wave 0 wins the scalar-pipe
        # slot and issues -1cy early relative to the MODAL-median; wave 1+ pays +1cy.
        if scal[i].first_branch_after_drain:
          issue_cycle += -1 if i == 0 else 1
          scal[i].consume_drain_branch()

    # VMEM_WR→VALU pipe contention: HW mb_vmem_store_b32_chain_n4 shows a VALU
    # following a global_store_b32 stamps +8cy after the store, while EMU produces
    # +1cy. The store dispatches through a serialized store-pipe that briefly
    # blocks subsequent VALU issues on the same wave. Evidence: chain_n4 waves 0-15
    # show consistent HW_Δ=8 for post-store v_add; emu had EMU_Δ=1.
    if cat == 'valu':
      issue_cycle = max(issue_cycle, vmem[i].post_wr_valu_ready)

    # VGPR RAW dispatch: SQTT stamps at DISPATCH, not completion. HW back-to-back
    # v_add_f32(v[1],1.0,v[1]) chains measure dt=1cy for wave 0 (mb_valu_add_nN
    # wave 0 unanimous). But waves 1+ pay 4cy extra once the chain is deep enough
    # — mb_valu_add_n16 wave 1-15 unanimous dt=5 from the 2nd add onwards, and
    # mb_e1_valu_add_n8 shows waves 4+ stall. Conservative wave-credit rule:
    # wave i ≥ 1 stalls on RAW chain of depth ≥ 4 (shorter chains preserved;
    # HW mb_valu_add_n1/n2/n4 show all waves dt=1).
    # Long-chain gate: pre-pass flagged positions in same-reg RAW chains of length
    # ≥6. For those, wave 1+ stalls on every RAW continuation (HW mb_valu_add_n16,
    # mb_f1_valu_fmac_n{10,12,16}, mb_e1_valu_add_n{10,12,24} unanimous dt=5).
    # Short chains (≤5) fall back to the backward-depth≥4 rule so n≤4 and n=5
    # short chains remain at dt=1 for all waves.
    _in_long_chain = (i < len(long_raw_chain) and pc[i] < len(long_raw_chain[i])
                      and long_raw_chain[i][pc[i]])
    # Slot-0 exemption + medium-chain drain: HW shows slot-0 waves (indices 0-3,
    # one per SIMD) win dt=1 on chains ≤11 deep — dispatch queue depth lets each
    # SIMD's oldest wave stream without stalling. Non-slot-0 waves on L∈[6,11]
    # stall only the FIRST 4 chain positions (dispatch queue depth ≈4), then drain
    # to dt=1 — see mb_f2_raw_then_vopd / mb_f2_vopd_then_raw (HW pos0-3 dt=5,
    # pos4+ dt=1 for all slot-1..3 waves). For L≥12 the queue saturates and every
    # non-wave-0 wave stalls on every RAW continuation (mb_valu_add_n{16,32}).
    _chain_L = raw_chain_L[i][pc[i]] if (i < len(raw_chain_L) and pc[i] < len(raw_chain_L[i])) else 0
    if cat == 'valu' and vgpr_r_regs and (valu[i].raw_chain_depth >= 4 or _in_long_chain):
      if _chain_L and 6 <= _chain_L <= 11:
        # Mixed kernel (chain embedded in significant non-chain VALU work) → HW drains
        # dispatch queue after 3 stalls (chain_pos 4+ at dt=1). Pure chain kernel
        # (just the chain + preamble) → HW stalls all positions at dt=5. Threshold
        # diff ≥ 5 cleanly separates pure chains (total_valu = L+1, from a single
        # preamble v_mov) from mixed kernels like mb_f2_raw_then_vopd where
        # total_valu = L+10 (8 extra v_movs + 1 VOPD + 1 store). Evidence: HW
        # mb_f2_raw_then_vopd wave 1 chain_pos 1-3 dt=5, chain_pos 4+ dt=1.
        # raw_chain_depth pre-increment: chain_pos N (N≥1) has depth N-1. So
        # stall fires at chain_pos 1-3 (depth 0-2), drains at chain_pos 4 (depth 3).
        if wave_valu_count[i] >= _chain_L + 5:
          _fire_stall = (i >= 4 and valu[i].raw_chain_depth < 3)
        else:
          _fire_stall = (i >= 4)
      elif _chain_L and _chain_L >= 14:
        # Long chain dispatch-queue saturation: HW shows wave 0 streams the first 14
        # chain adds at dt=1 (queue fills with RAW backlog), then stalls at chain_pos 14+
        # (queue slot must free before issue). Waves 1-15 stall on every RAW position
        # EXCEPT chain_pos 13 where the slot-0 wave's imminent queue-fill opens one
        # dispatch slot for a peer wave. See mb_f1_valu_add_n{14,15,18,20,22,28,32},
        # mb_e1_valu_add_n24, mb_f1_valu_{mul,fmac}_n16 — HW unanimous across waves 1-15
        # showing dt=1 at chain_pos 13 (position [18] in trace index) and wave 0
        # diverging to dt=4/5 starting at chain_pos 14 (position [19]).
        # raw_chain_depth is pre-increment: at chain_pos N (N ≥ 1) it equals N-1
        # since the increment happens AFTER this check. So chain_pos 13 → depth 12,
        # chain_pos 14 → depth 13.
        if i == 0:
          _fire_stall = (valu[i].raw_chain_depth >= 13)
        else:
          _fire_stall = (valu[i].raw_chain_depth != 12)
      else:
        _fire_stall = (i > 0)
      if _fire_stall:
        _vwt_raw = valu[i].vgpr_write_time_map()
        _vh_raw = valu[i].issue_hist()
        _last_valu_issue = _vh_raw[-1] if _vh_raw else -1
        # RAW continuation: a source reg was written by the immediately-previous VALU
        if _last_valu_issue > 0 and any(_vwt_raw.get(r, -1) == _last_valu_issue for r in vgpr_r_regs):
          issue_cycle = max(issue_cycle, _last_valu_issue + 5)
    # Enforce trans ALU pipeline occupancy: trans instructions wait for the trans unit to be free
    _is_trans = cat == 'valu' and pkt_cls is INST and kwargs.get('op') == InstOp.VALUT_4
    # Trans-written VGPR readiness: non-trans VALU must wait for full trans writeback (27/31 cycles)
    # Trans→trans uses internal ALU forwarding — only trans_pipe_avail applies (4 cycles)
    # HW stamps VOPD at dispatch (before trans wait); wait is absorbed by subsequent s_waitcnt_depctr.
    # For non-VOPD VALU, stall at dispatch (matches HW). For VOPD, defer to ready[i].
    _trans_read_deadline = 0
    if cat == 'valu' and not _is_trans and vgpr_r_regs:
      tvr = trans[i].vgpr_ready_map()
      for r in vgpr_r_regs:
        if r in tvr: _trans_read_deadline = max(_trans_read_deadline, tvr[r])
    # Integer 32-bit multiply (v_mul_lo_u32 / v_mul_hi_u32 / v_mad_u32_u24 /
    # *_i32) is a 4-cycle pipeline op — HW mb_e4_vmul_lo_u32_n4 and
    # mb_f6_vmul_hi_chain_n4 show dt=4 unanimous across waves 0-5 on chain
    # continuations. Enforce pipe occupancy mirroring trans_pipe_avail.
    _is_int_mul32 = cat == 'valu' and kwargs.get('int_mul32', False)
    if _is_int_mul32:
      issue_cycle = max(issue_cycle, valu[i].int_mul_pipe_avail)
    if _is_trans:
      _pre_trans_issue = issue_cycle
      issue_cycle = max(issue_cycle, trans[i].pipe_avail)
      # Per-wave trans chain stagger: HW mb_f4_{exp,log,rcp,rsq,sqrt}_chain_n8
      # show waves 0-3 dt=4 on chain continuations, waves 4-15 dt=10/14. Only
      # fires for LONG chains (length ≥ 6 via long_raw_chain pre-pass); short
      # chains like mb_trans_exp_n4 (chain=4) keep all waves at dt=4 unanimous.
      if (i >= 4 and trans[i].pipe_avail > 0 and _pre_trans_issue < trans[i].pipe_avail
          and i < len(long_raw_chain) and pc[i] < len(long_raw_chain[i])
          and long_raw_chain[i][pc[i]]):
        issue_cycle += 10
    # Enforce VOPD dual-issue occupancy: consecutive VOPDs need spacing
    _vopd_paid_phase_warmup = False  # tracks whether this VOPD paid the +2cy phase-warmup (for pair-follow-up)
    if is_vopd:
      # Batch C finding: VOPD MOV-only (V_DUAL_MOV_B32 on both lanes) doesn't actually use the
      # vsrc1 slots — the encoding slot is dummy-filled by the assembler. These pipeline at 1cy
      # regardless of producer VOPD's pipe_avail. HW mb_vopd_dualmov_{sgpr_pair,sgpr_chain_n4,
      # lit_pair,all_lit_chain_n4} and mix variants all show 1cy unanimous (16/16 waves).
      _vopd_mov_only = kwargs.get('vopd_mov_only', False)
      # Same-write-set bypass: a self-fwd VOPD following another self-fwd VOPD that wrote
      # the SAME VGPR set reuses the pipe slot at 1cy — the write-back-settling path was
      # already primed (HW mb_vopd_bank_conflict_{src,dst}).
      _curr_writes_vopd = frozenset(vgpr_w_regs) if isinstance(vgpr_w_regs, (tuple, list)) else frozenset()
      _same_write_set = (valu[i].last_vopd_issue >= 0 and _curr_writes_vopd
                         and _curr_writes_vopd == valu[i].last_vopd_writes)
      if _vopd_mov_only and valu[i].last_vopd_issue >= 0:
        issue_cycle = max(issue_cycle, valu[i].last_vopd_issue + 1)
      elif _same_write_set:
        issue_cycle = max(issue_cycle, valu[i].last_vopd_issue + 1)
      else:
        issue_cycle = max(issue_cycle, valu[i].vopd_pipe_avail)
      # VOPD-after-phase-shifted-cndmask-chain floor: HW exp_chain [37], [61] show VOPD
      # following a cndmask chain that consumed a phase-shifted cmp_lit chain pipelines
      # no sooner than last_cndmask_issue + 3. Only fires when in_phase_shifted_chain is
      # active (post-depctr) AND the chain has ≥4 cndmasks — shorter chains ([23-25])
      # pipeline tight at dt=1 (HW exp_chain [26]). Non-phase-shifted chains ([16], [47])
      # also pipeline tighter. paid_warmup is set regardless of whether the floor bumped:
      # the next VOPD uses pipe_gap=2 (warmup-tight) instead of 4 (self-fwd), matching HW [38] dt=2.
      if sgpr[i].in_phase_shifted_chain and valu[i].cndmask_cluster_vgprs >= 4:
        issue_cycle = max(issue_cycle, valu[i].last_cndmask_issue + 3)
        _vopd_paid_phase_warmup = True
      # VGPR bank port pressure rule was removed — HW mb_vopd_indep_n4 and
      # mb_vopd_chain_n4_raw both measure 1cy between VOPDs regardless of whether
      # the second reads a bank the first wrote. The SQ's VGPR forwarding lane
      # handles same-bank read-after-write without an extra cycle.

    # LIT v_cmp SGPR completion buffer (answer.md): depth-2 writer stall.
    # N-th LIT v_cmp in a chain must wait until (n-2)th has propagated (W[n-2] = I[n-2]+5).
    # Skipped in phase-shifted chains (post-depctr) — HW exp_chain [52-55] shows the
    # writer pipeline starts fresh after a depctr drain, so the depth-2 stall doesn't fire
    # until the chain is ≥4 deep. (Current cmp_hist length check approximates this.)
    if is_cmp_lit and not sgpr[i].in_phase_shifted_chain:
      cmp_hist = sgpr[i].cmp_lit_hist()
      if len(cmp_hist) >= 2:
        issue_cycle = max(issue_cycle, cmp_hist[-2] + _CMP_LIT_WB_LATENCY)

    # Apply VALU→DS/VMEM forwarding stall (wave selection already deferred this wave; just clamp)
    if cat == 'ds_wr': issue_cycle = max(issue_cycle, valu_ds_wr_deadline[i])
    elif cat == 'ds_rd': issue_cycle = max(issue_cycle, valu_ds_rd_deadline[i])
    elif cat == 'vmem_wr':
      pre_fwd_cycle = issue_cycle
      vmem_bytes = extra[0] if isinstance(extra, tuple) else extra
      store_vgprs = max(1, (vmem_bytes or 4) // 4) if vmem_bytes else 1
      # Back-to-back store pipe-reuse: once the VMEM store pipe is flowing, the
      # next store can enter 1cy after the prev store issues, independent of
      # VALU→VMEM_WR forwarding deadline. Applies only when the IMMEDIATELY
      # previous event for this wave was itself a vmem_wr (no intervening
      # VALU/waitcnt/etc). HW evidence: mb_f3_store_pair_then_pair / _chain_n4
      # wave 1+ shows 2nd store at dt=1-3 while EMU produced dt=4 (wr_deadline
      # anchored to last VALU+21 blocked pipe-reuse).
      _back_to_back_store = (pc[i] > 0 and wave_events[wid][pc[i] - 1][2] == 'vmem_wr'
                             and last_vmem_wr_issue[i] > 0)
      if _back_to_back_store:
        issue_cycle = max(issue_cycle, last_vmem_wr_issue[i] + 1)
      else:
        wr_deadline = _vmem_wr_issue(vmem[i].wr_deadline, store_vgprs,
                                                       valu[i].consecutive_selffwd_vgprs, valu[i].consecutive_vgprs_written,
                                                       valu[i].cndmask_cluster_vgprs)
        # Inter-wave VMEM bypass: another wave schedulable ≥4cy before deadline → SQ pipelines forwarding (21→17cy)
        if store_vgprs == 1 and _vmem_wr_bypass_active(i):
          wr_deadline -= _VALU_VMEM_WR_BYPASS
        issue_cycle = max(issue_cycle, wr_deadline)
      # Per-operand address VGPR forwarding: recently-written addr VGPRs need extra cycles to reach AGU
      if isinstance(extra, tuple) and extra[1] is not None:
        addr_wt = valu[i].vgpr_write_time_map().get(extra[1], 0)
        if addr_wt > 0: issue_cycle = max(issue_cycle, addr_wt + _VALU_VMEM_ADDR_FORWARD)
      _vmem_fwd_stall = issue_cycle - pre_fwd_cycle
    elif cat == 'vmem_rd':
      pre_fwd_cycle = issue_cycle
      rd_deadline = vmem[i].rd_deadline
      # Inter-wave VMEM_RD bypass: mirrors store bypass. HW 2-wave probe_branch_cost=22cy,
      # HW 16-wave microbench v_shift→gload=18cy. Other waves schedulable ≥4cy early → overlap.
      if _vmem_rd_bypass_active(i):
        rd_deadline -= _VALU_VMEM_RD_BYPASS
      issue_cycle = max(issue_cycle, rd_deadline)
      _vmem_fwd_stall = issue_cycle - pre_fwd_cycle

    # Barrier arrival penalty: non-first waves incur +1 cycle
    if cat == 'barrier' and any(at_barrier):
      issue_cycle += 1

    # Scalar branch NOT-TAKEN: HW stamps SQTT token 7cy after issue; 10cy total until next ready.
    # (Branch TAKEN: 3cy total — next inst redirects, no late stamp.)
    # HW validated: probe_branch_cost w0 s_cmp→branch=8cy (issue+7), branch→v_mov=3cy (cost=10).
    stamp_cycle = issue_cycle
    if pkt_cls is INST and kwargs.get('op') == InstOp.JUMP_NO:
      stamp_cycle = issue_cycle + 7

    # SIMD-arbiter shadow update (non-behavioural). For every VALU issue,
    # record whether the per-SIMD VALU port would have forced a later
    # issue_cycle than the one the heuristics produced, then mark the port
    # busy for 1cy. VOPD is modelled as 1cy on the same port (both lanes of
    # one SIMD). See test/mockgpu/amd/sq_timing/simd_arbiter.py for the
    # heuristics this is intended to eventually subsume.
    if cat == 'valu':
      _arb_simd = SimdArbiter.simd_for_wave(i)
      _arb_pa = arbiter.port_avail(_arb_simd)
      if _arb_pa > issue_cycle:
        _arb_would_stall_cy += (_arb_pa - issue_cycle)
        _arb_would_stall_count += 1
      # Per-event shadow record: peer-wave ready clustering is the signal for
      # genuine queue pressure (simd_arbiter.py dead-end: naive 1cy/SIMD
      # regresses 24K tokens because SQTT stamps at dispatch, which absorbs
      # back-to-back same-SIMD issues that aren't clustered).
      if _SIMD_ARB_SHADOW >= 2:
        _peer_c2 = 0
        _peer_c1 = 0
        for _pj in range(n):
          if _pj == i or at_barrier[_pj] or wave_done[_pj]: continue
          if SimdArbiter.simd_for_wave(_pj) != _arb_simd: continue
          _d = abs(ready[_pj] - issue_cycle)
          if _d <= 2: _peer_c2 += 1
          if _d <= 1: _peer_c1 += 1
        _simd_arb_shadow_events.append({
          "wave_idx": i,            # simulation index (used for simd mapping)
          "wave_id": wid,           # GPU wave ID (used for emu_traces correlation)
          "pc": pc[i],
          "simd": _arb_simd,
          "issue_cycle": issue_cycle,
          "stamp_cycle": stamp_cycle,
          "port_avail_before": _arb_pa,
          "would_stall_cy": max(0, _arb_pa - issue_cycle),
          "peer_cluster_2cy": _peer_c2,
          "peer_cluster_1cy": _peer_c1,
        })
      arbiter.set_port_avail(_arb_simd, issue_cycle + 1)
      arbiter.set_last_issue_cycle(_arb_simd, issue_cycle)

    prev_issue_cycle = issue_cycle
    prev_wave = i
    timed.append((stamp_cycle, wid, pkt_cls, kwargs))

    # Debug trace: collect per-instruction diagnostic data
    if _SQTT_DEBUG:
      dbg = {"wave": i, "pc_idx": pc[i], "cat": cat, "issue_cycle": issue_cycle, "stamp": stamp_cycle,
             "ready": ready[i], "clock": clock}
      if cat == 'valu':
        dbg["sgpr_write_time"] = dict(sgpr[i].write_time_map())
        dbg["vgpr_stall"] = {r: valu[i].vgpr_ready_map().get(r, 0) for r in (vgpr_r_regs or ())}
        dbg["sgpr_r_regs"] = sgpr_r_regs
        dbg["cond_sgpr"] = cond_sgpr
        dbg["has_delay_alu"] = has_delay_alu[i]
        dbg["trans_pipe_avail"] = trans[i].pipe_avail
      if cat == 'vmem_wr':
        dbg["vmem_wr_deadline"] = vmem[i].wr_deadline
        dbg["vmem_wr_set_time"] = vmem[i].wr_set_time
        dbg["vmem_drain_deadline"] = vmem[i].drain_deadline
        dbg["bypass_active"] = store_vgprs == 1 and _vmem_wr_bypass_active(i)
        dbg["valu_hist"] = list(valu[i].issue_hist())
        # Log other waves' state for bypass analysis
        for j in range(n):
          if j == i: continue
          jdone = wave_done[j]
          jpc = pc[j] if pc[j] < len(wave_events[wave_ids[j]]) else -1
          jcat = wave_events[wave_ids[j]][pc[j]][2] if jpc >= 0 else "done"
          dbg[f"w{j}_done"] = jdone
          dbg[f"w{j}_ready"] = ready[j]
          dbg[f"w{j}_cat"] = jcat
          dbg[f"w{j}_vmem_drain"] = vmem[j].drain_deadline
          dbg[f"w{j}_pc_idx"] = pc[j]
          dbg[f"w{j}_valu_hist"] = list(valu[j].issue_hist())
          dbg[f"w{j}_vmem_wr_deadline"] = vmem[j].wr_deadline
      if cat == 'nop': dbg["vmem_drain_deadline"] = vmem[i].drain_deadline
      if cat == 'branch': dbg["op"] = str(kwargs.get('op', ''))
      _sqtt_debug_log.append(dbg)

    issue_cost = _get_issue_cost(pkt_cls, kwargs)
    if pkt_cls is INST and kwargs.get('op') == InstOp.JUMP_NO:
      issue_cost = 10

    # Track memory operation completion times
    if cat == 'smem':
      lds.on_smem_issue(i, issue_cycle + _SMEM_LATENCY)
      if extra:
        for r in extra: sgpr[i].set_smem_ready(r, issue_cycle + _SMEM_LATENCY)
    elif cat in ('ds_rd', 'ds_wr'):
      if cat == 'ds_wr':  # LDS writes serialize through shared unit
        lds.on_ds_write_issue(i, issue_cycle)
      else:
        ds_bytes, ds_dest_base = (extra if extra is not None else (4, None))
        lds_complete, is_serialized = lds.on_ds_read_issue(i, issue_cycle, ds_bytes=ds_bytes)
        # b128 VGPR stagger: upper 2 VGPRs of SERIALIZED b128 loads have extended latency
        # Only the 2nd+ b128 load in a consecutive pair gets stagger (HW validated: layernorm 1st load v[2],v[3] no stagger)
        # VGPR-scoreboard state (vgpr_ready / vgpr_slow_fresh_until) lives on VAluPipe
        # (EMU_REWRITE_DESIGN §1.2, Step 5c). LdsPipe only reports `is_serialized`.
        if ds_bytes >= 16 and ds_dest_base is not None:
          sf = valu[i].vgpr_slow_fresh_map()
          if is_serialized:
            vr = valu[i].vgpr_ready_map()
            for off in (2, 3):
              stagger_ready = lds_complete + _LDS_B128_VGPR_STAGGER
              vr[ds_dest_base + off] = max(vr.get(ds_dest_base + off, 0), stagger_ready)
              sf[ds_dest_base + off] = stagger_ready + 4  # slow-freshness window: consuming within 4cy of ready yields 9cy VALU latency
          # Clear slow-freshness on all b128 VGPRs (fresh data overwrites any prior slow state)
          for off in range(4): sf.pop(ds_dest_base + off, None) if not is_serialized or off < 2 else None
    elif cat == 'vmem_rd':
      vmem[i].on_vmem_issue(issue_cycle + _VMEM_LATENCY)
      vmem[i].set_drain_deadline(issue_cycle + _VMEM_DRAIN_CYCLES)
    elif cat == 'vmem_wr':
      vmem[i].on_vmem_issue(issue_cycle + _VMEM_LATENCY)
      # Slow-fresh forwarding stall overlaps with VMEM execution: reduce drain by the slow-fresh extension
      _drain = max(_VMEM_EXEC_MIN, _VMEM_DRAIN_CYCLES - vmem[i].wr_slow_ext) if _vmem_fwd_stall > 0 else _VMEM_DRAIN_CYCLES
      vmem[i].set_drain_deadline(issue_cycle + _drain)
      # Post-store VALU stall: HW mb_vmem_store_b32_chain_n4 shows next VALU
      # stamps +8cy after store dispatch (emu had +1cy).
      vmem[i].set_post_wr_valu_ready(issue_cycle + 8)
      last_vmem_wr_issue[i] = issue_cycle

    # Track multi-cycle VALU (transcendental) completion for s_waitcnt_depctr, and trans pipeline occupancy
    if _is_trans:
      tn = kwargs.get('trans_name', '')
      if 'SQRT' in tn or 'RSQ' in tn or 'EXP' in tn:
        trans_lat = _TRANS_PIPELINE_LATENCY_SQRT  # complex trans: v_exp, v_sqrt, v_rsq = 31 cycles
      else:
        trans_lat = _TRANS_PIPELINE_LATENCY  # simple trans: v_log, v_rcp = 27 cycles
      trans[i].on_trans_issue(issue_cycle + trans_lat)
      trans[i].set_pipe_avail(issue_cycle + _TRANS_PIPE_CYCLES)
      # Trans→scalar visibility: scalar path (waitcnt, s_nop) stalls until trans pipeline clears
      trans[i].set_scalar_ready(issue_cycle + _TRANS_PIPE_CYCLES - 1)
    # Update int-mul pipe availability (4cy throughput).
    if _is_int_mul32:
      valu[i].set_int_mul_pipe_avail(issue_cycle + 4)

    # Track VOPD pipeline occupancy: next VOPD must wait for the dual-issue slot.
    # Batch D finding: non-self-fwd VOPD → VOPD chains at 1cy (confirmed
    # mb_d3_vopd_chain{2,4}_then_vcmp). Self-fwd VOPDs (read+write same VGPR)
    # force next to wait 4cy because the register-file bypass pipeline needs to settle
    # (confirmed exp_chain [16]→[17] both self-fwd at HW=4cy).
    if is_vopd:
      _vopd_selffwd = (isinstance(vgpr_r_regs, (tuple, list)) and isinstance(vgpr_w_regs, (tuple, list))
                       and bool(set(vgpr_r_regs) & set(vgpr_w_regs)))
      # Same-write-set bypass: when a self-fwd VOPD writes the SAME VGPRs as the previous
      # VOPD, the pipe slot is reused and the self-fwd 4cy cost isn't re-paid. HW
      # mb_vopd_bank_conflict_{src,dst} confirms 1cy for consecutive self-fwd VOPDs
      # writing v[4],v[5],v[4],v[5]... whereas exp_chain [16-17] (self-fwd writing
      # DIFFERENT VGPRs v[0,1] then v[2,3]) measures 4cy.
      _curr_writes = frozenset(vgpr_w_regs) if isinstance(vgpr_w_regs, (tuple, list)) else frozenset()
      _same_write_set = _vopd_selffwd and _curr_writes and _curr_writes == valu[i].last_vopd_writes
      if is_vopd_lit: _pipe_gap = 1
      elif _vopd_paid_phase_warmup: _pipe_gap = 2
      elif _vopd_selffwd and not _same_write_set: _pipe_gap = _VOPD_PIPE_CYCLES
      else: _pipe_gap = 1  # non-self-fwd OR same-write-set VOPD chains at 1cy (HW Batch D + bank_conflict)
      valu[i].set_vopd_pipe_avail(issue_cycle + _pipe_gap)
      valu[i].set_last_vopd_issue(issue_cycle)
      valu[i].set_last_vopd_writes(_curr_writes)
      # Track per-bank write time for inter-VOPD bank port pressure (Seb-V write-commit 1cy delay).
      if isinstance(vgpr_w_regs, (tuple, list)) and vgpr_w_regs:
        for r in vgpr_w_regs: valu[i].set_bank_vopd_write_time(r & 3, issue_cycle)
      # VOPD_LIT between depctr and cmp_lit-chain drains the scalar-pipe phase offset:
      # HW exp_chain chain-4 ([48] depctr → [50-51] VOPD_LIT → [52+] cmp_lit) has no +3
      # offset on A[0], while chain-3 ([27] depctr → [28] plain VOPD → [29+] cmp_lit) does.
      # `phase_shift_armed` stays set so GAP=1 still applies once cmp_lit reactivates it.
      if is_vopd_lit: sgpr[i].set_next_cmp_lit_phase_offset(0)

    # Track consecutive single-issue (non-VOPD non-trans) VALU run-length for dual-issue ramp (Seb-V).
    # VOPD after a chain of singles needs extra cycles to ramp the dual-issue pipeline.
    if cat == 'valu' and not _is_trans and not is_vopd:
      valu[i].inc_consecutive_single_valu()
    elif cat != 'valu' or is_vopd:
      valu[i].set_consecutive_single_valu(0)

    # Track last VALU for DS/VMEM forwarding stall (skip VOPC/VOP3_SDST — writes VCC/SGPR not VGPR)
    if cat == 'valu' and vgpr_w_regs:
      # Slow-fresh VALU extends forwarding deadlines: result arrives later, shifting all forwarding paths
      # The extension is 2×(lat-5) because the SQ pipeline detects both the issue stall AND the late write-back
      _sf_map = valu[i].vgpr_slow_fresh_map()
      _slow_consume = not _is_trans and vgpr_r_regs and any(issue_cycle <= _sf_map.get(r, 0) for r in vgpr_r_regs)
      _slow_extra = 8 if _slow_consume else 0
      valu_ds_wr_deadline[i] = issue_cycle + _VALU_DS_WR_FORWARD + _slow_extra
      valu_ds_rd_deadline[i] = issue_cycle + _VALU_DS_RD_FORWARD + _slow_extra
      vmem[i].set_wr_deadline(issue_cycle + _VALU_VMEM_WR_FORWARD + _slow_extra)
      vmem[i].set_wr_set_time(issue_cycle)
      vmem[i].set_wr_slow_ext(_slow_extra)
      vmem[i].set_rd_deadline(issue_cycle + _VALU_VMEM_RD_FORWARD + _slow_extra)
      # Track consecutive VGPR write patterns for VMEM store forwarding optimization
      n_written = len(set(vgpr_w_regs))
      is_selffwd = bool(vgpr_r_regs and set(vgpr_w_regs) & set(vgpr_r_regs))
      if is_selffwd:
        valu[i].add_consecutive_selffwd_vgprs(n_written)
      else:
        valu[i].set_consecutive_selffwd_vgprs(0)
      valu[i].add_consecutive_vgprs_written(n_written)
      # Cndmask cluster: accumulate on cndmask VGPR writes; any non-cndmask non-VOPC VALU
      # that writes a VGPR breaks the cluster. VOPCs are kept (handled in elif below).
      if is_cndmask:
        valu[i].add_cndmask_cluster_vgprs(n_written)
        valu[i].set_last_cndmask_issue(issue_cycle)
      else: valu[i].set_cndmask_cluster_vgprs(0)
    elif cat == 'valu':  # VOPC/VOP3_SDST — no VGPR writes, breaks forwarding chain
      valu[i].set_consecutive_selffwd_vgprs(0)
      valu[i].set_consecutive_vgprs_written(0)
      # VOPCs DO NOT reset cndmask_cluster_vgprs — they're expected between cndmasks.

    # Track VALU issue times for time-based delay_alu (non-trans only)
    if cat == 'valu' and not _is_trans:
      vh = valu[i].issue_hist()
      # Before appending this issue, track RAW chain depth (same-reg RAW on the
      # IMMEDIATELY previous VALU). A continuation bumps chain depth; anything
      # else resets. Used by the wave-credit stall above.
      _prev_valu_issue = vh[-1] if vh else -1
      if vgpr_r_regs and _prev_valu_issue > 0:
        _vwt_chain = valu[i].vgpr_write_time_map()
        if any(_vwt_chain.get(r, -1) == _prev_valu_issue for r in vgpr_r_regs):
          valu[i].inc_raw_chain_depth()
        else:
          valu[i].set_raw_chain_depth(0)
      else:
        valu[i].set_raw_chain_depth(0)
      vh.append(issue_cycle)
      if len(vh) > 4: vh.pop(0)

    # Track SGPR write times for SGPR dependency stall detection (non-trans VALU only)
    if cat == 'valu' and not _is_trans and sgpr_w_regs:
      for r in sgpr_w_regs:
        sgpr[i].set_write_time(r, issue_cycle)
        # Non-LIT-v_cmp SGPR write invalidates completion-buffer entry (standard SGPR latency applies)
        if not is_cmp_lit: sgpr[i].pop_cmp_lit_read_ready(r)

    # LIT v_cmp completion buffer: update write/commit tracking, compute A[n] for readers.
    if is_cmp_lit and sgpr_w_regs:
      # Apply the scalar-pipe phase offset (set after s_waitcnt_depctr). Only applies to
      # explicit SGPR writes (not implicit VCC=106) — HW exp_chain [33] shows cndmask reading
      # VCC is unaffected by the phase shift, only reads of fresh s[0..n] pay the offset.
      # Applied to the first cmp's W (which cascades through prev_C to later cmps); uniform
      # across the chain via prev_C chaining, not re-added per cmp.
      _nonvcc_writes = [r for r in sgpr_w_regs if r != 106]
      _phase_offset = sgpr[i].next_cmp_lit_phase_offset if _nonvcc_writes else 0
      if _phase_offset > 0:
        sgpr[i].set_next_cmp_lit_phase_offset(0)
        sgpr[i].set_in_phase_shifted_chain(True)  # mark chain as post-depctr for VOPD-warmup rule
      # phase_shift_armed preserves GAP=1 across a waitcnt drain that absorbed the offset
      # (exp_chain chain-2 [53-57]). Fire regardless of VCC/non-VCC, consume on first cmp_lit.
      if sgpr[i].phase_shift_armed:
        sgpr[i].set_phase_shift_armed(False)
        sgpr[i].set_in_phase_shifted_chain(True)
      W = issue_cycle + _CMP_LIT_WB_LATENCY
      prev_C = sgpr[i].cmp_lit_last_commit
      # Phase-shifted chains use COMMIT_GAP=1 (HW exp_chain [35],[36],[58] show subsequent
      # cndmask reads pipeline at 1cy dt not 2cy — the scalar-pipe phase allows tighter
      # commit-buffer packing). Non-shifted chains keep the standard GAP=2.
      _gap = 1 if sgpr[i].in_phase_shifted_chain else _SGPR_COMMIT_GAP
      C = max(W, prev_C + _gap) if prev_C else W
      C += _phase_offset  # phase shift lifts C uniformly for this chain entry
      A = C + 1
      # Interleaved cmp→cndmask bypass: when the NEXT inst is a cndmask that reads this
      # cmp's SGPR, HW forwards via direct bypass at I+1 instead of the full completion
      # buffer latency. HW mb_vcmp_interleave_cndmask (cmp,cndmask,cmp,cndmask...) shows
      # each pair at dt=1. Only applies when phase_shift_armed/offset are inactive
      # (exp_chain phase-shifted chains still need the buffer + GAP=1 model).
      _next_pc = pc[i] + 1
      if (_next_pc < len(wave_events[wid]) and not sgpr[i].in_phase_shifted_chain
          and _phase_offset == 0):
        _nxt_cls, _nxt_kw, _nxt_cat, _nxt_extra = wave_events[wid][_next_pc]
        if _nxt_cat == 'valu' and isinstance(_nxt_extra, tuple) and len(_nxt_extra) >= 10 and _nxt_extra[9]:
          # Next is a cndmask. Check if it reads any of our writes.
          _nxt_sgpr_r = _nxt_extra[2]
          if _nxt_sgpr_r and any(r in sgpr_w_regs for r in _nxt_sgpr_r):
            # Single-hop bypass for the writes that will be consumed immediately.
            A = min(A, issue_cycle + 1)
      sgpr[i].set_cmp_lit_last_commit(C)
      for r in sgpr_w_regs: sgpr[i].set_cmp_lit_read_ready(r, A)
      cmp_hist = sgpr[i].cmp_lit_hist()
      cmp_hist.append(issue_cycle)
      if len(cmp_hist) > 3: cmp_hist.pop(0)
    elif cat == 'valu':
      # Reset the phase-shifted-chain flag on a VALU that is neither cmp_lit nor any cndmask.
      # Chain builders (cmp_lit writes) and consumers (cndmask of any kind — VCC or non-VCC)
      # keep the chain alive. A VOPD or other VALU breaks it (after consuming the +2cy rule
      # at the VOPD issue site above).
      # Cndmask detector: reads SGPR 106 (VCC) or has cond_sgpr set (non-VCC condition).
      _is_any_cndmask = (cond_sgpr >= 0) or (sgpr_r_regs and 106 in sgpr_r_regs)
      if not _is_any_cndmask and not is_cmp_lit:
        sgpr[i].set_in_phase_shifted_chain(False)
      # Non-LIT-v_cmp VALU resets the LIT chain (commit buffer drains when out of chain)
      if not is_cmp_lit:
        if sgpr[i].cmp_lit_hist(): sgpr[i].clear_cmp_lit_hist()
        sgpr[i].set_cmp_lit_last_commit(0)

    # cndmask streak bookkeeping: bump on SGPR-reading cndmask VALU, reset on any other VALU.
    # Runs independent of the cmp_lit/elif branch above so cmp_lit correctly resets the streak.
    if cat == 'valu':
      _is_any_cndmask_streak = (cond_sgpr >= 0) or (sgpr_r_regs and 106 in sgpr_r_regs)
      if _is_any_cndmask_streak: sgpr[i].bump_cndmask_streak()
      else: sgpr[i].reset_cndmask_streak()

    # Track EXEC write time for v_cmpx → s_cbranch_execz/nz dependency
    if cat == 'valu' and kwargs.get('op') == InstOp.VALU1_WR_EXEC:
      scal[i].set_exec_write_time(issue_cycle)

    # Track SCC write time: SALU (s_cmp, s_cmpk, etc.) writes SCC for branch dependency
    if cat == 'salu':
      scal[i].set_scc_write_time(issue_cycle)
      scal[i].set_exec_write_time(issue_cycle)

    # Track VGPR write readiness: all VALU writes update scoreboard for RAW dependency tracking
    if cat == 'valu' and not _is_trans and vgpr_w_regs:
      vr = valu[i].vgpr_ready_map()
      sf = valu[i].vgpr_slow_fresh_map()
      # Check if any source VGPR is "slow-fresh" — consumed while register file data is still settling from b128 stagger
      slow_consume = vgpr_r_regs and any(issue_cycle <= sf.get(r, 0) for r in vgpr_r_regs)
      # VALU with no VGPR reads (e.g. v_mov with constant) has 1-cycle latency (no register file read needed)
      if not vgpr_r_regs: lat = 1
      elif slow_consume: lat = 9  # slow-fresh source: result write-back delayed (HW validated: layernorm [27],[39],[40])
      else: lat = 5
      for r in vgpr_w_regs:
        vr[r] = issue_cycle + lat
        trans[i].pop_vgpr_ready(r)  # non-trans write overrides trans readiness
        # Propagate slow-freshness: if this VALU is slow, its result is also slow-fresh for a window
        if lat == 9: sf[r] = vr[r] + 4
        else: sf.pop(r, None)  # clean VALU clears slow-freshness

    # Track VGPR readiness for trans ops: non-trans consumers must wait for full writeback
    # Trans→trans uses internal forwarding (handled by trans_pipe_avail), so we track separately
    if cat == 'valu' and _is_trans and vgpr_w_regs:
      for r in vgpr_w_regs:
        trans[i].set_vgpr_ready(r, issue_cycle + trans_lat)

    # Track per-VGPR write times for VMEM address forwarding (all non-trans VALU writes)
    if cat == 'valu' and not _is_trans and vgpr_w_regs:
      vwt = valu[i].vgpr_write_time_map()
      for r in vgpr_w_regs: vwt[r] = issue_cycle

    # Track VALU burst
    if cat == 'valu':
      if burst_wave != i:
        # New burst starting — only exclusive if other waves are truly stalled (not just 1cy behind from interleaving)
        burst_exclusive = all(wave_done[j] or at_barrier[j] or ready[j] > issue_cycle + 2
                             for j in range(n) if j != i)
      burst_wave = i
    else:
      burst_wave = -1
      burst_exclusive = False

    if cat == 'barrier':
      at_barrier[i] = True
      barrier_issue[i] = issue_cycle
      ready[i] = issue_cycle + issue_cost
    elif cat == 'waveend':
      wave_done[i] = True
    else:
      # Trans VGPR read stall is absorbed by s_waitcnt_depctr, not VALU dispatch — don't hold the wave.
      ready[i] = issue_cycle + issue_cost

    clock = issue_cycle + 1
    rr = (i + 1) % n
    pc[i] += 1

  if _SIMD_ARB_SHADOW:
    _simd_arb_shadow_log.append({
      "n_waves": n,
      "would_stall_cy": _arb_would_stall_cy,
      "would_stall_count": _arb_would_stall_count,
      "port_avail": list(arbiter.snapshot()["port_avail"]),
      # events_end is the exclusive upper bound into _simd_arb_shadow_events
      # for this kernel call — analysis slices [prev.events_end : events_end].
      "events_end": len(_simd_arb_shadow_events),
    })

  return timed

# ═══════════════════════════════════════════════════════════════════════════════

def _init_sqtt_encoder(entry_pc: int):
  from tinygrad.runtime.autogen.amd.rdna3.enum import SOPPOp as SOPPOp3, SOPKOp as SOPKOp3
  from tinygrad.runtime.autogen.amd.rdna4.enum import SOPPOp as SOPPOp4
  import re

  _SOPP = (ir3.SOPP, ir4.SOPP, irc.SOPP)
  _SMEM = (ir3.SMEM, ir4.SMEM, irc.SMEM)
  _VALU = (ir3.VOP1, ir3.VOP2, ir3.VOP3, ir3.VOP3P, ir3.VOPC, ir3.VOPD, ir3.VOP3SD, ir3.VOP3_SDST, ir3.VOP1_SDST,
           ir4.VOP1, ir4.VOP2, ir4.VOP3, ir4.VOP3P, ir4.VOPC, ir4.VOPD, ir4.VOP3SD, ir4.VOP3_SDST, ir4.VOP1_SDST,
           irc.VOP1, irc.VOP2, irc.VOP3, irc.VOP3P, irc.VOPC, irc.VOP3SD, irc.VOP3_SDST)
  _VOPC = (ir3.VOPC, ir4.VOPC, irc.VOPC)  # comparison ops write VCC, not VGPR — no DS forwarding stall
  _VOP3_SDST = (ir3.VOP3_SDST, ir3.VOP3SD, ir4.VOP3_SDST, ir4.VOP3SD, irc.VOP3_SDST, irc.VOP3SD)  # compare → named SGPR
  _VOPD = (ir3.VOPD, ir4.VOPD)  # dual-issue: two ops, two VGPR dests
  _VOPD_LIT = (ir3.VOPD_LIT, ir4.VOPD_LIT)  # VOPD with shared literal operand
  _CMP_LIT = (ir3.VOPC_LIT, ir3.VOP3_SDST_LIT, ir4.VOPC_LIT, ir4.VOP3_SDST_LIT)  # LIT-source v_cmp: uses SGPR write-back completion buffer
  _DS = (ir3.DS, ir4.DS, irc.DS)
  _GLOBAL = (ir3.GLOBAL, ir4.VGLOBAL, irc.GLOBAL)
  _FLAT = (ir3.FLAT, ir4.VFLAT, irc.FLAT)
  _SCRATCH = (ir3.SCRATCH, ir4.VSCRATCH, irc.SCRATCH)

  # SOPP classification sets
  _SOPP_SKIP = {SOPPOp3.S_ENDPGM.value, SOPPOp3.S_ENDPGM_SAVED.value, SOPPOp3.S_ENDPGM_ORDERED_PS_DONE.value,
                SOPPOp3.S_SENDMSG.value, SOPPOp3.S_SENDMSGHALT.value}
  _SOPP_IMMEDIATE = {SOPPOp3.S_NOP.value, SOPPOp3.S_CLAUSE.value, SOPPOp3.S_WAITCNT.value, SOPPOp3.S_WAITCNT_DEPCTR.value,
                     SOPPOp3.S_WAIT_IDLE.value, SOPPOp3.S_WAIT_EVENT.value, SOPPOp3.S_SLEEP.value,
                     SOPPOp3.S_SET_INST_PREFETCH_DISTANCE.value}
  for _op in (SOPPOp4.S_WAIT_ALU, SOPPOp4.S_WAIT_LOADCNT, SOPPOp4.S_WAIT_STORECNT, SOPPOp4.S_WAIT_SAMPLECNT,
              SOPPOp4.S_WAIT_BVHCNT, SOPPOp4.S_WAIT_EXPCNT, SOPPOp4.S_WAIT_DSCNT, SOPPOp4.S_WAIT_KMCNT,
              SOPPOp4.S_WAIT_LOADCNT_DSCNT, SOPPOp4.S_WAIT_STORECNT_DSCNT):
    _SOPP_IMMEDIATE.add(_op.value)
  _SOPP_BARRIER = {SOPPOp3.S_BARRIER.value}
  if hasattr(SOPPOp4, 'S_BARRIER_WAIT'): _SOPP_BARRIER.add(SOPPOp4.S_BARRIER_WAIT.value)
  if hasattr(SOPPOp4, 'S_BARRIER_LEAVE'): _SOPP_BARRIER.add(SOPPOp4.S_BARRIER_LEAVE.value)
  _SOPP_BRANCH = {SOPPOp3.S_BRANCH.value, SOPPOp3.S_CBRANCH_SCC0.value, SOPPOp3.S_CBRANCH_SCC1.value,
                  SOPPOp3.S_CBRANCH_VCCZ.value, SOPPOp3.S_CBRANCH_VCCNZ.value,
                  SOPPOp3.S_CBRANCH_EXECZ.value, SOPPOp3.S_CBRANCH_EXECNZ.value}
  _SOPP_BRANCH_EXEC = {SOPPOp3.S_CBRANCH_EXECZ.value, SOPPOp3.S_CBRANCH_EXECNZ.value}
  _SOPP_BRANCH_SCC = {SOPPOp3.S_CBRANCH_SCC0.value, SOPPOp3.S_CBRANCH_SCC1.value}

  # RDNA3-only SOPK waitcnt instructions (RDNA4 uses SOPP s_wait_* instead)
  _SOPK = (ir3.SOPK, ir4.SOPK, irc.SOPK)
  _SOPK_WAITCNT_LGKM = SOPKOp3.S_WAITCNT_LGKMCNT.value   # simm16[5:0] = lgkm threshold
  _SOPK_WAITCNT_VM = SOPKOp3.S_WAITCNT_VMCNT.value        # simm16[5:0] = vm threshold

  # VALU sub-classification patterns
  _VALUT_4_RE = re.compile(r'V_(EXP|LOG|RCP|RSQ|SQRT|SIN|COS|CEIL|FLOOR|TRUNC|RNDNE|FRACT|FREXP)_')
  _VALUB_2_RE = re.compile(r'V_(LSHLREV|LSHRREV|ASHRREV)_(B|I)64')
  _VALUB_4_RE = re.compile(r'V_MAD_(U|I)64')
  _VALUB_16_RE = re.compile(r'V_\w+_F64')

  def _valu_op(op_name: str) -> InstOp|None:
    if 'CMPX' in op_name: return InstOp.VALU1_WR_EXEC
    if _VALUB_2_RE.search(op_name): return InstOp.VALUB_2
    if _VALUB_4_RE.search(op_name): return InstOp.VALUB_4
    if _VALUB_16_RE.search(op_name): return InstOp.VALUB_16
    if _VALUT_4_RE.search(op_name): return InstOp.VALUT_4
    return None

  def _mem_op(t, op_name: str, inst=None) -> InstOp:
    is_store = "STORE" in op_name or "WRITE" in op_name
    m = re.search(r'_B(\d+)', op_name)
    if m: dwords = max(1, int(m.group(1)) // 32)
    elif 'DWORDX' in op_name: dwords = int(re.search(r'DWORDX(\d)', op_name).group(1))
    elif 'BLOCK' in op_name: dwords = 4
    else: dwords = 1
    if issubclass(t, _DS):
      if not is_store: return InstOp.LDS_RD
      if any(x in op_name for x in ("APPEND", "CONSUME", "ADDTID")): return InstOp.LDS_WR_1
      return InstOp[f"LDS_WR_{1 + dwords + (1 if '2ADDR' in op_name else 0)}"]
    if issubclass(t, _GLOBAL):
      saddr_null = inst is None or not hasattr(inst, 'saddr') or inst.saddr.offset in (124, 125)
      if not is_store: return InstOp.SGMEM_RD_2 if saddr_null else InstOp.SGMEM_RD_1
      return InstOp[f"SGMEM_WR_{1 + dwords + (1 if saddr_null else 0)}"]
    if issubclass(t, _FLAT): return InstOp[f"FLAT_WR_{2 + dwords}"] if is_store else InstOp.FLAT_RD_2
    if issubclass(t, _SCRATCH): return InstOp[f"FLAT_WR_{2 + dwords}"] if is_store else InstOp.FLAT_RD_2
    return InstOp.SALU

  # Deferred event storage: wave_id -> [(pkt_cls_or_None, kwargs, category, extra)]
  # Categories: 'wavestart', 'salu', 'valu', 'smem', 'ds_rd', 'ds_wr', 'vmem_rd', 'vmem_wr',
  #             'waitcnt', 'immediate', 'barrier', 'branch', 'delay_alu', 'waveend'
  wave_events: dict[int, list] = {}
  started: set[int] = set()

  def emit(wave_id: int, inst, branch_taken: bool|None):
    """Collect an instruction event for deferred timing simulation."""
    w = wave_id & 0x1F
    events = wave_events.setdefault(wave_id, [])
    if wave_id not in started:
      # id7 bit 5 = me=1 (MEC compute engine), bits 4:3 = pipe=0 — must match REG(slot=4) which encodes me=1,pipe=0
      events.append((WAVESTART, {'simd': 0, 'cu_lo': 0, 'wave': w, 'id7': 0x20}, 'wavestart', None))
      started.add(wave_id)
    inst_type, inst_op, op_name = type(inst), inst.op.value if hasattr(inst, 'op') else 0, inst.op.name if hasattr(inst, 'op') else ""

    if issubclass(inst_type, _SOPP):
      if inst_op in _SOPP_SKIP: return
      if inst_op == SOPPOp3.S_DELAY_ALU.value:
        events.append((None, {}, 'delay_alu', inst.simm16))
        return
      if inst_op in _SOPP_IMMEDIATE:
        if inst_op == SOPPOp3.S_WAITCNT.value:
          events.append((IMMEDIATE, {'wave': w}, 'waitcnt', inst.simm16))
        elif inst_op == SOPPOp3.S_WAITCNT_DEPCTR.value:
          events.append((IMMEDIATE, {'wave': w}, 'depctr', inst.simm16))
        elif inst_op == SOPPOp3.S_NOP.value:
          events.append((IMMEDIATE, {'wave': w}, 'nop', inst.simm16))
        elif inst_op == SOPPOp3.S_CLAUSE.value:
          events.append((IMMEDIATE, {'wave': w}, 'clause', None))
        else:
          events.append((IMMEDIATE, {'wave': w}, 'immediate', None))
        return
      if inst_op in _SOPP_BARRIER:
        events.append((INST, {'wave': w, 'op': InstOp.BARRIER}, 'barrier', None))
        return
      if inst_op in _SOPP_BRANCH:
        reads_exec = inst_op in _SOPP_BRANCH_EXEC
        reads_scc = inst_op in _SOPP_BRANCH_SCC
        events.append((INST, {'wave': w, 'op': InstOp.JUMP if branch_taken else InstOp.JUMP_NO}, 'branch', (reads_exec, reads_scc)))
        return
      events.append((INST, {'wave': w, 'op': InstOp.SALU}, 'salu', None))
      return

    if issubclass(inst_type, _SOPK):
      op_val = inst.op.value if hasattr(inst, 'op') else 0
      if op_val == _SOPK_WAITCNT_LGKM:
        # s_waitcnt_lgkmcnt: simm16[5:0]=lgkm threshold; encode for _drain_zero_cost: bits[9:4]=lgkm, bits[15:10]=vm(63=don't wait)
        lgkm_th = inst.simm16 & 0x3f
        events.append((IMMEDIATE, {'wave': w}, 'waitcnt', (lgkm_th << 4) | (0x3f << 10)))
        return
      if op_val == _SOPK_WAITCNT_VM:
        vm_th = inst.simm16 & 0x3f
        events.append((IMMEDIATE, {'wave': w}, 'waitcnt', (0x3f << 4) | (vm_th << 10)))
        return
      events.append((INST, {'wave': w, 'op': InstOp.SALU}, 'salu', None))
      return

    if issubclass(inst_type, _VALU):
      is_vopc = issubclass(inst_type, _VOPC)
      op = _valu_op(op_name)
      # Extract SGPR reads/writes for dependency stall detection
      sgpr_w: list[int] = []
      sgpr_r: list[int] = []
      vgpr_w: list[int] = []
      vgpr_r: list[int] = []
      # SGPR destinations: vdst is SSrcField(0-106) for VOP3_SDST, VGPRField(256+) for VOP3; sdst is SGPRField(0-106)
      if hasattr(inst, 'vdst'):
        o = getattr(inst.vdst, 'offset', -1)
        if 0 <= o <= 106: sgpr_w.append(o)
        elif o >= 256: vgpr_w.append(o - 256)
      if hasattr(inst, 'sdst'):
        o = getattr(inst.sdst, 'offset', -1)
        if 0 <= o <= 106: sgpr_w.append(o)
      if is_vopc: sgpr_w.append(106)   # VOPC implicitly writes VCC_LO (offset 106)
      # VOPD dual destinations: vdstx and vdsty
      if hasattr(inst, 'vdstx'):
        o = getattr(inst.vdstx, 'offset', -1)
        if o >= 256: vgpr_w.append(o - 256)
      if hasattr(inst, 'vdsty'):
        o = getattr(inst.vdsty, 'offset', -1)
        if o >= 256: vgpr_w.append(o - 256)
      # Sources: SGPR reads (SrcFields encode SGPRs 0-105, VCC 106-107; HW reads all encoded sources)
      # src2 handling: VOP3_SDST (v_cmp_e64) doesn't read src2; VOP3 v_cndmask reads src2 as condition SGPR
      is_sdst = issubclass(inst_type, _VOP3_SDST)
      cond_sgpr = -1  # condition SGPR for v_cndmask (non-VCC src2); -1 = none
      for fn in ('src0', 'src1', 'srcx0', 'srcy0'):
        if hasattr(inst, fn):
          o = getattr(getattr(inst, fn), 'offset', -1)
          if 0 <= o <= 106: sgpr_r.append(o)
          elif o >= 256: vgpr_r.append(o - 256)
      if hasattr(inst, 'src2'):
        o = getattr(inst.src2, 'offset', -1)
        if not is_sdst:  # VOP3_SDST (v_cmp_e64) has src2 in encoding but HW doesn't read it (validated: exp_chain [54])
          if 0 <= o <= 106:
            sgpr_r.append(o)
            if o != 106: cond_sgpr = o  # non-VCC condition SGPR (v_cndmask_b32_e64)
          elif o >= 256: vgpr_r.append(o - 256)
      # VGPR-only sources: vsrc1, vsrcx1, vsrcy1 are always VGPRs
      for fn in ('vsrc1', 'vsrcx1', 'vsrcy1'):
        if hasattr(inst, fn):
          o = getattr(getattr(inst, fn), 'offset', -1)
          if o >= 256: vgpr_r.append(o - 256)
      # Reliable cndmask detector: any op whose name contains "CNDMASK" (V_CNDMASK_B32_E32/E64,
      # V_DUAL_CNDMASK_B32, V_CNDMASK_B16_E64). Previously we inferred cndmask from
      # cond_sgpr/VCC reads, but VOP3 2-source ops like v_add_f32_e64 can have src2
      # populated by the assembler and misfire that heuristic.
      is_cndmask = 'CNDMASK' in op_name
      # Integer 32-bit multiply is a 4-cycle pipeline op on RDNA3 (not 1cy like f32
      # mul). HW mb_e4_vmul_lo_u32_n4 / mb_f6_vmul_hi_chain_n4 show dt=4 unanimous
      # between consecutive int-muls on the first 6 waves.
      # 4cy pipeline: v_mul_lo_u32 / v_mul_hi_u32 / i32 variants. HW measured dt=4
      # on chain continuations. NOT v_mad_u32_u24 (measured dt=1) and NOT *_u24
      # variants — those pipeline at 1cy like regular ALU.
      is_int_mul32 = any(p in op_name for p in (
          'V_MUL_LO_U32', 'V_MUL_HI_U32', 'V_MUL_LO_I32', 'V_MUL_HI_I32'))
      # Guard against false match on U32_U24 variants (those are 1cy).
      if is_int_mul32 and '_U24' in op_name: is_int_mul32 = False
      # FMAC / MAC / FMAAK / FMAMK: vdst is an implicit source (accumulator).
      # Add it to vgpr_r so RAW dep tracking sees the accumulator chain. HW
      # mb_f1_valu_fmac_n16 confirms dt=5 stalls kick in on waves 1+ exactly
      # like v_add_n16, which only works if fmac's chain is detected as RAW.
      if ('FMAC' in op_name or 'FMAMK' in op_name or 'FMAAK' in op_name or
          ('_MAC' in op_name and 'FMAC' not in op_name)):
        if hasattr(inst, 'vdst'):
          o = getattr(inst.vdst, 'offset', -1)
          if o >= 256 and (o - 256) not in vgpr_r: vgpr_r.append(o - 256)
      reg_info = (is_vopc, tuple(sgpr_w), tuple(sgpr_r), tuple(vgpr_w), tuple(vgpr_r), issubclass(inst_type, _VOPD), cond_sgpr,
                  issubclass(inst_type, _VOPD_LIT), issubclass(inst_type, _CMP_LIT), is_cndmask)
      # For VOPD: detect "MOV-only" pairs (V_DUAL_MOV_B32 on both lanes). These don't use the vsrc1
      # lanes even though the decoder reports v[0] there — the encoding slot is filled by the assembler
      # with a dummy value. HW (Batch C mb_vopd_dualmov_sgpr_*) shows these pipeline at 1cy.
      _kw_extras = {}
      if issubclass(inst_type, _VOPD):
        opx_name = inst.opx.name if hasattr(inst, 'opx') and hasattr(inst.opx, 'name') else ""
        opy_name = inst.opy.name if hasattr(inst, 'opy') and hasattr(inst.opy, 'name') else ""
        _kw_extras['vopd_mov_only'] = opx_name == 'V_DUAL_MOV_B32' and opy_name == 'V_DUAL_MOV_B32'
      if is_int_mul32: _kw_extras['int_mul32'] = True
      if op is None: events.append((VALUINST, {'wave': w, **_kw_extras}, 'valu', reg_info))
      else:
        kw = {'wave': w, 'op': op, **_kw_extras}
        if op == InstOp.VALUT_4: kw['trans_name'] = op_name
        events.append((INST, kw, 'valu', reg_info))
      return

    if issubclass(inst_type, _SMEM):
      smem_dst = ()
      if hasattr(inst, 'sdata'):
        o = getattr(inst.sdata, 'offset', -1)
        if 0 <= o <= 106:
          m = re.search(r'_B(\d+)', op_name)
          n_sgprs = int(m.group(1)) // 32 if m else 1
          smem_dst = tuple(range(o, o + n_sgprs))
      events.append((INST, {'wave': w, 'op': InstOp.SMEM_RD}, 'smem', smem_dst or None))
      return

    # DS / GLOBAL / FLAT / SCRATCH memory operations
    mem_op_val = _mem_op(inst_type, op_name, inst)
    is_store = "STORE" in op_name
    if issubclass(inst_type, _DS): cat = 'ds_wr' if is_store else 'ds_rd'
    elif issubclass(inst_type, (*_GLOBAL, *_FLAT, *_SCRATCH)): cat = 'vmem_wr' if is_store else 'vmem_rd'
    else: cat = 'salu'
    # For DS loads, extract width and dest VGPR for b128-specific latency/stagger
    ds_extra = None
    if cat == 'ds_rd':
      m = re.search(r'_B(\d+)', op_name)
      ds_bytes = int(m.group(1)) // 8 if m else 4
      # Extract dest VGPR base index for per-VGPR stagger tracking
      ds_dest_base = None
      vdst_field = getattr(inst, 'vdst', None)
      if vdst_field is not None and hasattr(vdst_field, 'offset') and vdst_field.offset >= 256:
        ds_dest_base = vdst_field.offset - 256
      ds_extra = (ds_bytes, ds_dest_base)
    # For VMEM stores, extract data width and address VGPR for per-operand forwarding
    vmem_extra = None
    if cat == 'vmem_wr':
      vmem_bytes = None
      m = re.search(r'_B(\d+)', op_name)
      if m: vmem_bytes = int(m.group(1)) // 8
      # Extract address VGPR index for per-operand forwarding (Reg.offset = 256 + vgpr_idx)
      addr_vgpr = None
      addr_field = getattr(inst, 'addr', None)
      if addr_field is not None and hasattr(addr_field, 'offset') and addr_field.offset >= 256:
        addr_vgpr = addr_field.offset - 256
      vmem_extra = (vmem_bytes, addr_vgpr)
    extra = ds_extra if cat == 'ds_rd' else vmem_extra
    events.append((INST, {'wave': w, 'op': mem_op_val}, cat, extra))

  def finish(wave_id: int):
    """Record wave completion for deferred encoding."""
    if wave_id in started:
      wave_events.setdefault(wave_id, []).append((WAVEEND, {'simd': 0, 'cu_lo': 0, 'wave': wave_id & 0x1F}, 'waveend', None))

  def finalize() -> tuple[bytes, int]:
    """Run timing simulation, encode interleaved SQTT stream with cycle-accurate deltas. Returns (blob, total_cycles)."""
    timed = _simulate_sq_timing(wave_events)
    total_cycles = max((ts for ts, _, _, _ in timed), default=0)
    # Sort: all WAVESTARTs first as a preamble (matching real hardware where all wave allocations
    # appear consecutively before any instructions), then remaining packets ordered by timestamp.
    timed.sort(key=lambda x: (0 if x[2] is WAVESTART else 1, x[0]))
    nibbles: list[int] = []
    _emit_nibbles(nibbles, LAYOUT_HEADER, layout=3, sel_a=6)
    # DISPATCH_INITIATOR: dispatch-scoped, emitted once globally.
    _emit_nibbles(nibbles, REG, delta=0, slot=4, hi_byte=0x82, subop=0x80, val32=0x80000003)
    # PGM_LO/HI: emitted once before all WAVESTARTs (real hardware emits at dispatch time, not per-wave).
    # slot=4 encodes me=1,pipe=0; subop=0xC/0xD are SPI_SHADER_PGM_LO/HI_CS registers.
    pgm = entry_pc >> 8
    _emit_nibbles(nibbles, REG, delta=0, slot=4, hi_byte=0x82, subop=0xC, val32=pgm & 0xFFFFFFFF)
    _emit_nibbles(nibbles, REG, delta=0, slot=4, hi_byte=0x82, subop=0xD, val32=(pgm >> 32) & 0xFFFFFFFF)
    _emit_nibbles(nibbles, SNAPSHOT, delta=0, snap=0)
    _emit_nibbles(nibbles, TS_WAVE_STATE, delta=0, coarse=1)  # wave_interest=True
    prev_time = 0
    for ts, _, pkt_cls, kwargs in timed:
      delta = max(ts - prev_time, 0)
      enc_kwargs = {k: v for k, v in kwargs.items() if k not in ('trans_name', 'vopd_mov_only', 'int_mul32')}
      _emit_with_delta(nibbles, pkt_cls, delta=delta, **enc_kwargs)
      prev_time = max(ts, prev_time)  # actual encoded time; clamped deltas (delta=0) don't advance time
    # Pad to 32-byte alignment
    while len(nibbles) % 2 != 0: nibbles.append(0)
    nibbles.extend([0] * 32)
    while len(nibbles) % 64 != 0: nibbles.append(0)
    return _nibbles_to_bytes(nibbles), total_cycles

  return emit, finish, finalize

def _c(val, dtype=dtypes.uint32): return UOp.const(dtype, val)

def _u64(lo: UOp, hi: UOp) -> UOp:
  """Combine two 32-bit UOps into a 64-bit UOp."""
  return lo.cast(dtypes.uint64) | (hi.cast(dtypes.uint64) << UOp.const(dtypes.uint64, 32))

def _split64(val: UOp) -> tuple[UOp, UOp]:
  """Split a 64-bit value into (lo, hi) 32-bit values."""
  v64 = val.bitcast(dtypes.uint64) if val.dtype == dtypes.float64 else val.cast(dtypes.uint64) if val.dtype != dtypes.uint64 else val
  return v64.cast(dtypes.uint32), (v64 >> UOp.const(dtypes.uint64, 32)).cast(dtypes.uint32)

_SRC_MOD_TYPES = {16: (dtypes.uint16, dtypes.half, 0x7FFF), 32: (dtypes.uint32, dtypes.float32, 0x7FFFFFFF),
                  64: (dtypes.uint64, dtypes.float64, 0x7FFFFFFFFFFFFFFF)}
def _apply_src_mods(val: UOp, mod_bit: int, abs_bits: int, neg_bits: int, bits: int = 32) -> UOp:
  """Apply abs/neg modifiers to source value based on bit width (16, 32, or 64)."""
  if not (abs_bits & (1 << mod_bit)) and not (neg_bits & (1 << mod_bit)): return val
  ut, ft, mask = _SRC_MOD_TYPES[bits]
  fv = val.cast(ut).bitcast(ft) if bits == 16 else val.bitcast(ft) if val.dtype == ut else val
  if abs_bits & (1 << mod_bit): fv = (fv.bitcast(ut) & UOp.const(ut, mask)).bitcast(ft)
  if neg_bits & (1 << mod_bit): fv = fv.neg()
  return fv.bitcast(ut).cast(dtypes.uint32) if bits == 16 else fv.bitcast(ut)

# Map VOPD ops to VOP2 ops for pcode lookup (both RDNA3 and RDNA4)
VOPD_TO_VOP2 = {
  ir3.VOPDOp.V_DUAL_FMAC_F32: ir3.VOP2Op.V_FMAC_F32_E32, ir3.VOPDOp.V_DUAL_MUL_F32: ir3.VOP2Op.V_MUL_F32_E32,
  ir3.VOPDOp.V_DUAL_ADD_F32: ir3.VOP2Op.V_ADD_F32_E32, ir3.VOPDOp.V_DUAL_SUB_F32: ir3.VOP2Op.V_SUB_F32_E32,
  ir3.VOPDOp.V_DUAL_SUBREV_F32: ir3.VOP2Op.V_SUBREV_F32_E32, ir3.VOPDOp.V_DUAL_MAX_F32: ir3.VOP2Op.V_MAX_F32_E32,
  ir3.VOPDOp.V_DUAL_MIN_F32: ir3.VOP2Op.V_MIN_F32_E32, ir3.VOPDOp.V_DUAL_ADD_NC_U32: ir3.VOP2Op.V_ADD_NC_U32_E32,
  ir3.VOPDOp.V_DUAL_LSHLREV_B32: ir3.VOP2Op.V_LSHLREV_B32_E32, ir3.VOPDOp.V_DUAL_AND_B32: ir3.VOP2Op.V_AND_B32_E32,
  ir3.VOPDOp.V_DUAL_MOV_B32: ir3.VOP1Op.V_MOV_B32_E32, ir3.VOPDOp.V_DUAL_CNDMASK_B32: ir3.VOP2Op.V_CNDMASK_B32_E32,
  ir3.VOPDOp.V_DUAL_FMAAK_F32: ir3.VOP2Op.V_FMAAK_F32_E32, ir3.VOPDOp.V_DUAL_FMAMK_F32: ir3.VOP2Op.V_FMAMK_F32_E32,
  ir3.VOPDOp.V_DUAL_DOT2ACC_F32_F16: ir3.VOP2Op.V_DOT2ACC_F32_F16_E32,
  # RDNA4 mappings (same VOP1/VOP2 targets, RDNA4 uses _NUM_ suffix for min/max)
  ir4.VOPDOp.V_DUAL_FMAC_F32: ir3.VOP2Op.V_FMAC_F32_E32, ir4.VOPDOp.V_DUAL_MUL_F32: ir3.VOP2Op.V_MUL_F32_E32,
  ir4.VOPDOp.V_DUAL_ADD_F32: ir3.VOP2Op.V_ADD_F32_E32, ir4.VOPDOp.V_DUAL_SUB_F32: ir3.VOP2Op.V_SUB_F32_E32,
  ir4.VOPDOp.V_DUAL_SUBREV_F32: ir3.VOP2Op.V_SUBREV_F32_E32, ir4.VOPDOp.V_DUAL_MAX_NUM_F32: ir3.VOP2Op.V_MAX_F32_E32,
  ir4.VOPDOp.V_DUAL_MIN_NUM_F32: ir3.VOP2Op.V_MIN_F32_E32, ir4.VOPDOp.V_DUAL_ADD_NC_U32: ir3.VOP2Op.V_ADD_NC_U32_E32,
  ir4.VOPDOp.V_DUAL_LSHLREV_B32: ir3.VOP2Op.V_LSHLREV_B32_E32, ir4.VOPDOp.V_DUAL_AND_B32: ir3.VOP2Op.V_AND_B32_E32,
  ir4.VOPDOp.V_DUAL_MOV_B32: ir3.VOP1Op.V_MOV_B32_E32, ir4.VOPDOp.V_DUAL_CNDMASK_B32: ir3.VOP2Op.V_CNDMASK_B32_E32,
  ir4.VOPDOp.V_DUAL_FMAAK_F32: ir3.VOP2Op.V_FMAAK_F32_E32, ir4.VOPDOp.V_DUAL_FMAMK_F32: ir3.VOP2Op.V_FMAMK_F32_E32,
  ir4.VOPDOp.V_DUAL_DOT2ACC_F32_F16: ir3.VOP2Op.V_DOT2ACC_F32_F16_E32,
}
def _wave_size(arch: str) -> int: return 64 if arch.startswith("cdna") else 32
# Special registers stored after inline constants (256-259)
PC_LO_IDX, PC_HI_IDX, SCRATCH_STRIDE_IDX = 256, 257, 259
# SGPR buffer: 0-127 = SGPRs, 128-255 = inline constants, 256-259 = special registers
SGPR_COUNT = 260
# Sentinel PC value for s_endpgm
ENDPGM_PC = 0xFFFFFFFFFFFFFFFF

def _op_name(inst) -> str:
  if hasattr(inst, 'opx'): return f"{inst.opx.name}_{inst.opy.name}"  # VOPD has opx/opy not op
  return inst.op.name if hasattr(inst.op, 'name') else str(inst.op)

def _to_u32(val: UOp) -> UOp:
  if val.dtype == dtypes.uint32: return val
  if val.dtype.itemsize == 4: return val.bitcast(dtypes.uint32)  # same size: bitcast (float32->uint32)
  return val.cast(dtypes.uint32)  # different size: cast (bool, int16, etc)
def _lane_active(exec_mask: UOp, lane: UOp) -> UOp:
  if exec_mask.dtype == dtypes.uint64: return ((exec_mask >> lane.cast(dtypes.uint64)) & UOp.const(dtypes.uint64, 1)).ne(UOp.const(dtypes.uint64, 0))
  return ((exec_mask >> lane.cast(dtypes.uint32)) & _c(1)).ne(_c(0))
def _hi16(v: UOp) -> UOp: return (v >> _c(16)) & _c(0xFFFF)
def _cond(cond, if_true, if_false):
  """Select between values based on condition (works with UOp or bool)."""
  return cond.where(if_true, if_false) if isinstance(cond, UOp) else if_true if cond else if_false
def _cond_hi16(cond, val: UOp) -> UOp: return _cond(cond, _hi16(val), val)
def _apply_opsel(val: UOp, sel_bit: int, opsel: int) -> UOp: return _hi16(val) if opsel & (1 << sel_bit) else val

def _set_lane_bit(old: UOp, lane: UOp, val: UOp, exec_mask: UOp) -> UOp:
  """Set/clear a single bit in a mask based on lane index, respecting exec mask."""
  if old.dtype in (dtypes.uint64, dtypes.int64):
    dt = dtypes.uint64
    mask = UOp.const(dt, 1) << lane.cast(dt)
    new_bit = _to_u32(val).cast(dt) << lane.cast(dt)
    cleared = old.cast(dt) & (mask ^ UOp.const(dt, 0xFFFFFFFFFFFFFFFF))
    return _lane_active(exec_mask, lane).where(cleared | new_bit, old.cast(dt))
  mask = _c(1) << lane.cast(dtypes.uint32)
  new_bit = _to_u32(val) << lane.cast(dtypes.uint32)
  cleared = old & (mask ^ _c(MASK32))
  return _lane_active(exec_mask, lane).where(cleared | new_bit, old)

def _val_to_u32(val: UOp) -> UOp:
  """Convert any value to uint32 for storage (bitcast floats, cast ints)."""
  if val.dtype == dtypes.uint32: return val
  if val.dtype == dtypes.float32: return val.bitcast(dtypes.uint32)
  if val.dtype == dtypes.half: return val.bitcast(dtypes.uint16).cast(dtypes.uint32)
  if val.dtype in (dtypes.uint16, dtypes.int16): return val.cast(dtypes.uint32)
  return val.cast(dtypes.uint32)

_pcode_fixes = {
  'V_DIV_FMAS_F32': ('D0.f32 = 2.0F ** 32 * fma(S0.f32, S1.f32, S2.f32)',
    'D0.f32 = (exponent(S2.f32) > 127) ? (2.0F ** 64 * fma(S0.f32, S1.f32, S2.f32)) : (2.0F ** -64 * fma(S0.f32, S1.f32, S2.f32))'),
  'V_DIV_FMAS_F64': ('D0.f64 = 2.0 ** 64 * fma(S0.f64, S1.f64, S2.f64)',
    'D0.f64 = (exponent(S2.f64) > 1023) ? (2.0 ** 128 * fma(S0.f64, S1.f64, S2.f64)) : (2.0 ** -128 * fma(S0.f64, S1.f64, S2.f64))'),
  'V_DIV_FIXUP_F32': ('D0.f32 = sign_out ? -abs(S0.f32) : abs(S0.f32)',
    'D0.f32 = isNAN(S0.f32) ? (sign_out ? -INF.f32 : +INF.f32) : (sign_out ? -abs(S0.f32) : abs(S0.f32))'),
  'V_DIV_FIXUP_F64': ('D0.f64 = sign_out ? -abs(S0.f64) : abs(S0.f64)',
    'D0.f64 = isNAN(S0.f64) ? (sign_out ? -INF : +INF) : (sign_out ? -abs(S0.f64) : abs(S0.f64))'),
  'V_TRIG_PREOP_F64': ("result = 64'F((1201'B(2.0 / PI)[1200 : 0] << shift.u32) & 1201'0x1fffffffffffff)", "result = trig_preop_result(shift)"),
}

def _get_pcode_dict(op) -> dict:
  """Return the PCODE dictionary for the given opcode based on its architecture."""
  return PCODE_CDNA if 'cdna' in type(op).__module__ else PCODE_RDNA4 if 'rdna4' in type(op).__module__ else PCODE_RDNA3

# Pcode parser
@functools.cache
def get_pcode(op) -> str:
  op_name = op.name
  pcode_dict = _get_pcode_dict(op)
  if op not in pcode_dict and op_name.endswith('_E64'):
    # VOP3 ops ending in _E64 may share pcode with VOP1 _E32 equivalents
    import importlib
    enum_mod = importlib.import_module(type(op).__module__)
    vop1_cls = getattr(enum_mod, 'VOP1Op', None)
    e32_name = op_name.replace('_E64', '_E32')
    if vop1_cls and hasattr(vop1_cls, e32_name): op = vop1_cls[e32_name]
  pcode = pcode_dict[op]
  fix_name = op_name.replace('_E64', '').replace('_E32', '')
  if fix_name in _pcode_fixes: pcode = pcode.replace(*_pcode_fixes[fix_name])
  if 'V_DIV_SCALE' in op_name:
    dt, exp_lim, ldexp_val = ('f32', '23', '64') if 'F32' in op_name else ('f64', '52', '128')
    for old, new in [(f'S2.{dt} / S1.{dt} == DENORM.{dt}', f'divWouldBeDenorm(S2.{dt}, S1.{dt})'), (f"1.0 / 64'F(S1.{dt}) == DENORM.f64", '0'),
                     (f'1.0 / S1.{dt} == DENORM.{dt}', '0'), (f'S1.{dt} == DENORM.{dt}', f'isDENORM(S1.{dt})'),
                     (f'D0.{dt} = NAN.{dt}', f'VCC = 0x1LL;\nD0.{dt} = NAN.{dt}'),
                     (f'elsif isDENORM(S1.{dt}) then\nD0.{dt} = ldexp(S0.{dt}, {ldexp_val})', f'elsif 1 == 0 then\nD0.{dt} = S0.{dt}'),
                      (f'elsif exponent(S2.{dt}) <= {exp_lim} then\n// Numerator is tiny\n'
                       f'D0.{dt} = ldexp(S0.{dt}, {ldexp_val})',
                       f'elsif exponent(S2.{dt}) <= {exp_lim} then\nVCC = 0x1LL;\n'
                       f'D0.{dt} = ldexp(S0.{dt}, {ldexp_val})'),
                      (f'elsif divWouldBeDenorm(S2.{dt}, S1.{dt}) then\nVCC = 0x1LL;\n'
                       f'if S0.{dt} == S2.{dt} then\n// Only scale the numerator\n'
                       f'D0.{dt} = ldexp(S0.{dt}, {ldexp_val})\nendif',
                       f'elsif divWouldBeDenorm(S2.{dt}, S1.{dt}) then\n'
                       f'VCC = 0x1LL;\nD0.{dt} = S0.{dt}'),
                      (f'D0.{dt} = ldexp(S0.{dt}, {ldexp_val})\nendif\nelsif',
                       f'D0.{dt} = ldexp(S0.{dt}, {ldexp_val})\nelse\n'
                       f'D0.{dt} = S0.{dt}\nendif\nelsif')]:
      pcode = pcode.replace(old, new)
    lines = pcode.rstrip().split('\n')
    for i in range(len(lines) - 1, -1, -1):
      if lines[i].strip() == 'endif':
        lines.insert(i, f'else\nD0.{dt} = S0.{dt}')
        break
    pcode = '\n'.join(lines) + f';\nif isDENORM(S1.{dt}) then\nD0.{dt} = NAN.{dt}\nendif'
    pcode = pcode.replace('VCC = 0x0LL', 'VCC.u64[laneId] = 0').replace('VCC = 0x1LL', 'VCC.u64[laneId] = 1')
  return pcode

def parse_pcode(pcode: str, srcs: dict[str, UOp | int] | None = None) -> tuple[dict, list[tuple[str, UOp]]]:
  env: dict = srcs.copy() if srcs else {}
  assigns: list[tuple[str, UOp]] = []
  raw_lines = [l.strip().rstrip(';') for l in pcode.split('\n') if l.strip() and not l.strip().startswith('//')]
  # TODO: pcode.py should tokenize full pcode string instead of line-by-line, then this hack can be removed
  lines: list[str] = []
  for l in raw_lines:
    if lines and lines[-1].endswith('&&'): lines[-1] = lines[-1] + ' ' + l
    else: lines.append(l)
  _, final, _ = parse_block(lines, 0, env, assigns=assigns)
  sliced = set(d.split('[')[0] for d, _ in assigns if '[' in d)
  for var, val in final.items():
    if var in ['D0', 'S0', 'SCC', 'VCC', 'EXEC', 'PC', 'RETURN_DATA', 'VDATA'] and isinstance(val, UOp):
      if var in sliced and not any(re.match(rf'{var}\.\w+\s*=', l) for l in lines): continue
      for l in lines:
        if (m := re.match(rf'{var}\.(\w+(?:\[\w+\])?)', l)):
          assigns.append((f'{var}.{m.group(1)}', val))
          break
      else: assigns.append((var, val))
  return env, assigns

def _write_64bit(val: UOp, wfn, reg_or_addr, is_mem: bool, *args) -> list[UOp]:
  """Write a 64-bit value as two 32-bit writes. args passed to wfn after reg/addr and lo/hi value."""
  lo, hi = _split64(val)
  incr = 4 if is_mem else 1  # 4 bytes for memory addresses, 1 for register indices
  return [wfn(reg_or_addr, lo, *args), wfn(reg_or_addr + (UOp.const(reg_or_addr.dtype, incr) if isinstance(reg_or_addr, UOp) else incr), hi, *args)]

def _write_val(bits: int, val: UOp, wfn, reg_or_addr, *args, is_mem: bool = False) -> list[UOp]:
  """Write value, splitting 64-bit if needed. bits=64 for 64-bit writes, otherwise 32-bit."""
  return _write_64bit(val, wfn, reg_or_addr, is_mem, *args) if bits == 64 else [wfn(reg_or_addr, _to_u32(val), *args)]

def _mem_store(mem: UOp, addr: UOp, val: UOp, active: UOp, addr_bits: int = 32, data_bits: int = 32) -> list[UOp]:
  """Conditional memory store with sub-word support. Returns list of store UOps."""
  adt = dtypes.uint64 if addr_bits == 64 else dtypes.uint32
  word_addr = addr >> UOp.const(adt, 2)
  idx = mem.index(word_addr.cast(dtypes.int), active)
  if data_bits == 32: return [idx.store(active.where(_to_u32(val), idx))]
  # Sub-word store: read-modify-write with mask
  byte_pos = addr.cast(dtypes.uint32) & _c(3)
  byte_shift = byte_pos * _c(8)
  val_u32, size_mask = val.cast(dtypes.uint32), _c(0xFF if data_bits == 8 else 0xFFFF)
  mask = size_mask << byte_shift
  new_word = (idx & (mask ^ _c(0xFFFFFFFF))) | ((val_u32 & size_mask) << byte_shift)
  if data_bits == 8: return [idx.store(active.where(new_word, idx))]
  # 16-bit cross-word case: byte_pos == 3 means value spans two words
  is_cross = byte_pos.eq(_c(3))
  cross_word0 = (idx & _c(0x00FFFFFF)) | ((val_u32 & _c(0xFF)) << _c(24))
  store0 = idx.store(active.where(is_cross.where(cross_word0, new_word), idx))
  next_idx = mem.index((word_addr + UOp.const(adt, 1)).cast(dtypes.int), active & is_cross)
  cross_word1 = (next_idx & _c(0xFFFFFF00)) | ((val_u32 >> _c(8)) & _c(0xFF))
  return [store0, next_idx.store((active & is_cross).where(cross_word1, next_idx))]

def _mem_store_bytes(mem: UOp, addr: UOp, val: UOp, active: UOp, data_bits: int = 32) -> list[UOp]:
  """Store to byte-addressable memory (scratch). addr is byte offset, mem is uint8 buffer."""
  stores = []
  val_u32 = val.cast(dtypes.uint32) if val.dtype != dtypes.uint32 else val
  for i in range(data_bits // 8):
    byte_val = (val_u32 >> UOp.const(dtypes.uint32, i * 8)) & UOp.const(dtypes.uint32, 0xFF)
    stores.append(mem.index((addr + UOp.const(dtypes.uint64, i)).cast(dtypes.int), active).store(byte_val.cast(dtypes.uint8)))
  return stores

def _collect_data_slices(assigns: list[tuple[str, UOp]], data_prefix: str, pcode_vars: dict | None = None, op_name: str = "") -> dict[int, UOp]:
  """Collect bit slices from assigns into {dword_idx: value} dict."""
  slices = {}
  for dest, val in assigns:
    if dest.startswith(f'{data_prefix}['):
      if (m := re.match(rf'{data_prefix}\[(\d+)\s*:\s*(\d+)\]', dest)):
        hi_bit, low_bit = int(m.group(1)), int(m.group(2))
        dword_idx = low_bit // 32
        # D16 loads preserve bits - use final value from pcode_vars which has hi bits preserved
        if pcode_vars and 'D16' in op_name and dword_idx == 0 and hi_bit < 32:
          slices[0] = _to_u32(pcode_vars.get(data_prefix, val))
        else: slices[dword_idx] = _to_u32(val)
    elif dest.startswith(data_prefix): slices[0] = _to_u32(val)
  return slices

# ═══════════════════════════════════════════════════════════════════════════════
# INSTRUCTION COMPILER - converts decoded instruction to UOp SINK
# ═══════════════════════════════════════════════════════════════════════════════

class _Ctx:
  """Context for instruction compilation - holds buffers and helpers."""
  __slots__ = ('inst_size', 'dyn_fields', '_axis_id', 'wave_size', 'vgpr', 'accvgpr')
  sgpr = UOp(Ops.PARAM, dtypes.uint32.ptr(SGPR_COUNT), arg=0)
  vmem = UOp(Ops.PARAM, dtypes.uint32.ptr(1 << 46), arg=2)
  lds = UOp(Ops.PARAM, dtypes.uint32.ptr(16384), arg=3)
  scratch = UOp(Ops.PARAM, dtypes.uint8.ptr(1 << 30), arg=4)
  # Cache PARAM UOps by wave_size so all _Ctx instances with same wave_size share identical UOp references
  _vgpr_cache: dict[int, UOp] = {}
  _accvgpr_cache: dict[int, UOp] = {}

  def __init__(self, inst_size: int, wave_size: int = 32):
    self.inst_size, self._axis_id, self.wave_size = inst_size, 0, wave_size
    self.dyn_fields: list[tuple[int, int]] = []  # (lo, hi) of fields read dynamically
    if wave_size not in _Ctx._vgpr_cache: _Ctx._vgpr_cache[wave_size] = UOp(Ops.PARAM, dtypes.uint32.ptr(256 * wave_size), arg=1)
    self.vgpr = _Ctx._vgpr_cache[wave_size]
    if wave_size == 64:
      if wave_size not in _Ctx._accvgpr_cache: _Ctx._accvgpr_cache[wave_size] = UOp(Ops.PARAM, dtypes.uint32.ptr(256 * wave_size), arg=5)
      self.accvgpr = _Ctx._accvgpr_cache[wave_size]
    else:
      self.accvgpr = self.vgpr

  def range(self, n: int | None = None) -> UOp:
    """Create a lane range UOp with unique axis ID."""
    if n is None: n = self.wave_size
    self._axis_id += 1
    return UOp.range(n, self._axis_id, AxisType.LOOP, dtype=dtypes.int)

  def unroll_lanes(self, get_lane_bit, exec_mask: UOp, apply_exec: bool = True) -> UOp:
    """Combine lane bits into a mask using RANGE+REDUCE (32-bit for RDNA, 64-bit for CDNA)."""
    lane = self.range()
    if self.wave_size <= 32:
      bit = get_lane_bit(lane).cast(dtypes.uint32) << lane.cast(dtypes.uint32)
      result = bit.reduce(lane, arg=Ops.ADD)
    else:
      bit = get_lane_bit(lane).cast(dtypes.uint64) << lane.cast(dtypes.uint64)
      result = bit.reduce(lane, arg=Ops.ADD)
    return result & exec_mask if apply_exec else result

  def inst_word(self, dword_idx: int) -> UOp:
    """Read instruction dword from vmem at PC + dword_idx*4."""
    pc = self.rpc()
    addr = pc if dword_idx == 0 else pc + UOp.const(dtypes.uint64, dword_idx * 4)
    return self.vmem.index((addr >> UOp.const(dtypes.uint64, 2)).cast(dtypes.int), ptr=True).load()

  def inst_field(self, field) -> UOp:
    """Extract field bits from instruction encoding. Tracks field for canonical key computation."""
    lo, hi = field.lo, field.hi
    self.dyn_fields.append((lo, hi))
    dword_idx = lo // 32
    lo_in_dword = lo % 32
    hi_in_dword = hi % 32
    word = self.inst_word(dword_idx)
    if lo // 32 == hi // 32:  # Same dword
      mask = (1 << (hi - lo + 1)) - 1
      shifted = word if lo_in_dword == 0 else word >> UOp.const(dtypes.uint32, lo_in_dword)
      return shifted & UOp.const(dtypes.uint32, mask)
    else:  # Spans two dwords
      lo_bits = 32 - lo_in_dword
      lo_mask = (1 << lo_bits) - 1
      hi_mask = (1 << (hi_in_dword + 1)) - 1
      lo_part = (word >> UOp.const(dtypes.uint32, lo_in_dword)) & UOp.const(dtypes.uint32, lo_mask)
      hi_part = self.inst_word(dword_idx + 1) & UOp.const(dtypes.uint32, hi_mask)
      return lo_part | (hi_part << UOp.const(dtypes.uint32, lo_bits))

  def inst_field_signed(self, field) -> UOp:
    """Extract field and sign-extend based on field width."""
    val = self.inst_field(field)
    width = field.hi - field.lo + 1
    sign_bit = 1 << (width - 1)
    return (val.cast(dtypes.int) ^ _c(sign_bit, dtypes.int)) - _c(sign_bit, dtypes.int)

  def canonical_mask(self, inst_bytes: bytes) -> tuple[int, int, int]:
    """Compute canonical (base, mask, size) for cache lookup.
    base = instruction bits with dynamic fields zeroed
    mask = bitmask with 1s for static bits, 0s for dynamic bits
    size = instruction size in bytes"""
    size = self.inst_size
    base = int.from_bytes(inst_bytes[:size], 'little')
    mask = (1 << (size * 8)) - 1  # all 1s initially
    for lo, hi in self.dyn_fields:
      field_mask = ((1 << (hi - lo + 1)) - 1) << lo
      base &= ~field_mask  # zero dynamic bits in base
      mask &= ~field_mask  # zero dynamic bits in mask
    return base, mask, size

  def rexec(self) -> UOp:
    """Read full EXEC mask (32-bit for RDNA, 64-bit for CDNA)."""
    lo = self.rsgpr_dyn(_c(EXEC_LO.offset))
    if self.wave_size <= 32: return lo
    hi = self.rsgpr_dyn(_c(EXEC_LO.offset + 1))
    return _u64(lo, hi)

  # Dynamic register access (takes UOp index instead of int)
  def rsgpr_dyn(self, reg: UOp, valid: UOp | None = None) -> UOp:
    """Read SGPR with dynamic register index."""
    if valid is not None: return self.sgpr.index(reg.cast(dtypes.int), valid, ptr=True).load()
    return self.sgpr.index(reg.cast(dtypes.int), ptr=True).load()

  def wsgpr_dyn(self, reg: UOp, val: UOp) -> UOp:
    """Write SGPR with dynamic register index. On RDNA, index 124 = NULL (writes discarded). On CDNA, index 124 = M0 (read/write)."""
    # RDNA: NULL (124) discards writes. CDNA: M0 (124) is writable.
    valid = None if self.wave_size == 64 else reg.ne(_c(124))
    return self.sgpr.index(reg.cast(dtypes.int), valid).store(val.cast(dtypes.uint32))

  def wmask(self, reg: UOp, val: UOp) -> list[UOp]:
    """Write a lane mask (VCC/EXEC). Splits into lo/hi for wave64."""
    if self.wave_size > 32:
      lo, hi = _split64(val)
      return [self.wsgpr_dyn(reg, lo), self.wsgpr_dyn(reg + _c(1), hi)]
    return [self.wsgpr_dyn(reg, val)]

  def rmask(self, reg: UOp) -> UOp:
    """Read a lane mask (VCC/EXEC). Combines lo/hi for wave64."""
    if self.wave_size > 32: return _u64(self.rsgpr_dyn(reg), self.rsgpr_dyn(reg + _c(1)))
    return self.rsgpr_dyn(reg)

  def rvgpr_dyn(self, reg: UOp, lane: UOp, valid: UOp | None = None) -> UOp:
    """Read VGPR with dynamic register index."""
    idx = reg.cast(dtypes.int) * _c(self.wave_size, dtypes.int) + lane.cast(dtypes.int)
    return self.vgpr.index(idx, valid, ptr=True).load() if valid is not None else self.vgpr.index(idx, ptr=True).load()

  def wvgpr_dyn(self, reg: UOp, lane: UOp, val: UOp, exec_mask: UOp, after: UOp | None = None) -> UOp:
    """Write VGPR with dynamic register index."""
    buf = self.vgpr.after(after) if after is not None else self.vgpr
    offset = reg.cast(dtypes.int) * _c(self.wave_size, dtypes.int) + lane.cast(dtypes.int)
    return buf.index(offset, _lane_active(exec_mask, lane)).store(val.cast(dtypes.uint32))

  def raccvgpr_dyn(self, reg: UOp, lane: UOp, valid: UOp | None = None) -> UOp:
    """Read ACCVGPR with dynamic register index (CDNA only)."""
    idx = reg.cast(dtypes.int) * _c(self.wave_size, dtypes.int) + lane.cast(dtypes.int)
    return self.accvgpr.index(idx, valid, ptr=True).load() if valid is not None else self.accvgpr.index(idx, ptr=True).load()

  def waccvgpr_dyn(self, reg: UOp, lane: UOp, val: UOp, exec_mask: UOp, after: UOp | None = None) -> UOp:
    """Write ACCVGPR with dynamic register index (CDNA only)."""
    buf = self.accvgpr.after(after) if after is not None else self.accvgpr
    offset = reg.cast(dtypes.int) * _c(self.wave_size, dtypes.int) + lane.cast(dtypes.int)
    return buf.index(offset, _lane_active(exec_mask, lane)).store(val.cast(dtypes.uint32))

  def rsrc_dyn(self, off: UOp, lane: UOp | None, bits: int = 32, literal: UOp | None = None, is_f64: bool = False, do_cast: bool = True) -> UOp:
    """Read source operand with dynamic offset. Handles SGPR/inline constants (<256), VGPR (>=256).
    If lane is None, only scalar access is supported (off must be < 256).
    is_f64: True for F64 operations where 64-bit literals go in high 32 bits."""
    is_float_const = (off >= _c(240)) & (off <= _c(248))
    is_vgpr = off >= _c(256)
    is_sgpr = is_vgpr.ne(True)
    sgpr_lo = self.rsgpr_dyn(off, is_sgpr)

    if lane is not None:
      vgpr_reg = off - _c(256)
      vgpr_lo = self.rvgpr_dyn(vgpr_reg, lane, is_vgpr)
      vgpr_val = _u64(vgpr_lo, self.rvgpr_dyn(vgpr_reg + _c(1), lane, is_vgpr)) if bits == 64 else vgpr_lo

    if bits == 64:
      sgpr_hi = self.rsgpr_dyn(off + _c(1), is_sgpr)
      sgpr_val = _u64(sgpr_lo, sgpr_hi)
      # Integer inline constants: sign-extend 32-bit value from buffer to 64-bit
      # Float constants: cast F32 to F64
      int_inline = sgpr_lo.cast(dtypes.int32).cast(dtypes.int64)
      float_inline = sgpr_lo.bitcast(dtypes.float32).cast(dtypes.float64)
      # compute inline
      inline = is_float_const.where(float_inline.bitcast(dtypes.uint64), int_inline.bitcast(dtypes.uint64))
      # Literal handling: F64 VOP puts literal in high 32 bits; B64/I64/U64 VOP and SOP zero-extend
      if literal is not None:
        lit_val = literal.cast(dtypes.uint64) << UOp.const(dtypes.uint64, 32) if is_f64 else literal.cast(dtypes.uint64)
        inline = off.eq(_c(255)).where(lit_val, inline)
      scalar_val = (off < _c(128)).where(sgpr_val, inline)
    else:
      scalar_val = sgpr_lo
      if literal is not None: scalar_val = off.eq(_c(255)).where(literal, scalar_val)
      if bits == 16 and do_cast:  # Float constants: cast F32 to F16
        scalar_val = is_float_const.where(scalar_val.bitcast(dtypes.float32).cast(dtypes.half).bitcast(dtypes.uint16).cast(dtypes.uint32), scalar_val)

    return is_vgpr.where(vgpr_val, scalar_val) if lane is not None else scalar_val

  def rpc(self) -> UOp:
    """Read PC as 64-bit byte address."""
    # Index at PC_LO, then cast to uint64 ptr and load
    return self.sgpr.index(_c(PC_LO_IDX, dtypes.int), ptr=True).cast(dtypes.uint64.ptr(SGPR_COUNT // 2)).load()

  def inc_pc(self) -> list[UOp]:
    """Increment PC by instruction size in bytes. Returns [store]."""
    new_pc = self.rpc() + UOp.const(dtypes.uint64, self.inst_size)
    return [self.sgpr.index(_c(PC_LO_IDX, dtypes.int), ptr=True).cast(dtypes.uint64.ptr(SGPR_COUNT // 2)).store(new_pc)]

  def scalar_stores(self, assigns: list[tuple[str, UOp]], sdst_reg: UOp, sdst_size: int = 1) -> list[UOp]:
    """Generate stores for scalar assigns with dynamic destination register (D0, SCC, EXEC, VCC)."""
    stores: list[UOp] = []
    for dest, val in assigns:
      if dest.startswith('D0'):
        if sdst_size == 2:
          lo, hi = _split64(val)
          stores.extend([self.wsgpr_dyn(sdst_reg, lo), self.wsgpr_dyn(sdst_reg + _c(1), hi)])
        else: stores.append(self.wsgpr_dyn(sdst_reg, _val_to_u32(val)))
      elif dest.startswith('SCC'): stores.append(self.wsgpr_dyn(_c(SCC.offset), _to_u32(val)))
      elif dest.startswith('EXEC'):
        if self.wave_size > 32 and val.dtype in (dtypes.uint64, dtypes.int64):
          lo, hi = _split64(val)
          stores.extend([self.wsgpr_dyn(_c(EXEC_LO.offset), lo), self.wsgpr_dyn(_c(EXEC_LO.offset + 1), hi)])
        else: stores.append(self.wsgpr_dyn(_c(EXEC_LO.offset), _to_u32(val)))
      elif dest.startswith('VCC'): stores.extend(self.wmask(_c(VCC_LO.offset), val))
    return stores

  def compile_sop_pcode(self, op, srcs: dict[str, UOp | int], sdst_reg: UOp, sdst_size: int) -> UOp:
    """Compile a scalar instruction with dynamic destination register."""
    pcode = get_pcode(op)
    srcs.update({'VCC': self.rmask(_c(VCC_LO.offset)), 'EXEC': self.rexec(), 'SCC': self.rsgpr_dyn(_c(SCC.offset)),
                 '_wave_size': self.wave_size})
    if 'D0' not in srcs: srcs['D0'] = self.rsgpr_dyn(sdst_reg)  # D0 is current dest value for read-modify-write ops
    _, assigns = parse_pcode(pcode, srcs)
    return UOp.sink(*self.scalar_stores(assigns, sdst_reg, sdst_size), *self.inc_pc())

  def compile_lane_pcode(self, op, inst) -> UOp:
    """Compile cross-lane ops (READLANE/WRITELANE/PERMLANE) using pcode parser."""
    pcode = get_pcode(op)
    op_name = op.name if hasattr(op, 'name') else str(op)
    src0_off, vdst_off = self.inst_field(type(inst).src0), self.inst_field(type(inst).vdst)
    src0_reg = (src0_off >= _c(256)).where(src0_off - _c(256), _c(0))  # VGPR index or 0
    src1_off = self.inst_field(type(inst).src1) if hasattr(type(inst), 'src1') else None
    src2_off = self.inst_field(type(inst).src2) if hasattr(type(inst), 'src2') else None
    exec_val = self.rexec()
    exec_lo = exec_val.cast(dtypes.uint32) if exec_val.dtype == dtypes.uint64 else exec_val
    srcs = {
      'SRC0': src0_reg, 'VDST': vdst_off, 'EXEC_LO': exec_lo, 'EXEC': exec_val if exec_val.dtype == dtypes.uint64 else exec_val.cast(dtypes.uint64),
      '_vgpr': self.vgpr, '_wave_size': self.wave_size,
      'S0': self.rsrc_dyn(src0_off, _c(0, dtypes.int)) if 'WRITELANE' in op_name else src0_reg,
      'S1': self.rsrc_dyn(src1_off, _c(0, dtypes.int)) if src1_off is not None else _c(0),
      'S2': self.rsrc_dyn(src2_off, _c(0, dtypes.int)) if src2_off is not None else _c(0),
    }
    _, assigns = parse_pcode(pcode, srcs)
    stores = []
    for dest, val in assigns:
      if dest.startswith('D0'): stores.append(self.wsgpr_dyn(vdst_off, val.cast(dtypes.uint32)))
      elif dest.startswith('VGPR['): stores.append(self.vgpr.index(val[0].cast(dtypes.int)).store(val[1].cast(dtypes.uint32)))
    return UOp.sink(*stores, *self.inc_pc())

  def compile_vop_pcode(self, op, srcs: dict[str, UOp | int], lane: UOp, vdst_reg: UOp, exec_mask: UOp,
                        opsel_dst_hi: bool | UOp = False, sdst_reg: int | None = None, clmp: int = 0,
                        src0_off: UOp | None = None) -> UOp:
    """Compile VOP instruction. Returns sink with stores and inc_pc."""
    pcode = get_pcode(op)
    vcc_reg = sdst_reg if sdst_reg is not None else VCC_LO.offset
    if 'VCC' not in srcs: srcs['VCC'] = self.rmask(_c(vcc_reg))
    srcs.update({'EXEC': exec_mask, 'SCC': self.rsgpr_dyn(_c(SCC.offset)), 'laneId': lane, 'VDST': vdst_reg,
                 'ROUND_MODE': _c(0), 'ROUND_TOWARD_ZERO': _c(0), 'ROUND_NEAREST_EVEN': _c(0), '_vgpr': self.vgpr, '_wave_size': self.wave_size,
                 # CDNA SDWA byte/word select constants (E32 always uses BYTE0/WORD0 defaults)
                 'SDWA_SRC0_SEL': _c(0), 'BYTE0': _c(0), 'BYTE1': _c(1), 'BYTE2': _c(2), 'BYTE3': _c(3),
                 'WORD0': _c(0), 'WORD1': _c(1)})  # rounding mode and SDWA constants
    _, assigns = parse_pcode(pcode, srcs)

    # For integer ops with clamp, compute overflow using wide arithmetic
    # NOTE: MUL_LO ops don't saturate - they always return the low bits
    int_saturate = None
    if clmp and any(p in op.name for p in ('_NC_U', '_MAD_U', '_NC_I', '_MAD_I')):
      is_signed, is_16bit = '_I' in op.name and '_U' not in op.name, '16' in op.name
      if not (is_16bit and is_signed):  # Skip 16-bit signed ops due to codegen issues
        s0, s1, s2 = srcs.get('S0'), srcs.get('S1'), srcs.get('S2')
        if s0 is not None and s1 is not None:
          narrow_dt = dtypes.uint16 if is_16bit else (dtypes.int32 if is_signed else dtypes.uint32)
          wide_dt = dtypes.int32 if is_16bit else dtypes.int64
          narrow_max, narrow_min = (0xFFFF, 0) if is_16bit else ((0x7FFFFFFF, -0x80000000) if is_signed else (0xFFFFFFFF, 0))
          def to_wide(x): return (x.bitcast(narrow_dt) if x.dtype.itemsize == narrow_dt.itemsize else x.cast(narrow_dt)).cast(wide_dt)
          is_sub, is_mad = 'SUB' in op.name, 'MAD' in op.name
          full = (to_wide(s0) * to_wide(s1) + to_wide(s2)) if is_mad and s2 is not None else \
                 (to_wide(s1) - to_wide(s0)) if is_sub and 'SUBREV' in op.name else \
                 (to_wide(s0) - to_wide(s1)) if is_sub else (to_wide(s0) + to_wide(s1))
          int_saturate = full.clamp(narrow_min, narrow_max).cast(narrow_dt)
    # V_SUB_U32 / V_ADD_U32 with clamp: unsigned saturate (SUB underflow->0, ADD overflow->0xFFFFFFFF)
    if clmp and int_saturate is None and any(p in op.name for p in ('_SUB_U32', '_ADD_U32', '_SUB_U16', '_ADD_U16')):
      s0, s1 = srcs.get('S0'), srcs.get('S1')
      if s0 is not None and s1 is not None:
        assert isinstance(s0, UOp) and isinstance(s1, UOp)
        a, b = (s1.cast(dtypes.uint32), s0.cast(dtypes.uint32)) if 'SUBREV' in op.name else (s0.cast(dtypes.uint32), s1.cast(dtypes.uint32))
        if 'SUB' in op.name:
          int_saturate = (a < b).where(_c(0), a - b)  # underflow -> 0
        else:
          raw_sum = a + b
          int_saturate = (raw_sum < a).where(_c(0xFFFFFFFF), raw_sum)  # overflow -> MAX

    raw_stores: list = []
    vcc_val, exec_val = None, None
    for dest, val in assigns:
      # VGPR bit-slice assignment: VGPR[lane][reg][hi:lo] = (vgpr_idx, rhs_val, hi, lo[, cond]) -> read-modify-write
      if dest.startswith('VGPR[') and re.search(r'\[\d+:\d+\]', dest):
        # VGPR bit-slice: (vgpr_idx, rhs_val, hi_bit, lo_bit) - hi/lo are UOp constants
        hi_bit, lo_bit = int(val[2].arg), int(val[3].arg)
        width = hi_bit - lo_bit + 1
        old = self.vgpr.index(val[0].cast(dtypes.int), ptr=True).load()
        new_val = _set_bits(old, _val_to_bits(val[1]), width, lo_bit).cast(dtypes.uint32)
        active = _lane_active(exec_mask, lane)
        raw_stores.append(('vgpr_direct', self.vgpr.index(val[0].cast(dtypes.int), active).store(new_val)))
        continue
      if 'D0' in dest and '[laneId]' in dest:
        old_vcc = self.rmask(_c(VCC_LO.offset))
        new_vcc = _set_lane_bit(old_vcc, lane, val, exec_mask)
        raw_stores.extend([('vcc', s) for s in self.wmask(_c(VCC_LO.offset), new_vcc)])
      elif dest.startswith('D0'):
        if (slice_match := re.match(r'D0\[(\d+)\s*:\s*(\d+)\]', dest)):
          d0_hi_bit, d0_lo_bit = int(slice_match.group(1)), int(slice_match.group(2))
          if d0_hi_bit != 31 or d0_lo_bit != 0:
            d0_width, slice_mask = d0_hi_bit - d0_lo_bit + 1, (1 << (d0_hi_bit - d0_lo_bit + 1)) - 1
            val_bits = val.bitcast(dtypes.uint16).cast(dtypes.uint32) if val.dtype == dtypes.half else \
                       val.cast(dtypes.uint32) if val.dtype in (dtypes.uint16, dtypes.int16) else \
                       val.cast(dtypes.uint32) & UOp.const(dtypes.uint32, slice_mask)
            raw_stores.append(('vgpr_slice', (d0_lo_bit, d0_width, val_bits)))
            continue
        # For integer ops with clamp, use pre-computed saturated value; for floats, clamp to [0,1]
        if int_saturate is not None: val = int_saturate
        elif clmp and val.dtype in (dtypes.float32, dtypes.half, dtypes.float64):
          val = val.maximum(UOp.const(val.dtype, 0.0)).minimum(UOp.const(val.dtype, 1.0))
        if val.dtype in (dtypes.uint64, dtypes.int64, dtypes.float64):
          lo, hi = _split64(val)
          raw_stores.extend([('vgpr', self.wvgpr_dyn(vdst_reg, lane, lo, exec_mask)),
                             ('vgpr', self.wvgpr_dyn(vdst_reg + _c(1), lane, hi, exec_mask))])
        elif val.dtype in (dtypes.half, dtypes.uint16, dtypes.int16):
          result, old_val = _val_to_u32(val), self.rvgpr_dyn(vdst_reg, lane)
          hi_result = (old_val & UOp.const(dtypes.uint32, 0xFFFF)) | (result << UOp.const(dtypes.uint32, 16))
          # GFX9/CDNA zeroes upper 16 bits on lo-half write; RDNA preserves them
          lo_result = (result & UOp.const(dtypes.uint32, 0xFFFF)) if self.wave_size == 64 else \
                      (old_val & UOp.const(dtypes.uint32, 0xFFFF0000)) | (result & UOp.const(dtypes.uint32, 0xFFFF))
          result = opsel_dst_hi.where(hi_result, lo_result) if isinstance(opsel_dst_hi, UOp) else hi_result if opsel_dst_hi else lo_result
          raw_stores.append(('vgpr', self.wvgpr_dyn(vdst_reg, lane, result, exec_mask)))
        else: raw_stores.append(('vgpr', self.wvgpr_dyn(vdst_reg, lane, _val_to_u32(val), exec_mask)))
      elif dest.startswith('S0') and src0_off is not None:
        # Write back to src0 VGPR (e.g. v_swap_b32). src0_off is raw encoding (256+ = VGPR)
        src0_vgpr = src0_off - _c(256)
        raw_stores.append(('vgpr_s0', self.wvgpr_dyn(src0_vgpr, lane, _val_to_u32(val), exec_mask)))
      elif dest.startswith('VCC'): vcc_val = val
      elif dest.startswith('EXEC'): exec_val = val
      elif dest.startswith('SCC'): raw_stores.append(('scc', self.wsgpr_dyn(_c(SCC.offset), _to_u32(val))))

    lane_stores = [s for t, s in raw_stores if t in ('vgpr', 'vgpr_s0', 'vgpr_direct')]
    stores, scalar_stores = [], [s for t, s in raw_stores if t == 'scc']
    slice_stores = [s for t, s in raw_stores if t == 'vgpr_slice']
    if slice_stores:
      result = self.rvgpr_dyn(vdst_reg, lane)
      for lo_bit, width, val_bits in slice_stores:
        mask = UOp.const(dtypes.uint32, ((1 << width) - 1) << lo_bit)
        result = (result & (mask ^ UOp.const(dtypes.uint32, 0xFFFFFFFF))) | (val_bits << UOp.const(dtypes.uint32, lo_bit))
      lane_stores.append(self.wvgpr_dyn(vdst_reg, lane, result, exec_mask))
    # VCC/EXEC mask writes must be computed BEFORE VGPR stores to avoid reading modified VGPRs.
    # When vdst overlaps with src operands (e.g. v_add_co_u32 v[0], vcc, s[8], v[0]), the carry
    # computation reads the original source values only if its range loop runs before the VGPR write loop.
    mask_stores: list[UOp] = []
    for mask_val, reg in [(vcc_val, vcc_reg), (exec_val, EXEC_LO.offset)]:
      if mask_val is None: continue
      def get_bit(l, v=mask_val): return (_to_u32(v.substitute({lane: l})) & _c(1)).cast(dtypes.uint32)
      mask_stores.extend(self.wmask(_c(reg), self.unroll_lanes(get_bit, exec_mask, apply_exec=False)))
    stores.extend(mask_stores)
    if lane_stores: stores.append(UOp.sink(*lane_stores).end(lane))
    stores.extend(scalar_stores)
    return UOp.sink(*stores, *self.inc_pc())

# ═══════════════════════════════════════════════════════════════════════════════
# INSTRUCTION HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

def _compile_sopp(inst: ir3.SOPP | ir4.SOPP, ctx: _Ctx) -> UOp:
  simm16 = ctx.inst_field_signed(type(inst).simm16).cast(dtypes.int16)
  if inst.op in (ir3.SOPPOp.S_ENDPGM, ir4.SOPPOp.S_ENDPGM, irc.SOPPOp.S_ENDPGM):
    return UOp.sink(ctx.wsgpr_dyn(_c(PC_LO_IDX), UOp.const(dtypes.uint32, 0xFFFFFFFF)),
                          ctx.wsgpr_dyn(_c(PC_HI_IDX), UOp.const(dtypes.uint32, 0xFFFFFFFF)))
  # S_BARRIER: advance PC past the barrier instruction. The execution loop detects barriers before executing and handles synchronization.
  barrier_ops = {ir3.SOPPOp.S_BARRIER, irc.SOPPOp.S_BARRIER}
  if hasattr(ir4.SOPPOp, 'S_BARRIER_WAIT'): barrier_ops.add(ir4.SOPPOp.S_BARRIER_WAIT)
  if inst.op in barrier_ops: return UOp.sink(*ctx.inc_pc())
  # S_NOP and S_WAITCNT are no-ops in emulator (no pipeline/cache to wait on)
  if inst.op in (ir3.SOPPOp.S_NOP, ir4.SOPPOp.S_NOP, irc.SOPPOp.S_NOP, irc.SOPPOp.S_WAITCNT): return UOp.sink(*ctx.inc_pc())
  # NOTE: we ignore SOPPs without PCODE
  if inst.op in _get_pcode_dict(inst.op):
    pcode = get_pcode(inst.op)
    pc_bytes = ctx.rpc()  # PC is already 64-bit byte address
    vcc, exec_val = ctx.rmask(_c(VCC_LO.offset)), ctx.rexec()
    srcs = {'PC': pc_bytes.cast(dtypes.int64), 'SIMM16': simm16, 'SCC': ctx.rsgpr_dyn(_c(SCC.offset)), 'VCC': vcc,
            'VCCZ': vcc.eq(UOp.const(vcc.dtype, 0)).cast(dtypes.uint32),
            'EXECZ': exec_val.eq(UOp.const(exec_val.dtype, 0)).cast(dtypes.uint32)}
    for dest, val in parse_pcode(pcode, srcs)[1]:
      if dest == 'PC' or dest.startswith('PC.'):
        lo, hi = _split64(val.cast(dtypes.uint64))
        return UOp.sink(ctx.wsgpr_dyn(_c(PC_LO_IDX), lo), ctx.wsgpr_dyn(_c(PC_HI_IDX), hi))
  return UOp.sink(*ctx.inc_pc())

def _compile_smem(inst: ir3.SMEM | ir4.SMEM, ctx: _Ctx) -> UOp:
  # Cache invalidation instructions are no-ops in the emulator (we don't model caches)
  if '_INV' in inst.op.name: return UOp.sink(*ctx.inc_pc())
  # Dynamic sbase field (bits 5:0) - SGPR pair, field value * 2 = register offset
  sbase = ctx.inst_field(type(inst).sbase) * _c(2)
  # Dynamic sdata field (bits 12:6) - destination SGPR
  sdata_reg = ctx.inst_field(type(inst).sdata)
  # RDNA4 uses 'ioffset', RDNA3 uses 'offset' - use type(inst) to get correct field
  offset_field = type(inst).ioffset if hasattr(type(inst), 'ioffset') else type(inst).offset  # type: ignore[union-attr]
  offset = ctx.inst_field_signed(offset_field)  # signed immediate
  # Dynamic soffset field - SGPR for additional offset (NULL=124 reads as 0, CDNA soffset_en=0 means no soffset)
  soffset_val = _c(0).cast(dtypes.uint64)
  if not (isinstance(inst, irc.SMEM) and not inst.soffset_en):
    soffset_val = ctx.rsgpr_dyn(ctx.inst_field(type(inst).soffset)).cast(dtypes.uint64)
  addr = _u64(ctx.rsgpr_dyn(sbase), ctx.rsgpr_dyn(sbase + _c(1))) + offset.cast(dtypes.uint64) + soffset_val
  # S_LOAD_(DTYPE) series: B32/DWORD=1, B64/DWORDX2=2, U8=0.25, I8=-0.25, etc.
  op_name = _op_name(inst)
  assert (op_name).startswith('S_LOAD_'), f"unexpected SMEM op: {op_name}"
  part = op_name.rsplit('_', 1)[1]  # B32, DWORD, DWORDX2, U8, I8, etc.
  nval = int(part.removeprefix('DWORD').removeprefix('X') or '1') if 'DWORD' in part else int(part[1:]) / 32 * (-1 if part[0] == 'I' else 1)
  ndwords = max(1, int(abs(nval)))
  dword_base = addr >> UOp.const(dtypes.uint64, 2)
  vals = [ctx.vmem.index((dword_base + UOp.const(dtypes.uint64, i)).cast(dtypes.int)) for i in range(ndwords)]
  if abs(nval) < 1:
    nbits = int(abs(nval) * 32)
    byte_off = (addr & UOp.const(dtypes.uint64, 3)).cast(dtypes.uint32) * UOp.const(dtypes.uint32, 8)
    extracted = (vals[0] >> byte_off) & UOp.const(dtypes.uint32, (1 << nbits) - 1)
    vals[0] = extracted.cast({8: dtypes.int8, 16: dtypes.int16}[nbits]).cast(dtypes.int32).bitcast(dtypes.uint32) if nval < 0 else extracted
  stores = [ctx.wsgpr_dyn(sdata_reg + _c(i), vals[i]) for i in range(ndwords)]
  return UOp.sink(*stores, *ctx.inc_pc())

def _compile_sop(inst: ir3.SOP1|ir3.SOP2|ir3.SOPC|ir3.SOPK|ir4.SOP1|ir4.SOP2|ir4.SOPC|ir4.SOPK|irc.SOP1|irc.SOP2|irc.SOPC|irc.SOPK, ctx: _Ctx) -> UOp:
  bits = inst.canonical_op_bits
  literal = ctx.inst_field(type(inst).literal) if hasattr(type(inst), 'literal') else None  # type: ignore[union-attr]

  if isinstance(inst, (ir3.SOPK, ir4.SOPK, irc.SOPK)):
    sdst_off = ctx.inst_field(type(inst).sdst)
    simm16 = ctx.inst_field(type(inst).simm16)
    # Sign-extend simm16
    simm16_sext = simm16.cast(dtypes.int16).cast(dtypes.int32)
    # RDNA4 pcodes use S0.i16 for the immediate (e.g., S_MULK_I32), RDNA3 uses S0 for the register (e.g., S_CMPK_*)
    # CDNA pcode uses S0 for the immediate in MOVK/MULK/ADDK/CMOVK, but S0 = register for CMPK/SETREG
    op_name = _op_name(inst)
    if isinstance(inst, ir4.SOPK): s0 = simm16
    elif isinstance(inst, irc.SOPK) and 'CMPK' not in op_name and 'SETREG' not in op_name: s0 = simm16_sext
    else: s0 = ctx.rsgpr_dyn(sdst_off)
    srcs = {'S0': s0, 'S1': simm16_sext, 'SIMM16': simm16_sext, 'D0': ctx.rsgpr_dyn(sdst_off)}
    dst_off, dst_size = sdst_off, 1
    # S_GETREG_B32: extract bits from HW register. Handle as special case since HW_REGISTERS is not a normal variable.
    # HW register values are stored at SGPR[SGPR_COUNT-16 + hwRegId] by _init_wave.
    if 'GETREG' in op_name:
      hw_reg_id = simm16.cast(dtypes.uint32) & _c(0x3F)
      offset = (simm16.cast(dtypes.uint32) >> _c(6)) & _c(0x1F)
      size = ((simm16.cast(dtypes.uint32) >> _c(11)) & _c(0x1F)) + _c(1)
      hw_val = ctx.rsgpr_dyn(_c(SGPR_COUNT - 16) + hw_reg_id)
      mask = (_c(1) << size) - _c(1)
      result = (hw_val >> offset) & mask
      return UOp.sink(ctx.wsgpr_dyn(sdst_off, result), *ctx.inc_pc())
  elif isinstance(inst, (ir3.SOP1, ir4.SOP1, irc.SOP1)):
    # S_BARRIER_SIGNAL: no-op in emulator, barrier sync handled by execution loop
    if isinstance(inst, ir4.SOP1) and inst.op in _BARRIER_SOP1_OPS: return UOp.sink(*ctx.inc_pc())
    sdst_off = ctx.inst_field(type(inst).sdst)
    ssrc0_off = ctx.inst_field(type(inst).ssrc0)
    srcs = {'S0': ctx.rsrc_dyn(ssrc0_off, None, bits['s0'], literal)}
    dst_off, dst_size = sdst_off, bits['d'] // 32
  elif isinstance(inst, (ir3.SOP2, ir4.SOP2, irc.SOP2)):
    sdst_off = ctx.inst_field(type(inst).sdst)
    ssrc0_off = ctx.inst_field(type(inst).ssrc0)
    ssrc1_off = ctx.inst_field(type(inst).ssrc1)
    srcs = {'S0': ctx.rsrc_dyn(ssrc0_off, None, bits['s0'], literal),
            'S1': ctx.rsrc_dyn(ssrc1_off, None, bits['s1'], literal)}
    if literal is not None: srcs['SIMM32'] = literal
    dst_off, dst_size = sdst_off, bits['d'] // 32
  elif isinstance(inst, (ir3.SOPC, ir4.SOPC, irc.SOPC)):
    ssrc0_off = ctx.inst_field(type(inst).ssrc0)
    ssrc1_off = ctx.inst_field(type(inst).ssrc1)
    srcs = {'S0': ctx.rsrc_dyn(ssrc0_off, None, bits['s0'], literal),
            'S1': ctx.rsrc_dyn(ssrc1_off, None, bits['s1'], literal)}
    dst_off, dst_size = _c(0), 0  # SOPC writes to SCC, not sdst
  else:
    raise RuntimeError(f"unknown SOP type: {type(inst).__name__}")

  return ctx.compile_sop_pcode(inst.op, srcs, dst_off, dst_size)

def _sdwa_select(val: UOp, sel: UOp, sext: UOp) -> UOp:
  """Apply SDWA byte/word selection and optional sign extension to a 32-bit value."""
  # sel: 0-3=BYTE_0..3, 4=WORD_0, 5=WORD_1, 6=DWORD
  b0 = val & _c(0xFF)
  b1 = (val >> _c(8)) & _c(0xFF)
  b2 = (val >> _c(16)) & _c(0xFF)
  b3 = (val >> _c(24)) & _c(0xFF)
  w0 = val & _c(0xFFFF)
  w1 = (val >> _c(16)) & _c(0xFFFF)
  selected = sel.eq(_c(1)).where(b1, sel.eq(_c(2)).where(b2, sel.eq(_c(3)).where(b3,
    sel.eq(_c(4)).where(w0, sel.eq(_c(5)).where(w1, sel.eq(_c(6)).where(val, b0))))))
  # Sign extend when sext=1
  is_byte = sel < _c(4)
  byte_sext = (selected & _c(0x80)).ne(_c(0)).where(selected | _c(0xFFFFFF00), selected)
  word_sext = (selected & _c(0x8000)).ne(_c(0)).where(selected | _c(0xFFFF0000), selected)
  return sext.ne(_c(0)).where(is_byte.where(byte_sext, word_sext), selected)

def _sdwa_write(old: UOp, val: UOp, dst_sel: UOp, dst_unused: UOp) -> UOp:
  """Apply SDWA destination selection: write selected byte/word, handle unused bits."""
  # dst_unused: 0=PAD(zero), 1=SEXT, 2=PRESERVE
  # dst_sel: 0-3=BYTE, 4=WORD_0, 5=WORD_1, 6=DWORD
  is_byte = dst_sel < _c(4)
  is_word = (dst_sel >= _c(4)) & (dst_sel < _c(6))
  shift = is_byte.where(dst_sel * _c(8), (dst_sel - _c(4)) * _c(16))
  mask = is_byte.where(_c(0xFF), is_word.where(_c(0xFFFF), _c(0xFFFFFFFF)))
  placed = (val & mask) << shift
  preserve_mask = (mask << shift) ^ _c(0xFFFFFFFF)
  preserved = (old & preserve_mask) | placed
  # For PAD and SEXT, unused bits are zero (PAD) or sign-extended (SEXT). For DWORD, just return val.
  return dst_sel.eq(_c(6)).where(val, dst_unused.eq(_c(2)).where(preserved, placed))

def _compile_sdwa(inst: irc.VOP1_SDWA | irc.VOP2_SDWA | irc.VOP2_SDWA_SDST | irc.VOPC_SDWA_SDST, ctx: _Ctx) -> UOp:
  """Compile CDNA SDWA (Sub-Dword Access) VOP1/VOP2/VOPC instructions."""
  is_vopc = isinstance(inst, irc.VOPC_SDWA_SDST)
  exec_mask = ctx.rexec()
  # sd=1 means use sdst register, sd=0 means use VCC (for VOPC_SDWA_SDST and VOP2_SDWA_SDST)
  if isinstance(inst, (irc.VOP2_SDWA_SDST, irc.VOPC_SDWA_SDST)):
    sdst_off = _c(inst.sdst.offset) if getattr(inst, 'sd', False) else _c(VCC_LO.offset)
  else:
    sdst_off = _c(VCC_LO.offset)
  # Read SDWA fields (these are dynamic but shared across lanes)
  src0_sel = ctx.inst_field(type(inst).src0_sel)
  src0_sext = ctx.inst_field(type(inst).src0_sext)
  vsrc0_reg = ctx.inst_field(type(inst).vsrc0)
  pcode = get_pcode(inst.op)
  if isinstance(inst, (irc.VOP2_SDWA, irc.VOP2_SDWA_SDST, irc.VOPC_SDWA_SDST)):
    src1_sel = ctx.inst_field(type(inst).src1_sel)
    src1_sext = ctx.inst_field(type(inst).src1_sext)
    vsrc1_reg = ctx.inst_field(type(inst).vsrc1)

  # For VOPC: use unroll_lanes to build the bitmask from scratch (no read-modify-write on stale data)
  if is_vopc:
    def get_cmp_bit(lane) -> UOp:
      lc = lane.cast(dtypes.int) if isinstance(lane, UOp) else _c(lane, dtypes.int)
      s0_raw = ctx.rsgpr_dyn(vsrc0_reg) if inst.s0 else ctx.rvgpr_dyn(vsrc0_reg, lc)
      s0 = _sdwa_select(s0_raw, src0_sel, src0_sext)
      s1_raw = ctx.rsgpr_dyn(vsrc1_reg) if inst.s1 else ctx.rvgpr_dyn(vsrc1_reg, lc)
      s1 = _sdwa_select(s1_raw, src1_sel, src1_sext)
      srcs = {'S0': s0, 'S1': s1, 'laneId': lc}
      for dest, val in parse_pcode(pcode, srcs)[1]:
        if '[laneId]' in dest and ('D0' in dest or 'EXEC' in dest): return val.cast(dtypes.uint32)
      return _c(0)
    new_result = ctx.unroll_lanes(get_cmp_bit, exec_mask, apply_exec=False) & exec_mask
    stores = ctx.wmask(sdst_off, new_result)
    return UOp.sink(*stores, *ctx.inc_pc())

  # Non-VOPC path: VOP1_SDWA, VOP2_SDWA, VOP2_SDWA_SDST — uses lane loop
  lane = ctx.range()
  vdst_reg = ctx.inst_field(type(inst).vdst)  # type: ignore[union-attr]
  s0_raw = ctx.rsgpr_dyn(vsrc0_reg) if inst.s0 else ctx.rvgpr_dyn(vsrc0_reg, lane)
  s0 = _sdwa_select(s0_raw, src0_sel, src0_sext)
  if isinstance(inst, (irc.VOP2_SDWA, irc.VOP2_SDWA_SDST)):
    s1_raw = ctx.rsgpr_dyn(vsrc1_reg) if inst.s1 else ctx.rvgpr_dyn(vsrc1_reg, lane)
    s1 = _sdwa_select(s1_raw, src1_sel, src1_sext)
    srcs:dict[str, UOp | int] = {'S0': s0, 'S1': s1, 'D0': ctx.rvgpr_dyn(vdst_reg, lane)}
  else:
    srcs = {'S0': s0}
  # dst_sel and dst_unused
  has_dst_sel = hasattr(type(inst), 'dst_sel')
  if has_dst_sel:
    dst_sel = ctx.inst_field(type(inst).dst_sel)  # type: ignore[union-attr]
    dst_unused = ctx.inst_field(type(inst).dst_unused)  # type: ignore[union-attr]
  srcs.update({'VCC': ctx.rmask(_c(VCC_LO.offset)), 'EXEC': exec_mask, 'SCC': ctx.rsgpr_dyn(_c(SCC.offset)),
               'laneId': lane, 'VDST': vdst_reg, 'ROUND_MODE': _c(0), 'ROUND_TOWARD_ZERO': _c(0),
               'ROUND_NEAREST_EVEN': _c(0), '_vgpr': ctx.vgpr, '_wave_size': ctx.wave_size,
               'SDWA_SRC0_SEL': _c(0), 'BYTE0': _c(0), 'BYTE1': _c(1), 'BYTE2': _c(2), 'BYTE3': _c(3),
               'WORD0': _c(0), 'WORD1': _c(1)})
  _, assigns = parse_pcode(pcode, srcs)
  stores = []
  vcc_val = None
  for dest, val in assigns:
    if 'D0' in dest and '[laneId]' in dest:
      vcc_val = val
    elif dest.startswith('D0'):
      result = _val_to_u32(val)
      if has_dst_sel:
        old = ctx.rvgpr_dyn(vdst_reg, lane)
        result = _sdwa_write(old, result, dst_sel, dst_unused)
      stores.append(ctx.wvgpr_dyn(vdst_reg, lane, result, exec_mask))
    elif dest.startswith('VCC'):
      old_vcc = ctx.rmask(_c(VCC_LO.offset))
      stores.extend(ctx.wmask(_c(VCC_LO.offset), _set_lane_bit(old_vcc, lane, val, exec_mask)))
  if vcc_val is not None:
    # Initialize sdst to 0 before lane loop (old value may be unrelated data), then set lane bits in loop
    init_stores = [ctx.wsgpr_dyn(sdst_off, _c(0)), ctx.wsgpr_dyn(sdst_off + _c(1), _c(0))]
    old_sdst = ctx.rmask(sdst_off)
    stores.extend(ctx.wmask(sdst_off, _set_lane_bit(old_sdst, lane, vcc_val, exec_mask)))
    if stores:
      return UOp.sink(*init_stores, UOp.sink(*stores).end(lane), *ctx.inc_pc())
    return UOp.sink(*init_stores, *ctx.inc_pc())
  if stores:
    return UOp.sink(UOp.sink(*stores).end(lane), *ctx.inc_pc())
  return UOp.sink(*ctx.inc_pc())

def _compile_vop12(inst: ir3.VOP1 | ir3.VOP1_SDST | ir3.VOP2 | ir4.VOP1 | ir4.VOP1_SDST | ir4.VOP2 | irc.VOP1 | irc.VOP2, ctx: _Ctx) -> UOp:
  op_name = _op_name(inst)
  if op_name in ('V_READFIRSTLANE_B32_E32', 'V_PERMLANE64_B32_E32'): return ctx.compile_lane_pcode(inst.op, inst)
  # v_accvgpr_mov_b32: ACCVGPR[vdst] = ACCVGPR[src0] (VOP1 encoding, no pcode)
  if 'ACCVGPR_MOV' in op_name:
    lane, exec_mask = ctx.range(), ctx.rexec()
    vdst_reg = ctx.inst_field(type(inst).vdst)  # VGPRField: raw ACCVGPR index (0-255)
    src0_off = ctx.inst_field(type(inst).src0)  # SrcField: raw 256 + ACCVGPR index
    val = ctx.raccvgpr_dyn(src0_off - _c(256), lane)
    return UOp.sink(ctx.waccvgpr_dyn(vdst_reg, lane, val, exec_mask).end(lane), *ctx.inc_pc())
  lane, exec_mask, bits = ctx.range(), ctx.rexec(), inst.canonical_op_bits
  literal = ctx.inst_field(type(inst).literal) if hasattr(type(inst), 'literal') else None  # type: ignore[union-attr]
  is_f64 = 'F64' in op_name and 'B64' not in op_name
  vdst_reg = ctx.inst_field(type(inst).vdst)
  write_hi_half = bits['d'] == 16 and (vdst_reg >= _c(128))
  if isinstance(write_hi_half, UOp): vdst_reg = write_hi_half.where(vdst_reg - _c(128), vdst_reg)
  elif write_hi_half: vdst_reg -= 128
  if isinstance(inst, (ir3.VOP1, ir4.VOP1, irc.VOP1)):
    # Handle VOP1 hi-half source operand (src0 >= v[128] for 16-bit ops)
    src0_off = ctx.inst_field(type(inst).src0)
    s0 = ctx.rsrc_dyn(src0_off, lane, bits['s0'], literal, is_f64)
    if bits['s0'] == 16:
      src0_hi = src0_off >= _c(384)
      # Only compute hi-half when src0_off >= 384, use guarded index to prevent OOB access
      src0_reg = src0_hi.where(src0_off - _c(384), _c(0))
      s0 = src0_hi.where(_hi16(ctx.rvgpr_dyn(src0_reg, lane)), s0)
    d0 = _cond_hi16(write_hi_half, ctx.rvgpr_dyn(vdst_reg, lane))
    srcs:dict[str, UOp | int] = {'S0': s0, 'D0': d0}
  else:
    vsrc1_reg = ctx.inst_field(type(inst).vsrc1)
    vsrc1_hi = bits['s0'] == 16 and (vsrc1_reg >= _c(128))
    vsrc1_actual = _cond(vsrc1_hi, vsrc1_reg - _c(128), vsrc1_reg)
    if bits['s1'] == 64:
      s1 = _u64(ctx.rvgpr_dyn(vsrc1_reg, lane), ctx.rvgpr_dyn(vsrc1_reg + _c(1), lane))
      d0 = _u64(ctx.rvgpr_dyn(vdst_reg, lane), ctx.rvgpr_dyn(vdst_reg + _c(1), lane))
    else:
      s1 = _cond_hi16(vsrc1_hi, ctx.rvgpr_dyn(vsrc1_actual, lane))
      d0 = _cond_hi16(write_hi_half, ctx.rvgpr_dyn(vdst_reg, lane))  # FMAC/FMAMK hi-half dest needs hi-half accumulator
    # Handle VOP2 hi-half src0 operand (src0 >= v[128] for 16-bit ops)
    src0_off = ctx.inst_field(type(inst).src0)
    s0 = ctx.rsrc_dyn(src0_off, lane, bits['s0'], literal, is_f64)
    if bits['s0'] == 16:
      src0_hi = src0_off >= _c(384)
      # Only compute hi-half when src0_off >= 384, use guarded index to prevent OOB access
      src0_reg = src0_hi.where(src0_off - _c(384), _c(0))
      s0 = src0_hi.where(_hi16(ctx.rvgpr_dyn(src0_reg, lane)), s0)
    srcs = {'S0': s0, 'S1': s1, 'D0': d0}
    # FMAAK_(DTYPE)_E32 series
    if 'V_FMAA' in _op_name(inst) or 'V_FMAM' in _op_name(inst):
      assert literal is not None
      srcs['SIMM32'] = literal
  return ctx.compile_vop_pcode(inst.op, srcs, lane, vdst_reg, exec_mask, opsel_dst_hi=write_hi_half, src0_off=src0_off)

def _compile_vopc(inst: ir3.VOPC|ir3.VOP3|ir4.VOPC|ir4.VOP3|irc.VOPC|irc.VOP3, ctx: _Ctx,
                  opsel: int = 0, abs_bits: int = 0, neg_bits: int = 0) -> UOp:
  exec_mask, op_name, bits = ctx.rexec(), _op_name(inst), inst.canonical_op_bits
  is_cmpx, is_vopc = 'CMPX' in op_name, hasattr(inst, 'vsrc1')  # is_vopc: e32 vs e64

  # Handle both VOPC (vsrc1) and VOP3 (src1) instruction formats - read operands dynamically
  if is_vopc:
    src0_off = ctx.inst_field(type(inst).src0)
    vsrc1_off = ctx.inst_field(type(inst).vsrc1)  # type: ignore[union-attr]
    # For 16-bit ops, vsrc1 >= 128 means hi-half of v[vsrc1-128]
    if bits['s0'] == 16:
      vsrc1_hi = vsrc1_off >= _c(128)
      src1_off = _c(256) + vsrc1_hi.where(vsrc1_off - _c(128), vsrc1_off)
    else:
      vsrc1_hi = False
      src1_off = _c(256) + vsrc1_off
  else:
    src0_off = ctx.inst_field(type(inst).src0)
    src1_off = ctx.inst_field(type(inst).src1)  # type: ignore[union-attr]
    dst_off = ctx.inst_field(type(inst).vdst)  # type: ignore[union-attr]
    vsrc1_hi = False
  literal = ctx.inst_field(type(inst).literal) if hasattr(type(inst), 'literal') else None  # type: ignore[union-attr]

  is_float, is_f64, pcode = any(x in op_name for x in ('_F32', '_F64', '_F16')), '_F64' in op_name, get_pcode(inst.op)
  def get_cmp_bit(lane) -> UOp:
    lc = lane.cast(dtypes.int) if isinstance(lane, UOp) else _c(lane, dtypes.int)
    s0 = ctx.rsrc_dyn(src0_off, lc, bits['s0'], literal, is_f64)
    s1 = _cond_hi16(vsrc1_hi, ctx.rsrc_dyn(src1_off, lc, bits['s1'], literal, is_f64)) if bits['s0'] == 16 \
      else ctx.rsrc_dyn(src1_off, lc, bits['s1'], literal, is_f64)
    if bits['s0'] == 16 and opsel: s0, s1 = _apply_opsel(s0, 0, opsel), _apply_opsel(s1, 1, opsel)
    if is_float:
      s0 = _apply_src_mods(s0, 0, abs_bits, neg_bits, bits['s0'])
      s1 = _apply_src_mods(s1, 1, abs_bits, neg_bits, bits['s1'])
    for dest, val in parse_pcode(pcode, {'S0': s0, 'S1': s1, 'laneId': lc, 'D0': UOp.const(dtypes.uint64, 0)})[1]:
      if '[laneId]' in dest and ('D0' in dest or 'EXEC' in dest): return val.cast(dtypes.uint32)
    return _c(0)

  new_bits = ctx.unroll_lanes(get_cmp_bit, exec_mask, apply_exec=False)
  # Both VOPC and VOP3 clear inactive lane bits (hardware verified)
  new_result = new_bits & exec_mask

  # CMPX e32: writes EXEC only; CMPX e64: writes both EXEC and SDST; non-CMPX: writes dst only
  if is_cmpx:
    stores = ctx.wmask(_c(EXEC_LO.offset), new_result)
    if not is_vopc: stores.extend(ctx.wmask(dst_off, new_result))
  else:
    stores = ctx.wmask(dst_off, new_result) if not is_vopc else ctx.wmask(_c(VCC_LO.offset), new_result)
  return UOp.sink(*stores, *ctx.inc_pc())


def _compile_bitop3(inst, ctx: _Ctx, exec_mask: UOp, bits: dict, op_name: str) -> UOp:
  """BITOP3: 3-input truth table. abs/neg/omod encode the truth table, not source modifiers."""
  lane = ctx.range()
  vdst_reg = ctx.inst_field(type(inst).vdst)
  ops = inst.canonical_operands
  src0 = ctx.rsrc_dyn(ctx.inst_field(type(inst).src0), lane, bits['s0'], None, 's0' in ops and ops['s0'][0] == Fmt.FMT_NUM_F64)
  src1 = ctx.rsrc_dyn(ctx.inst_field(type(inst).src1), lane, bits['s1'], None, 's1' in ops and ops['s1'][0] == Fmt.FMT_NUM_F64)
  src2 = ctx.rsrc_dyn(ctx.inst_field(type(inst).src2), lane, bits['s2'], None, 's2' in ops and ops['s2'][0] == Fmt.FMT_NUM_F64)
  # Truth table: TTBL = { omod[1:0], abs[2:0], neg[2:0] } = 8-bit LUT
  ttbl = ((getattr(inst, 'omod', 0) or 0) << 6) | ((getattr(inst, 'abs', 0) or 0) << 3) | (getattr(inst, 'neg', 0) or 0)
  is_16 = 'B16' in op_name
  dt, mask = (dtypes.uint16, 0xFFFF) if is_16 else (dtypes.uint32, 0xFFFFFFFF)
  s0, s1, s2 = src0.cast(dt), src1.cast(dt), src2.cast(dt)
  def bnot(v): return v ^ UOp.const(dt, mask)
  result = UOp.const(dt, 0)
  for i in range(8):
    if not (ttbl & (1 << i)): continue
    result = result | ((s0 if i & 4 else bnot(s0)) & (s1 if i & 2 else bnot(s1)) & (s2 if i & 1 else bnot(s2)))
  return UOp.sink(ctx.wvgpr_dyn(vdst_reg, lane, result.cast(dtypes.uint32), exec_mask).end(lane), *ctx.inc_pc())

def _compile_vop3(inst: ir3.VOP3 | ir4.VOP3 | irc.VOP3, ctx: _Ctx) -> UOp:
  exec_mask = ctx.rexec()
  bits = inst.canonical_op_bits
  opsel, op_name = getattr(inst, 'opsel', 0) or 0, _op_name(inst)

  # Lane operations
  if op_name in ('V_READLANE_B32', 'V_READFIRSTLANE_B32', 'V_READFIRSTLANE_B32_E64', 'V_WRITELANE_B32'):
    return ctx.compile_lane_pcode(inst.op, inst)

  # V_PERMLANE16_B32 / V_PERMLANEX16_B32: cross-lane swizzle via pcode
  if 'PERMLANE16' in op_name or 'PERMLANEX16' in op_name:
    return ctx.compile_lane_pcode(inst.op, inst)

   # VOP3 VOPC (v_cmp_*_e64) - delegate to unified VOPC handler
  if 'V_CMP' in op_name or 'V_CMPX' in op_name:
    return _compile_vopc(inst, ctx, opsel=opsel, abs_bits=getattr(inst, 'abs', 0) or 0, neg_bits=getattr(inst, 'neg', 0) or 0)

  # BITOP3: abs/neg/omod encode truth table, not source modifiers
  if 'BITOP3' in op_name:
    return _compile_bitop3(inst, ctx, exec_mask, bits, op_name)

  # VOP3 specific fields
  vdst_reg = ctx.inst_field(type(inst).vdst)
  literal = ctx.inst_field(type(inst).literal) if hasattr(type(inst), 'literal') else None  # type: ignore[union-attr]
  abs_bits, neg_bits = getattr(inst, 'abs', 0) or 0, getattr(inst, 'neg', 0) or 0

  # VOP3_SDST: v_s_* instructions goes to SGPR
  if 'V_S_' in op_name:
    src0 = _apply_src_mods(ctx.rsrc_dyn(ctx.inst_field(type(inst).src0), _c(0, dtypes.int), bits['s0'], literal), 0, abs_bits, neg_bits, bits['s0'])
    srcs = {'S0': src0, 'EXEC': exec_mask, 'SCC': ctx.rsgpr_dyn(_c(SCC.offset)), 'laneId': _c(0, dtypes.int),
            'ROUND_MODE': _c(0), 'ROUND_TOWARD_ZERO': _c(0)}
    _, assigns = parse_pcode(get_pcode(inst.op), srcs)
    stores = [ctx.wsgpr_dyn(vdst_reg, _val_to_u32(val)) for dest, val in assigns if dest.startswith('D0')]
    return UOp.sink(*stores, *ctx.inc_pc())

  # Regular VOP3 - read operands dynamically
  lane = ctx.range()
  ops = inst.canonical_operands
  src0 = ctx.rsrc_dyn(ctx.inst_field(type(inst).src0), lane, bits['s0'], literal, 's0' in ops and ops['s0'][0] == Fmt.FMT_NUM_F64)
  src1 = ctx.rsrc_dyn(ctx.inst_field(type(inst).src1), lane, bits['s1'], literal, 's1' in ops and ops['s1'][0] == Fmt.FMT_NUM_F64)
  src2 = ctx.rsrc_dyn(ctx.inst_field(type(inst).src2), lane, bits['s2'], literal, 's2' in ops and ops['s2'][0] == Fmt.FMT_NUM_F64)
  if bits['s0'] == 16:
    src0 = _apply_opsel(src0, 0, opsel)
    src1 = _apply_opsel(src1, 1, opsel)
    src2 = _apply_opsel(src2, 2, opsel)
  src0 = _apply_src_mods(src0, 0, abs_bits, neg_bits, bits['s0'])
  src1 = _apply_src_mods(src1, 1, abs_bits, neg_bits, bits['s1'])
  src2 = _apply_src_mods(src2, 2, abs_bits, neg_bits, bits['s2'])
  srcs = {'S0': src0, 'S1': src1, 'S2': src2, 'OPSEL': UOp.const(dtypes.uint32, opsel)}
  if 'CNDMASK' in op_name and src2 is not None: srcs['VCC'] = src2
  # FMAC instructions need D0 (accumulator) from destination register
  if 'FMAC' in op_name: srcs['D0'] = ctx.rvgpr_dyn(vdst_reg, lane)
  opsel_dst_hi = bool(opsel & 0b1000) and bits['d'] == 16
  return ctx.compile_vop_pcode(inst.op, srcs, lane, vdst_reg, exec_mask, opsel_dst_hi=opsel_dst_hi, clmp=getattr(inst, 'clmp', 0))

def _compile_vop3sd(inst: ir3.VOP3SD | ir4.VOP3SD | irc.VOP3SD, ctx: _Ctx) -> UOp:
  exec_mask = ctx.rexec()
  bits, pcode, ops = inst.canonical_op_bits, get_pcode(inst.op), inst.canonical_operands

  # Read operands dynamically from instruction encoding
  vdst_reg, sdst_off = ctx.inst_field(type(inst).vdst), ctx.inst_field(type(inst).sdst)
  src0_off, src1_off, src2_off = ctx.inst_field(type(inst).src0), ctx.inst_field(type(inst).src1), ctx.inst_field(type(inst).src2)
  literal = ctx.inst_field(type(inst).literal) if hasattr(type(inst), 'literal') else None  # type: ignore[union-attr]

  has_carry_in = 's2' in ops and ops['s2'][2] == OpType.OPR_SREG
  vcc_in_off = src2_off if has_carry_in else sdst_off

  def load_srcs(lane_uop):
    ret = {'VCC': ctx.rmask(vcc_in_off), 'EXEC': exec_mask, 'SCC': ctx.rsgpr_dyn(_c(SCC.offset)), 'laneId': lane_uop}
    ret['S0'] = ctx.rsrc_dyn(src0_off, lane_uop, bits['s0'], literal, ops['s0'][0] == Fmt.FMT_NUM_F64)
    ret['S1'] = ctx.rsrc_dyn(src1_off, lane_uop, bits['s1'], literal, ops['s1'][0] == Fmt.FMT_NUM_F64)
    if 's2' in ops: ret['S2'] = ctx.rsrc_dyn(src2_off, lane_uop, bits['s2'], literal, ops['s2'][0] == Fmt.FMT_NUM_F64)
    return ret

  lane = ctx.range()
  srcs = load_srcs(lane)
  _, assigns = parse_pcode(pcode, srcs)

  has_per_lane_vcc = any('[laneId]' in dest for dest, _ in assigns if dest.startswith('VCC') or dest.startswith('D0.u64'))
  clmp = getattr(inst, 'clmp', 0)
  if has_per_lane_vcc:
    # VCC computation: RANGE+REDUCE gets axis ID first (lower ID = runs first)
    # This ensures VCC reads source values BEFORE VGPR stores modify them
    def get_vcc_bit(lane_uop) -> UOp:
      vcc_bit = _c(0)
      for dest, val in parse_pcode(pcode, load_srcs(lane_uop))[1]:
        if dest.startswith('VCC') or (dest.startswith('D0.u64') and '[laneId]' in dest): vcc_bit = val.cast(dtypes.uint32)
      return vcc_bit
    final_vcc = ctx.unroll_lanes(get_vcc_bit, exec_mask)
    # VGPR stores: RANGE gets axis ID second (higher ID = runs after VCC loop)
    lane3 = ctx.range()
    d0_val, vcc_per_lane = None, None
    for dest, val in parse_pcode(pcode, load_srcs(lane3))[1]:
      if dest.startswith('D0') and '[laneId]' not in dest: d0_val = val
      if dest.startswith('VCC') or (dest.startswith('D0.u64') and '[laneId]' in dest): vcc_per_lane = val
    vgpr_stores = []
    if d0_val is not None:
      # Apply clamp using carry/borrow bit: ADD overflow->0xFFFFFFFF, SUB underflow->0
      if clmp and vcc_per_lane is not None:
        is_sub = 'SUB' in inst.op.name
        sat_val = _c(0) if is_sub else _c(0xFFFFFFFF)
        d0_val = vcc_per_lane.cast(dtypes.bool).where(sat_val, d0_val.cast(dtypes.uint32))
      if d0_val.dtype in (dtypes.uint64, dtypes.int64, dtypes.float64):
        lo, hi = _split64(d0_val)
        vgpr_stores.extend([ctx.wvgpr_dyn(vdst_reg, lane3, lo, exec_mask), ctx.wvgpr_dyn(vdst_reg + _c(1), lane3, hi, exec_mask)])
      else:
        d0_u32 = d0_val.bitcast(dtypes.uint32) if d0_val.dtype in (dtypes.float32, dtypes.half) else d0_val.cast(dtypes.uint32)
        vgpr_stores.append(ctx.wvgpr_dyn(vdst_reg, lane3, d0_u32, exec_mask))
    # Write carry output (wmask handles lo/hi split for wave64)
    vcc_writes = ctx.wmask(sdst_off, final_vcc)
    return UOp.sink(*vcc_writes, UOp.group(*vgpr_stores).end(lane3), *ctx.inc_pc())
  else:
    return ctx.compile_vop_pcode(inst.op, srcs, lane, vdst_reg, exec_mask, sdst_reg=inst.sdst.offset)

def _compile_mfma(inst: irc.VOP3P, ctx: _Ctx) -> UOp:
  """CDNA MFMA matrix multiply-accumulate emulation.

  Uses local temp arrays to cache inputs, avoiding aliasing issues when vdst overlaps src0/src1.
  Phase 1: Read all input f32 values from VGPRs into temp arrays (range loop over 64 lanes).
  Phase 2: Compute 256 output values using temp arrays and write to VGPRs (range loop over 64 lanes)

  Register layout (wave64):
  - 16x16: 4 groups of 16 lanes. Each lane in group holds k_per_grp elements. 4 output ACCVGPRs per lane.
  - 32x32: 2 groups of 32 lanes. lanes%16 = M/N index within block, lanes//16 selects block. 16 output ACCVGPRs per lane.
  - 4x4: 16 groups of 4 lanes. 4 output ACCVGPRs per lane.
  """
  import re as _re
  op_name = _op_name(inst)
  exec_mask = ctx.rexec()
  vdst_reg = ctx.inst_field(type(inst).vdst)
  src0_off = ctx.inst_field(type(inst).src0)
  src1_off = ctx.inst_field(type(inst).src1)
  src0_r = src0_off - _c(256)  # VGPR-relative index (only valid when src is VGPR)
  src1_r = src1_off - _c(256)
  src2_off = ctx.inst_field(type(inst).src2)
  # Check if sources are VGPRs (offset >= 256) vs inline constants/SGPRs
  src0_is_vgpr = src0_off >= _c(256)
  src1_is_vgpr = src1_off >= _c(256)

  m = _re.search(r'(\d+)X(\d+)X(\d+)', op_name)
  if m is None: raise ValueError(f"could not parse MFMA dimensions from {op_name}")
  M, N, K = int(m.group(1)), int(m.group(2)), int(m.group(3))

  is_bf16 = 'BF16' in op_name
  is_fp8 = 'FP8' in op_name or 'F8' in op_name
  is_i8 = 'I8' in op_name
  # Source type is the LAST type in the name: V_MFMA_F32_16X16X32_**F16** -> source is F16, not F32
  src_type = op_name.rsplit('_', 1)[-1]  # e.g. "F16", "BF16", "F32", "I8"
  is_f32_src = src_type == 'F32'
  is_int_out = 'I32' in op_name.split('_')[2]  # V_MFMA_I32_...

  # Determine elements per VGPR and conversion function
  if is_i8: vpg = 4
  elif is_f32_src: vpg = 1
  elif is_fp8: vpg = 4
  else: vpg = 2

  # For 16x16: grp_size=16, n_grps=4, out_per_lane=4
  # For 32x32: grp_size=32, n_grps=2, out_per_lane=16
  # For 4x4: grp_size=4, n_grps=16, out_per_lane=4
  if M == 16 and N == 16:
    grp_size, n_grps, out_per_lane = 16, 4, 4
  elif M == 32 and N == 32:
    grp_size, n_grps, out_per_lane = 32, 2, 16
  elif M == 4 and N == 4:
    grp_size, n_grps, out_per_lane = 4, 16, 4
  else:
    raise RuntimeError(f"unsupported MFMA shape {M}x{N}x{K}")

  # For 4x4: each group independently computes a 4x4 block. K is NOT split across groups.
  # For 16x16/32x32: K IS split across groups (each group has K/n_grps elements).
  k_per_grp = K if M == 4 else K // n_grps
  # Temp array size: for 4x4, store all 16 independent blocks; for others, store shared MxK/NxK
  n_a_elems = n_grps * M * K if M == 4 else M * K
  n_b_elems = n_grps * N * K if M == 4 else N * K

  # src2 can be VGPR (>=256) or inline constant/SGPR (<256)
  src2_is_vgpr = src2_off >= _c(256)
  src2_r = src2_off - _c(256)
  if is_int_out:
    acc_scalar = ctx.rsgpr_dyn(src2_off, src2_is_vgpr.ne(True)).cast(dtypes.int32)
  else:
    acc_scalar = ctx.rsgpr_dyn(src2_off, src2_is_vgpr.ne(True)).bitcast(dtypes.float32)

  # Phase 1: Read all A and B values from VGPRs into temp arrays.
  # Layout: tmp[0..n_a_elems-1] = A[m][k], tmp[n_a_elems..n_a_elems+n_b_elems-1] = B[n][k]
  # Within each group of lanes, lane%grp_sub gives M/N index, lane//grp_sub gives sub-block
  grp_sub = min(M, 16)  # lanes within group mapped to M/N dimension
  b_off = UOp.const(dtypes.int, n_a_elems)
  acc_dt = dtypes.int32 if is_int_out else dtypes.float32
  # Use uint32 temp array to prevent optimizer from eliminating f16→f32 bitcast chains.
  # The optimizer folds bitcast(uint32→float32) stores to float32 arrays, losing the conversion.
  tmp = UOp(Ops.DEFINE_LOCAL, dtypes.uint32.ptr(n_a_elems + n_b_elems, addrspace=AddrSpace.LOCAL), arg=(n_a_elems + n_b_elems,))

  def cvt_elem(raw: UOp, sub_idx: int) -> UOp:
    if is_i8:
      # Extract i8, sign-extend to i32
      byte_val = (raw >> UOp.const(dtypes.uint32, sub_idx * 8)) & UOp.const(dtypes.uint32, 0xFF)
      return (byte_val.cast(dtypes.int32) ^ UOp.const(dtypes.int32, 0x80)) - UOp.const(dtypes.int32, 0x80)
    elif is_f32_src:
      return raw  # already uint32 (f32 bit pattern)
    elif is_fp8:
      return ((raw >> UOp.const(dtypes.uint32, sub_idx * 8)) & UOp.const(dtypes.uint32, 0xFF)).cast(dtypes.uint32)
    elif is_bf16:
      # bf16→f32 bits: just shift left by 16 (bf16 is upper 16 bits of f32)
      return ((raw >> UOp.const(dtypes.uint32, sub_idx * 16)) & UOp.const(dtypes.uint32, 0xFFFF)) << UOp.const(dtypes.uint32, 16)
    else:
      # f16→f32 conversion using float arithmetic to avoid UOp optimizer eliminating the conversion.
      # The optimizer folds bitcast(uint32→float32) chains, so we compute the float value directly.
      h = (raw >> UOp.const(dtypes.uint32, sub_idx * 16)) & UOp.const(dtypes.uint32, 0xFFFF)
      sign = (h >> UOp.const(dtypes.uint32, 15)) & UOp.const(dtypes.uint32, 1)
      exp = (h >> UOp.const(dtypes.uint32, 10)) & UOp.const(dtypes.uint32, 0x1F)
      mant = h & UOp.const(dtypes.uint32, 0x3FF)
      # Use bf16 path: shift left by 16 to create bf16 bits, then shift mantissa and adjust exponent in float domain
      # bf16 bits = (sign << 15) | (exp_bf16 << 7) | mant_bf16 -- but f16 and bf16 have different formats
      # Instead: construct f32 bits properly, use a DEFINE_LOCAL uint32 array to force materialization
      f32_bits = (sign << UOp.const(dtypes.uint32, 31)) | \
                 ((exp + UOp.const(dtypes.uint32, 112)) << UOp.const(dtypes.uint32, 23)) | \
                 (mant << UOp.const(dtypes.uint32, 13))
      is_zero = exp.eq(UOp.const(dtypes.uint32, 0))
      # Return uint32 (f32 bit pattern) — stored directly to uint32 temp array, bitcast to float on read
      return is_zero.where(UOp.const(dtypes.uint32, 0), f32_bits)

  read_lane = ctx.range()
  # For 32x32: lane%16 = M/N index within 16-wide block, lane//16 = which of 4 quarter-waves
  # Groups: lanes 0-31 = group 0, lanes 32-63 = group 1
  # Within group: (lane%32)%16 = M/N[0-15], (lane%32)//16 selects M/N[0-15] or [16-31]
  lane_in_grp = read_lane % UOp.const(dtypes.int, grp_size)
  grp_idx = read_lane // UOp.const(dtypes.int, grp_size)

  if M == 32:
    # 32x32: lane_in_grp%16 = sub-row/col (0-15), lane_in_grp//16 = block (0=rows 0-15, 1=rows 16-31)
    sub_mn = lane_in_grp % UOp.const(dtypes.int, 16)
    block_mn = lane_in_grp // UOp.const(dtypes.int, 16)
    mn_idx = block_mn * UOp.const(dtypes.int, 16) + sub_mn  # actual M/N index (0-31)
  else:
    mn_idx = lane_in_grp  # for 16x16 and 4x4

  read_stores = []
  for kl in range(k_per_grp):
    reg_idx, sub_idx = kl // vpg, kl % vpg
    # Read A/B sources. Use rsrc_dyn for inline constants/SGPRs (src_off < 256), rvgpr_dyn for VGPRs (src_off >= 256).
    a_raw = src0_is_vgpr.where(ctx.rvgpr_dyn(src0_r + _c(reg_idx), read_lane),
                                ctx.rsrc_dyn(src0_off, _c(0, dtypes.int), 32))
    a_val = cvt_elem(a_raw, sub_idx)
    if M == 4:
      a_idx = grp_idx * UOp.const(dtypes.int, M * K) + mn_idx * UOp.const(dtypes.int, K) + UOp.const(dtypes.int, kl)
    else:
      a_idx = mn_idx * UOp.const(dtypes.int, K) + grp_idx * UOp.const(dtypes.int, k_per_grp) + UOp.const(dtypes.int, kl)
    read_stores.append(tmp.index(a_idx).store(a_val))

    b_raw = src1_is_vgpr.where(ctx.rvgpr_dyn(src1_r + _c(reg_idx), read_lane),
                                ctx.rsrc_dyn(src1_off, _c(0, dtypes.int), 32))
    b_val = cvt_elem(b_raw, sub_idx)
    if M == 4:
      b_idx = b_off + grp_idx * UOp.const(dtypes.int, N * K) + mn_idx * UOp.const(dtypes.int, K) + UOp.const(dtypes.int, kl)
    else:
      b_idx = b_off + mn_idx * UOp.const(dtypes.int, K) + grp_idx * UOp.const(dtypes.int, k_per_grp) + UOp.const(dtypes.int, kl)
    read_stores.append(tmp.index(b_idx).store(b_val))

  read_phase = UOp.group(*read_stores).end(read_lane)

  # Phase 2: Compute dot products and write outputs.
  # For 16x16: each lane computes 4 outputs. n_idx = lane%16, grp selects which 4 rows.
  # For 32x32: each lane computes 16 outputs. Layout: lane%16 selects n within block, lane//16 selects column block.
  #   Output mapping: out_reg r at lane l -> D[m][n] where
  #   n = (l%32)%16 + ((l%32)//16)*16, m = (l//32)*4 + r (for r in 0..3), with 4 groups of 4 rows -> 16 outputs total
  #   Actually: 16 ACCVGPRs per lane, organized as 4 groups (l//32 gives half, each half has 2 sub-groups) of 4 rows
  tmp2 = tmp.after(read_phase)

  compute_lane = ctx.range()
  compute_stores = []

  if M == 32 and N == 32:
    # 32x32: each lane has 16 output ACCVGPRs
    # Lane mapping: n = (lane%32)%16 + ((lane%32)//16)*16, gives column 0-31
    # Row groups: 4 groups of 4, covering rows 0-31. Group g (0-3): rows g*4 .. g*4+3
    # group assignment: lane//16 gives quarter (0-3), each quarter maps to 4 rows
    c_lane_in_32 = compute_lane % UOp.const(dtypes.int, 32)
    c_sub = c_lane_in_32 % UOp.const(dtypes.int, 16)
    c_block = c_lane_in_32 // UOp.const(dtypes.int, 16)
    n_idx = c_block * UOp.const(dtypes.int, 16) + c_sub
    c_half = compute_lane // UOp.const(dtypes.int, 32)  # 0 or 1

    for out_reg in range(16):
      # Each half covers 8 rows. out_reg 0-3: rows 0-3 (half0) or 16-19 (half1)
      # out_reg 4-7: rows 4-7 (half0) or 20-23 (half1), etc.
      # Actually: for 32x32, the output layout per lane is:
      # acc[0:3] -> rows 0-3 (half 0) or rows 0-3 (half 1)?
      # Let me use the ISA doc: for 32x32, D has 16 dwords per lane. The mapping is:
      # acc[r] at lane l -> D[m][n] where n = (l%32)%16 + ((l%32)//16)*16
      # m = (l//32)*16 + (r//4)*4 + (r%4)  ... giving rows in blocks of 4
      # So: m_base = half * 16 + (out_reg // 4) * 4 + (out_reg % 4)
      m_base = c_half * UOp.const(dtypes.int, 16) + UOp.const(dtypes.int, (out_reg // 4) * 4 + (out_reg % 4))

      acc_v = ctx.raccvgpr_dyn(src2_r + _c(out_reg), compute_lane, src2_is_vgpr)
      if is_int_out: acc_v = acc_v.cast(dtypes.int32)
      else: acc_v = acc_v.bitcast(dtypes.float32)
      acc = src2_is_vgpr.where(acc_v, acc_scalar)

      for k in range(K):
        a_val = tmp2.index(m_base * UOp.const(dtypes.int, K) + UOp.const(dtypes.int, k)).bitcast(acc_dt)
        b_val = tmp2.index(b_off + n_idx * UOp.const(dtypes.int, K) + UOp.const(dtypes.int, k)).bitcast(acc_dt)
        acc = acc + a_val * b_val

      if is_int_out:
        compute_stores.append(ctx.waccvgpr_dyn(vdst_reg + _c(out_reg), compute_lane, acc.cast(dtypes.uint32), exec_mask))
      else:
        compute_stores.append(ctx.waccvgpr_dyn(vdst_reg + _c(out_reg), compute_lane, acc.bitcast(dtypes.uint32), exec_mask))
  else:
    # 16x16 and 4x4: each lane computes out_per_lane outputs
    n_idx = compute_lane % UOp.const(dtypes.int, grp_sub)
    c_grp = compute_lane // UOp.const(dtypes.int, grp_sub)

    for out_reg in range(out_per_lane):
      acc_v = ctx.raccvgpr_dyn(src2_r + _c(out_reg), compute_lane, src2_is_vgpr)
      if is_int_out: acc_v = acc_v.cast(dtypes.int32)
      else: acc_v = acc_v.bitcast(dtypes.float32)
      acc = src2_is_vgpr.where(acc_v, acc_scalar)

      if M == 4:
        # 4x4: each group is independent. A/B indexed per-group.
        m_base = c_grp * UOp.const(dtypes.int, M * K) + UOp.const(dtypes.int, out_reg * K)
        for k in range(K):
          a_val = tmp2.index(m_base + UOp.const(dtypes.int, k)).bitcast(acc_dt)
          b_val = tmp2.index(b_off + c_grp * UOp.const(dtypes.int, N*K) + n_idx * UOp.const(dtypes.int, K)+UOp.const(dtypes.int, k)).bitcast(acc_dt)
          acc = acc + a_val * b_val
      else:
        # 16x16: K is split across groups. Shared MxK/NxK arrays.
        m_base = c_grp * UOp.const(dtypes.int, out_per_lane) + UOp.const(dtypes.int, out_reg)
        for k in range(K):
          a_val = tmp2.index(m_base * UOp.const(dtypes.int, K) + UOp.const(dtypes.int, k)).bitcast(acc_dt)
          b_val = tmp2.index(b_off + n_idx * UOp.const(dtypes.int, K) + UOp.const(dtypes.int, k)).bitcast(acc_dt)
          acc = acc + a_val * b_val

      if is_int_out:
        compute_stores.append(ctx.waccvgpr_dyn(vdst_reg + _c(out_reg), compute_lane, acc.cast(dtypes.uint32), exec_mask))
      else:
        compute_stores.append(ctx.waccvgpr_dyn(vdst_reg + _c(out_reg), compute_lane, acc.bitcast(dtypes.uint32), exec_mask))

  compute_phase = UOp.group(*compute_stores).end(compute_lane)
  return UOp.sink(read_phase, compute_phase, *ctx.inc_pc())

def _compile_wmma(inst: ir3.VOP3P | ir4.VOP3P | irc.VOP3P, ctx: _Ctx) -> UOp:
  op_name = _op_name(inst)
  exec_mask = ctx.rexec()
  vdst_reg = ctx.inst_field(type(inst).vdst)
  src0_r = ctx.inst_field(type(inst).src0) - _c(256)
  src1_r = ctx.inst_field(type(inst).src1) - _c(256)
  src2_r = ctx.inst_field(type(inst).src2) - _c(256)
  is_f16_output = 'F16_16X16X16_F16' in op_name or 'BF16_16X16X16_BF16' in op_name  # F16/BF16 output vs F32 output
  is_bf16 = 'BF16' in op_name
  cvt = _FUNCS['bf16_to_f32'] if is_bf16 else _FUNCS['f16_to_f32']
  is_rdna4 = isinstance(inst, ir4.VOP3P)
  # read 16x16 F16/BF16 matrix from VGPRs → flat f32 array[row*16+k]
  def read_f16_val(src, lane, vgpr, half):
    v = ctx.rvgpr_dyn(src + _c(vgpr), UOp.const(dtypes.int, lane))
    return cvt((v >> UOp.const(dtypes.uint32, 16)) if half else (v & UOp.const(dtypes.uint32, 0xFFFF)))

  # RDNA3: 16 lanes × 8 VGPRs × 2 halves, k maps linearly
  # RDNA4: 32 lanes × 4 VGPRs × 2 halves, k bits are scrambled (k[2] goes to lane bit 4)
  def read_f16_mat(src):
  # (row, k) → (lane, vgpr, half)
    def ab_map(i, k):
      elem, lane = ((k & 3) | ((k >> 1) & 4), i + ((k >> 2) & 1) * 16) if is_rdna4 else (k, i)
      return lane, elem // 2, elem % 2
    return [read_f16_val(src, *ab_map(row, k)) for row in range(16) for k in range(16)]
  mat_a, mat_b = read_f16_mat(src0_r), read_f16_mat(src1_r)
  # (row, col) -> (lane, vgpr)
  def d_map(m, n):
    lane_bit, vgpr = (m >> 3, m & 7) if is_rdna4 else (m & 1, m >> 1)
    return n + lane_bit * 16, vgpr
  if is_f16_output:
    # read accumulator C with f16 layout: for RDNA4, pairs of f32 vgprs pack into one f16 vgpr
    # for RDNA3, same layout as f32 but only lo 16 bits used
    mat_c = [read_f16_val(src2_r, *((lane, vgpr // 2, vgpr % 2) if is_rdna4 else (lane, vgpr, 0)))
             for m in range(16) for n in range(16) for lane, vgpr in [d_map(m, n)]]
    mat_d = [sum(mat_a[r*16+k] * mat_b[c*16+k] for k in range(16)) + mat_c[r*16+c] for r in range(16) for c in range(16)]
    def f32_to_f16_bits(v: UOp) -> UOp: return v.cast(dtypes.half).bitcast(dtypes.uint16).cast(dtypes.uint32)
    def f32_to_bf16_bits(v: UOp) -> UOp: return (v.bitcast(dtypes.uint32) >> UOp.const(dtypes.uint32, 16)) & UOp.const(dtypes.uint32, 0xFFFF)
    out_cvt = f32_to_bf16_bits if is_bf16 else f32_to_f16_bits
    if is_rdna4:  # pack 2 f16 per VGPR: adjacent m values share (lane, vgpr) since vgpr=m&7, half=m&1
      stores = [ctx.wvgpr_dyn(vdst_reg + _c(d_map(m, n)[1] // 2), UOp.const(dtypes.int, d_map(m, n)[0]),
                out_cvt(mat_d[m*16+n]) | (out_cvt(mat_d[(m+1)*16+n]) << UOp.const(dtypes.uint32, 16)), exec_mask)
                for n in range(16) for m in range(0, 16, 2)]
    else:  # (rdna3) 1 f16 per VGPR (lo half only)
      stores = [ctx.wvgpr_dyn(vdst_reg + _c(d_map(m, n)[1]), UOp.const(dtypes.int, d_map(m, n)[0]), out_cvt(mat_d[m*16+n]), exec_mask)
                for m in range(16) for n in range(16)]
  else: # f32
    mat_c = [ctx.rvgpr_dyn(src2_r + _c(d_map(m, n)[1]), UOp.const(dtypes.int, d_map(m, n)[0])).bitcast(dtypes.float32)
             for m in range(16) for n in range(16)]
    mat_d = [sum(mat_a[r*16+k] * mat_b[c*16+k] for k in range(16)) + mat_c[r*16+c] for r in range(16) for c in range(16)]
    stores = [ctx.wvgpr_dyn(vdst_reg + _c(d_map(m, n)[1]), UOp.const(dtypes.int, d_map(m, n)[0]), mat_d[m*16+n].bitcast(dtypes.uint32), exec_mask)
              for m in range(16) for n in range(16)]
  return UOp.sink(*stores, *ctx.inc_pc())

def _compile_vop3p(inst: ir3.VOP3P | ir4.VOP3P | irc.VOP3P, ctx: _Ctx) -> UOp:
  op_name = _op_name(inst)
  if 'WMMA' in op_name and ('16X16X16_F16' in op_name or '16X16X16_BF16' in op_name): return _compile_wmma(inst, ctx)
  if 'MFMA' in op_name and any(f'{s}X{s}X' in op_name for s in ('4', '16', '32')) and isinstance(inst, irc.VOP3P): return _compile_mfma(inst, ctx)

  # ACCVGPR_WRITE/READ/MOV: copies between VGPR and ACCVGPR register files
  # Detect by checking operand types for ACCVGPR involvement
  ops = inst.operands
  src0_is_acc = ops.get('src0', (None, None, None))[2] in (OpType.OPR_SRC_ACCVGPR, OpType.OPR_ACCVGPR)
  vdst_is_acc = ops.get('vdst', (None, None, None))[2] in (OpType.OPR_ACCVGPR,)
  if src0_is_acc or vdst_is_acc:
    lane = ctx.range()
    exec_mask = ctx.rexec()
    vdst_reg = ctx.inst_field(type(inst).vdst)
    src0_off = ctx.inst_field(type(inst).src0)
    if src0_is_acc and not vdst_is_acc:
      # v_accvgpr_read: VGPR[vdst] = ACCVGPR[src0]
      val = ctx.raccvgpr_dyn(src0_off - _c(256), lane)
      return UOp.sink(ctx.wvgpr_dyn(vdst_reg, lane, val, exec_mask).end(lane), *ctx.inc_pc())
    elif vdst_is_acc and not src0_is_acc:
      # v_accvgpr_write: ACCVGPR[vdst] = src0 (src0 can be VGPR or SGPR/const)
      src0 = ctx.rsrc_dyn(src0_off, lane, 32)
      return UOp.sink(ctx.waccvgpr_dyn(vdst_reg, lane, src0, exec_mask).end(lane), *ctx.inc_pc())
    else:
      # v_accvgpr_mov: ACCVGPR[vdst] = ACCVGPR[src0]
      val = ctx.raccvgpr_dyn(src0_off - _c(256), lane)
      return UOp.sink(ctx.waccvgpr_dyn(vdst_reg, lane, val, exec_mask).end(lane), *ctx.inc_pc())

  lane = ctx.range()
  exec_mask = ctx.rexec()
  vdst_reg = ctx.inst_field(type(inst).vdst)
  is_pk_f32 = 'PK' in op_name and 'F32' in op_name and 'MOV' not in op_name  # CDNA packed F32 ops
  is_pk_mov_b32 = 'PK_MOV_B32' in op_name  # CDNA packed MOV needs special handling
  do_cast = any(x in op_name for x in ('F16', 'F32', 'BF16')) and 'IU' not in op_name and not is_pk_f32
  literal = ctx.inst_field(type(inst).literal) if hasattr(type(inst), 'literal') else None  # type: ignore[union-attr]
  src0 = ctx.rsrc_dyn(ctx.inst_field(type(inst).src0), lane, 16, literal=literal, do_cast=do_cast)
  src1 = ctx.rsrc_dyn(ctx.inst_field(type(inst).src1), lane, 16, literal=literal, do_cast=do_cast)
  src2 = ctx.rsrc_dyn(ctx.inst_field(type(inst).src2), lane, 16, literal=literal, do_cast=do_cast)
  opsel, opsel_hi = getattr(inst, 'opsel', 0) or 0, getattr(inst, 'opsel_hi', 3) if getattr(inst, 'opsel_hi', 3) is not None else 3
  opsel_hi2 = getattr(inst, 'opsel_hi2', 1) if getattr(inst, 'opsel_hi2', 1) is not None else 1
  neg, neg_hi = getattr(inst, 'neg', 0) or 0, getattr(inst, 'neg_hi', 0) or 0

  if is_pk_mov_b32:
    # v_pk_mov_b32: D[lo] = src0[opsel_bit0 ? hi : lo], D[hi] = src1[opsel_bit1 ? hi : lo]
    src_offs = [ctx.inst_field(type(inst).src0), ctx.inst_field(type(inst).src1)]
    def _pk_mov_sel(src_lo: UOp, src_off: UOp, sel_bit: int) -> UOp:
      is_vgpr = src_off >= _c(256)
      vgpr_lo = ctx.rvgpr_dyn(src_off - _c(256), lane) if lane is not None else _c(0)
      vgpr_hi = ctx.rvgpr_dyn(src_off - _c(256) + _c(1), lane) if lane is not None else _c(0)
      is_sgpr_pair = src_off < _c(128)
      sgpr_hi = ctx.rsgpr_dyn(src_off + _c(1), is_sgpr_pair)
      scalar_sel = is_sgpr_pair.where(sgpr_hi, src_lo) if sel_bit else src_lo
      return is_vgpr.where(vgpr_hi if sel_bit else vgpr_lo, scalar_sel)
    lo_val = _pk_mov_sel(src0, src_offs[0], opsel & 1)
    hi_val = _pk_mov_sel(src1, src_offs[1], opsel & 2)
    result = _u64(lo_val, hi_val)
    lo_out, hi_out = _split64(result)
    stores = [ctx.wvgpr_dyn(vdst_reg, lane, lo_out, exec_mask), ctx.wvgpr_dyn(vdst_reg + _c(1), lane, hi_out, exec_mask)]
    return UOp.sink(UOp.group(*stores).end(lane), *ctx.inc_pc())

  srcs: dict[str, UOp | int] = {}
  if is_pk_f32:
    # CDNA packed F32: read 32-bit sources, build 64-bit packed values using opsel.
    # For VGPRs: opsel selects between v[reg] (0) and v[reg+1] (1) for each half.
    # For SGPR pairs (off < 128): s[N] = lo float32, s[N+1] = hi float32.
    # For inline constants (128 <= off < 256): broadcast same value to both halves.
    src_offs = [ctx.inst_field(type(inst).src0), ctx.inst_field(type(inst).src1), ctx.inst_field(type(inst).src2)]
    def build_pk_f32(src_lo: UOp, src_off: UOp, opsel_lo: int, opsel_hi_bit: int, neg_lo: int, neg_hi_bit: int) -> UOp:
      is_vgpr = src_off >= _c(256)
      vgpr_lo = ctx.rvgpr_dyn(src_off - _c(256), lane) if lane is not None else _c(0)
      vgpr_hi = ctx.rvgpr_dyn(src_off - _c(256) + _c(1), lane) if lane is not None else _c(0)
      # For SGPR pairs, opsel selects between s[N] (0) and s[N+1] (1); inline constants always broadcast.
      is_sgpr_pair = src_off < _c(128)
      sgpr_hi = ctx.rsgpr_dyn(src_off + _c(1), is_sgpr_pair)
      scalar_lo_sel = src_lo if not opsel_lo else is_sgpr_pair.where(sgpr_hi, src_lo)
      scalar_hi_sel = src_lo if not opsel_hi_bit else is_sgpr_pair.where(sgpr_hi, src_lo)
      lo = is_vgpr.where(vgpr_hi if opsel_lo else vgpr_lo, scalar_lo_sel)
      hi = is_vgpr.where(vgpr_hi if opsel_hi_bit else vgpr_lo, scalar_hi_sel)
      if neg_lo: lo = lo ^ UOp.const(dtypes.uint32, 0x80000000)
      if neg_hi_bit: hi = hi ^ UOp.const(dtypes.uint32, 0x80000000)
      return _u64(lo, hi)
    srcs = {'S0': build_pk_f32(src0, src_offs[0], opsel & 1, opsel_hi & 1, neg & 1, neg_hi & 1),
            'S1': build_pk_f32(src1, src_offs[1], opsel & 2, opsel_hi & 2, neg & 2, neg_hi & 2),
            'S2': build_pk_f32(src2, src_offs[2], opsel & 4, 1 if opsel_hi2 else 0, neg & 4, neg_hi & 4)}
  elif 'FMA_MIX' in op_name or 'MAD_MIX' in op_name:
    combined_opsel_hi = (opsel_hi & 0x3) | ((opsel_hi2 & 0x1) << 2)
    # For FMA_MIX: neg_hi is ABS (not neg!), neg is actual negation
    def apply_abs(v, bit, opsel_hi_bit, opsel_bit):
      if not (neg_hi & bit): return v
      # Apply abs based on whether source is f32 or f16
      if not (combined_opsel_hi & opsel_hi_bit): return v & UOp.const(dtypes.uint32, 0x7FFFFFFF)  # f32 abs
      if opsel & opsel_bit: return v & UOp.const(dtypes.uint32, 0x7FFF0000)  # f16 hi abs (preserve lo)
      return v & UOp.const(dtypes.uint32, 0xFFFF7FFF)  # f16 lo abs (preserve hi)
    def apply_neg_mix(v, bit, opsel_hi_bit, opsel_bit):
      if not (neg & bit): return v
      if not (combined_opsel_hi & opsel_hi_bit): return v ^ UOp.const(dtypes.uint32, 0x80000000)  # f32 neg
      if opsel & opsel_bit: return v ^ UOp.const(dtypes.uint32, 0x80000000)  # f16 hi neg
      return v ^ UOp.const(dtypes.uint32, 0x00008000)  # f16 lo neg
    s0_mod = apply_neg_mix(apply_abs(src0, 1, 1, 1), 1, 1, 1)
    s1_mod = apply_neg_mix(apply_abs(src1, 2, 2, 2), 2, 2, 2)
    s2_mod = apply_neg_mix(apply_abs(src2, 4, 4, 4), 4, 4, 4)
    srcs = {'S@0': s0_mod, 'S@1': s1_mod, 'S@2': s2_mod,
            'OPSEL_HI': UOp.const(dtypes.uint32, combined_opsel_hi), 'OPSEL': UOp.const(dtypes.uint32, opsel)}
  else:
    def get_half_bits(val: UOp, use_hi: bool, apply_neg: bool = False) -> UOp:
      bits = ((val >> UOp.const(dtypes.uint32, 16)) if use_hi else val) & UOp.const(dtypes.uint32, 0xFFFF)
      if apply_neg: bits = bits.cast(dtypes.uint16).bitcast(dtypes.half).neg().bitcast(dtypes.uint16).cast(dtypes.uint32)
      return bits
    def build_remapped_src(src: UOp, opsel_lo_bit: int, opsel_hi_bit: int, neg_lo_bit: int, neg_hi_bit: int) -> UOp:
      lo = get_half_bits(src, bool(opsel_lo_bit), bool(neg_lo_bit))
      hi = get_half_bits(src, bool(opsel_hi_bit), bool(neg_hi_bit))
      return lo | (hi << UOp.const(dtypes.uint32, 16))
    # DOT IU instructions use NEG bits for signed/unsigned selection, not fp16 negation
    is_dot_iu = 'DOT' in op_name and 'IU' in op_name
    n0, n1, n2, nh0, nh1, nh2 = (0, 0, 0, 0, 0, 0) if is_dot_iu else (neg & 1, neg & 2, neg & 4, neg_hi & 1, neg_hi & 2, neg_hi & 4)
    srcs = {'S0': build_remapped_src(src0, opsel & 1, opsel_hi & 1, n0, nh0),
            'S1': build_remapped_src(src1, opsel & 2, opsel_hi & 2, n1, nh1),
            'S2': build_remapped_src(src2, opsel & 4, 1 if opsel_hi2 else 0, n2, nh2)}
    if is_dot_iu: srcs['NEG'] = UOp.const(dtypes.uint32, neg)
  return ctx.compile_vop_pcode(inst.op, srcs, lane, vdst_reg, exec_mask)

def _compile_vopd(inst: ir3.VOPD | ir4.VOPD, ctx: _Ctx) -> UOp:
  exec_mask = ctx.rexec()
  # Read operands dynamically - use type(inst) to get correct field descriptors
  inst_type = type(inst)
  vdstx_reg = ctx.inst_field(inst_type.vdstx)
  # vdsty has complex encoding: actual = (raw << 1) | ((vdstx & 1) ^ 1)
  vdsty_raw = ctx.inst_field(inst_type.vdsty)
  vdsty_reg = (vdsty_raw << _c(1)) | ((vdstx_reg & _c(1)) ^ _c(1))
  srcx0_off = ctx.inst_field(inst_type.srcx0)
  srcy0_off = ctx.inst_field(inst_type.srcy0)
  vsrcx1_reg = ctx.inst_field(inst_type.vsrcx1)
  vsrcy1_reg = ctx.inst_field(inst_type.vsrcy1)
  literal = ctx.inst_field(inst_type.literal) if hasattr(inst_type, 'literal') else None

  lane = ctx.range()
  srcy0, srcy1 = ctx.rsrc_dyn(srcy0_off, lane, literal=literal), ctx.rvgpr_dyn(vsrcy1_reg, lane)
  all_stores = []
  srcs:dict[str, UOp | int] = {}
  for op, src0_off, vsrc1_reg, vdst_reg, label in [(inst.opx, srcx0_off, vsrcx1_reg, vdstx_reg, 'X'),
                                                    (inst.opy, srcy0_off, vsrcy1_reg, vdsty_reg, 'Y')]:
    vop = VOPD_TO_VOP2.get(op)
    assert vop is not None, f"no VOP mapping for VOPD {label}: {op}"
    if label == 'Y': srcs = {'S0': srcy0, 'S1': srcy1, 'D0': ctx.rvgpr_dyn(vdst_reg, lane)}
    else: srcs = {'S0': ctx.rsrc_dyn(src0_off, lane, literal=literal), 'S1': ctx.rvgpr_dyn(vsrc1_reg, lane), 'D0': ctx.rvgpr_dyn(vdst_reg, lane)}
    # VOP2_FMAAK/FMAMK_(DTYPE)_E32
    if vop in (ir3.VOP2Op.V_FMAAK_F32_E32, ir3.VOP2Op.V_FMAMK_F32_E32, ir3.VOP2Op.V_FMAAK_F32_E32, ir3.VOP2Op.V_FMAMK_F32_E32):
      assert literal is not None
      srcs['SIMM32'] = literal
    if op in (ir3.VOPDOp.V_DUAL_CNDMASK_B32, ir4.VOPDOp.V_DUAL_CNDMASK_B32): srcs['VCC'] = ctx.rmask(_c(VCC_LO.offset))
    pcode = get_pcode(vop)
    srcs.update({'VCC': ctx.rmask(_c(VCC_LO.offset)), 'EXEC': exec_mask, 'SCC': ctx.rsgpr_dyn(_c(SCC.offset)), 'laneId': lane})
    for dest, val in parse_pcode(pcode, srcs)[1]:
      if dest.startswith('D0'): all_stores.append(ctx.wvgpr_dyn(vdst_reg, lane, _val_to_u32(val), exec_mask, after=srcy1))
  return UOp.sink(UOp.group(*all_stores).end(lane), *ctx.inc_pc())

def _compile_mem_op(inst: ir3.DS|ir3.FLAT|ir3.GLOBAL|ir3.SCRATCH|ir4.DS|ir4.VFLAT|ir4.VGLOBAL|ir4.VSCRATCH
                    |irc.DS|irc.FLAT|irc.GLOBAL|irc.SCRATCH, ctx: _Ctx) -> UOp:
  """Unified memory operation compiler for DS, FLAT, GLOBAL, SCRATCH."""
  exec_mask, op_name = ctx.rexec(), _op_name(inst)
  pcode = get_pcode(inst.op)
  # CDNA pcode uses CalcGlobalAddr/CalcDsAddr to compute address from raw components, but make_addr already handles this.
  # Strip the addr computation line and use pre-computed ADDR directly (rename 'addr' -> 'ADDR' in remaining pcode).
  if isinstance(inst, (irc.GLOBAL, irc.FLAT, irc.SCRATCH, irc.DS, ir4.VSCRATCH)) and 'Calc' in pcode and 'Addr' in pcode:
    pcode = re.sub(r'addr\s*=\s*Calc\w+Addr\([^)]*\)\s*;?\n?', '', pcode).replace('MEM[addr', 'MEM[ADDR')

  is_lds = isinstance(inst, (ir3.DS, ir4.DS, irc.DS))
  is_scratch = isinstance(inst, (ir3.SCRATCH, ir4.VSCRATCH, irc.SCRATCH))
  # CDNA acc bit: when set, VGPR operands (vdst/vdata) target ACCVGPR file instead of VGPR
  use_acc = bool(getattr(inst, 'acc', 0))
  mem = ctx.lds if is_lds else ctx.scratch if is_scratch else ctx.vmem
  addr_shift = UOp.const(dtypes.uint32 if is_lds else dtypes.uint64, 2)

  # Extract register info - all dynamic for deduplication
  if is_lds:
    addr_reg = ctx.inst_field(type(inst).addr)  # type: ignore[union-attr]
    vdata_reg = ctx.inst_field(type(inst).data0)  # type: ignore[union-attr]
    vdst_reg = ctx.inst_field(type(inst).vdst)
    offset0 = ctx.inst_field(type(inst).offset0)  # type: ignore[union-attr]
    offset1 = ctx.inst_field(type(inst).offset1)  # type: ignore[union-attr]
    offset = (offset1 << _c(8)) | offset0  # DS offset is 16-bit: (offset1 << 8) | offset0
    saddr_reg = None
  elif isinstance(inst, (ir4.VGLOBAL, ir4.VSCRATCH, ir4.VFLAT)):  # RDNA4: vaddr, vsrc, ioffset
    addr_reg = ctx.inst_field(type(inst).vaddr)
    vdata_reg = ctx.inst_field(type(inst).vsrc)
    vdst_reg = ctx.inst_field(type(inst).vdst)
    offset = ctx.inst_field_signed(type(inst).ioffset)
    offset0, offset1 = _c(0), _c(0)
    saddr_reg = ctx.inst_field(type(inst).saddr) if hasattr(type(inst), 'saddr') else None
  else:  # RDNA3: addr, data, offset
    addr_reg = ctx.inst_field(type(inst).addr)  # type: ignore[union-attr]
    vdata_reg = ctx.inst_field(type(inst).data)  # type: ignore[union-attr]
    vdst_reg = ctx.inst_field(type(inst).vdst)
    offset = ctx.inst_field_signed(type(inst).offset)  # type: ignore[union-attr]
    offset0, offset1 = _c(0), _c(0)
    saddr_reg = ctx.inst_field(type(inst).saddr) if hasattr(type(inst), 'saddr') else None  # type: ignore[union-attr]

  # Data width from canonical_op_bits (32/64/96/128), default to 32 for untyped ops
  data_bits_mem = inst.canonical_op_bits.get('data', 32)
  is_atomic, glc = 'ATOMIC' in op_name, getattr(inst, 'glc', 0)
  has_data1 = is_lds and hasattr(inst, 'data1') and inst.data1 is not None
  data1_reg = ctx.inst_field(type(inst).data1) if is_lds else _c(0)  # type: ignore[union-attr]

  # DS_PERMUTE/DS_BPERMUTE: cross-lane VGPR access via pcode
  if is_lds and 'PERMUTE' in op_name:
    pcode = get_pcode(inst.op)
    srcs = {'ADDR': addr_reg, 'DATA0': vdata_reg, 'VDST': vdst_reg, 'OFFSET': offset,
            'EXEC': exec_mask.cast(dtypes.uint64), '_vgpr': ctx.vgpr, '_wave_size': ctx.wave_size}
    _, assigns = parse_pcode(pcode, srcs)
    stores = [ctx.vgpr.index(val[0].cast(dtypes.int)).store(val[1].cast(dtypes.uint32)) for dest, val in assigns if dest.startswith('VGPR[')]
    return UOp.sink(*stores, *ctx.inc_pc())

  def make_addr(lane: UOp) -> UOp:
    if is_lds:
      addr = ctx.rvgpr_dyn(addr_reg, lane)
      # Some DS pcode (e.g. DS_STORE_B16) uses MEM[ADDR] without adding OFFSET explicitly.
      # In those cases, add the instruction offset to ADDR here.
      if 'OFFSET' not in pcode: addr = addr + offset
      return addr
    offset64 = offset.cast(dtypes.uint64)
    # Dynamic saddr check: saddr < 124 means valid SGPR, otherwise use VGPR pair for address
    use_saddr = (saddr_reg < _c(124)) if saddr_reg is not None else UOp.const(dtypes.bool, False)
    if is_scratch:
      scratch_stride = ctx.rsgpr_dyn(_c(SCRATCH_STRIDE_IDX)).cast(dtypes.uint64)
      base = lane.cast(dtypes.uint64) * scratch_stride
      # SVE (Scratch VGPR Enable): when SVE=1, VADDR is used as offset; when SVE=0, VADDR is ignored
      sve = getattr(inst, 'sve', 0)
      vaddr = ctx.rvgpr_dyn(addr_reg, lane).cast(dtypes.uint64)
      addr_offset = vaddr if sve == 1 else UOp.const(dtypes.uint64, 0)
      # Add saddr value only if use_saddr is true (saddr < 124)
      saddr_contrib = use_saddr.where(ctx.rsgpr_dyn(saddr_reg).cast(dtypes.uint64), UOp.const(dtypes.uint64, 0)) \
        if saddr_reg is not None else UOp.const(dtypes.uint64, 0)
      return base + addr_offset + saddr_contrib + offset64
    # FLAT/GLOBAL: choose between SGPR base (saddr) or VGPR pair (addr) based on saddr validity
    saddr_base = _u64(ctx.rsgpr_dyn(saddr_reg), ctx.rsgpr_dyn(saddr_reg + _c(1))) if saddr_reg is not None else UOp.const(dtypes.uint64, 0)
    vaddr_base = _u64(ctx.rvgpr_dyn(addr_reg, lane), ctx.rvgpr_dyn(addr_reg + _c(1), lane))
    # When saddr is valid: base = saddr pair, vaddr is 32-bit offset; otherwise: base = 0, vaddr is 64-bit address
    base_addr = use_saddr.where(saddr_base + ctx.rvgpr_dyn(addr_reg, lane).cast(dtypes.uint64), vaddr_base)
    return base_addr + offset64

  def wmem(addr: UOp, val: UOp, active: UOp, data_bits: int = 32) -> UOp:
    if data_bits < 32:
      # Sub-dword LDS write: read-modify-write within the uint32 slot
      word_addr = (addr >> addr_shift).cast(dtypes.int)
      idx = mem.index(word_addr, active)
      byte_pos = addr.cast(dtypes.uint32) & _c(3)
      byte_shift = byte_pos * _c(8)
      size_mask = _c(0xFF if data_bits == 8 else 0xFFFF)
      mask = size_mask << byte_shift
      new_word = (idx & (mask ^ _c(0xFFFFFFFF))) | ((val.cast(dtypes.uint32) & size_mask) << byte_shift)
      return idx.store(active.where(new_word, idx))
    idx = mem.index((addr >> addr_shift).cast(dtypes.int))
    return idx.store(active.where(val, idx.load()))

  def make_srcs(lane: UOp) -> dict:
    addr = make_addr(lane)
    if is_lds:
      if data_bits_mem == 128:
        data = {'DATA': ctx.rvgpr_dyn(vdata_reg, lane), 'DATA1': ctx.rvgpr_dyn(vdata_reg + _c(1), lane),
                'DATA2': ctx.rvgpr_dyn(vdata_reg + _c(2), lane), 'DATA3': ctx.rvgpr_dyn(vdata_reg + _c(3), lane)}
      elif data_bits_mem == 96:
        data = {'DATA': ctx.rvgpr_dyn(vdata_reg, lane), 'DATA1': ctx.rvgpr_dyn(vdata_reg + _c(1), lane),
                'DATA2': ctx.rvgpr_dyn(vdata_reg + _c(2), lane)}
      elif data_bits_mem <= 32:
        data = {'DATA': ctx.rvgpr_dyn(vdata_reg, lane), 'DATA2': ctx.rvgpr_dyn(data1_reg, lane) if has_data1 else UOp.const(dtypes.uint32, 0)}
      else:
        data = {'DATA': _u64(ctx.rvgpr_dyn(vdata_reg, lane), ctx.rvgpr_dyn(vdata_reg + _c(1), lane)),
                'DATA2': _u64(ctx.rvgpr_dyn(data1_reg, lane), ctx.rvgpr_dyn(data1_reg + _c(1), lane)) if has_data1 else UOp.const(dtypes.uint64, 0)}
      # RDNA3 uses ADDR/OFFSET, RDNA4 uses vgpr_a/offset (lowercase) + CalcDsAddr function
      return {'ADDR': addr, 'ADDR_BASE': addr, 'OFFSET': offset, 'OFFSET0': offset0, 'OFFSET1': offset1, '_lds': mem, 'laneId': lane,
              'vgpr_a': ctx.rvgpr_dyn(addr_reg, lane), 'offset': offset, **data}
    active = _lane_active(exec_mask, lane)
    # saddr < 124 means valid SGPR pair, otherwise use 0 (NULL means no saddr contribution)
    use_saddr = (saddr_reg < _c(124)) if saddr_reg is not None else UOp.const(dtypes.bool, False)
    saddr_raw = _u64(ctx.rsgpr_dyn(saddr_reg), ctx.rsgpr_dyn(saddr_reg + _c(1))) if saddr_reg is not None else UOp.const(dtypes.uint64, 0)
    saddr_base = use_saddr.where(saddr_raw, UOp.const(dtypes.uint64, 0))
    # Sign-extend offset to 64-bit for the final address calculation
    ioffset64 = offset.cast(dtypes.int64).cast(dtypes.uint64)
    # v_addr for CalcGlobalAddr: when saddr valid, use low 32 bits as offset; otherwise full 64-bit address. Include ioffset.
    vaddr_full = _u64(ctx.rvgpr_dyn(addr_reg, lane), ctx.rvgpr_dyn(addr_reg + _c(1), lane))
    vaddr_lo = ctx.rvgpr_dyn(addr_reg, lane).cast(dtypes.uint64)
    vaddr_base = use_saddr.where(vaddr_lo + ioffset64, vaddr_full + ioffset64)
    if is_atomic:
      atomic_data = _u64(ctx.rvgpr_dyn(vdata_reg, lane), ctx.rvgpr_dyn(vdata_reg + _c(1), lane)) \
        if data_bits_mem == 64 else ctx.rvgpr_dyn(vdata_reg, lane)
      return {'ADDR': addr, 'DATA': atomic_data, '_vmem': mem, '_active': active,
              'laneId': lane, 'v_addr': vaddr_base, 's_saddr': saddr_base}
    # acc bit: read/write ACCVGPR instead of VGPR for data operands
    _rvdata = (lambda r, l, *a: ctx.raccvgpr_dyn(r, l)) if use_acc else ctx.rvgpr_dyn
    vdata = _rvdata(vdata_reg, lane).cast(dtypes.uint64) if 'STORE' in op_name \
      else _rvdata(vdst_reg, lane) if 'D16' in op_name else UOp.const(dtypes.uint32, 0)
    if 'STORE' in op_name and data_bits_mem >= 64:
      vdata = vdata | (_rvdata(vdata_reg + _c(1), lane).cast(dtypes.uint64) << UOp.const(dtypes.uint64, 32))
    srcs = {'ADDR': addr, 'VDATA': vdata, '_vmem': mem, '_active': active,
            'laneId': lane, 'v_addr': vaddr_base, 's_saddr': saddr_base, 'SADDR': saddr_base, 'OFFSET': offset}
    for i in range(data_bits_mem // 32):
      srcs[f'VDATA{i}'] = _rvdata(vdata_reg + _c(i), lane) if 'STORE' in op_name else UOp.const(dtypes.uint32, 0)
    return srcs

  def make_stores(dest: str, val: UOp, lane: UOp, active: UOp, writes_return_data: bool) -> list[UOp]:
    # Parse bit width from dest format: MEM[...].b32 or RETURN_DATA[63:32].b64
    parts = dest.rsplit('.', 1)
    data_bits = int(parts[1][1:]) if len(parts) == 2 else 32
    if dest.startswith('MEM['):
      if is_lds or is_atomic:
        if data_bits < 32 and is_lds: return [wmem(val[0], val[1], active, data_bits)]
        return _write_val(data_bits, val[1], wmem, val[0], active, is_mem=True)
      if is_scratch: return _mem_store_bytes(mem, val[0], val[1], active, data_bits)
      return _mem_store(mem, val[0], val[1], active, 64, data_bits)
    if dest.startswith('RETURN_DATA') and writes_return_data:
      _wdata = (lambda r, v, l, e: ctx.waccvgpr_dyn(r, l, v, e)) if use_acc else (lambda r, v, l, e: ctx.wvgpr_dyn(r, l, v, e))
      if (m := re.match(r'RETURN_DATA\[(\d+)\s*:\s*(\d+)\]', dest)):
        bit_width, dword_idx = int(m.group(1)) - int(m.group(2)) + 1, int(m.group(2)) // 32
        return _write_val(bit_width, val, _wdata, vdst_reg + _c(dword_idx), lane, exec_mask)
      return _write_val(data_bits, val, _wdata, vdst_reg, lane, exec_mask)
    return []

  # DS-specific: check for 2ADDR pattern needing separate ranges
  if is_lds:
    dummy_lane = ctx.range()
    _, assigns = parse_pcode(pcode, make_srcs(dummy_lane))
    mem_assigns = [d for d, _ in assigns if d.startswith('MEM[')]
    mem_addrs = set(m.group(1) if (m := re.match(r'MEM\[([^\]]+)\]', d)) else d for d in mem_assigns)
    use_separate_ranges = (len(mem_addrs) > 1 or '2ADDR' in op_name) and 'STOREXCHG' not in op_name
    if use_separate_ranges:
      # Split assigns into MEM writes (stores) and RETURN_DATA writes (loads).
      # Stores to different addresses need separate lane ranges. Loads must share a single lane range so the
      # addr vgpr is read before any vdst write (hardware reads addr once, then writes all results).
      store_assigns = [(i, d) for i, (d, _) in enumerate(assigns) if d.startswith('MEM[')]
      load_assigns = [(i, d) for i, (d, _) in enumerate(assigns) if d.startswith('RETURN_DATA')]
      ended: list[UOp] = []
      for i, dest in store_assigns:
        lane = ctx.range()
        active = _lane_active(exec_mask, lane)
        _, lane_assigns = parse_pcode(pcode, make_srcs(lane))
        ended.extend(s.end(lane) for s in make_stores(dest, lane_assigns[i][1], lane, active, True))
      if load_assigns:
        lane = ctx.range()
        active = _lane_active(exec_mask, lane)
        _, lane_assigns = parse_pcode(pcode, make_srcs(lane))
        load_stores: list[UOp] = []
        for i, dest in load_assigns:
          load_stores.extend(make_stores(dest, lane_assigns[i][1], lane, active, True))
        if load_stores: ended.append(UOp.group(*load_stores).end(lane))
      return UOp.sink(*ended, *ctx.inc_pc())

  # Standard path: single lane range
  writes_return_data = '_RTN' in op_name or (is_lds and (op_name.startswith('DS_LOAD') or op_name.startswith('DS_READ'))) or bool(is_atomic and glc)
  lane = ctx.range()
  active = _lane_active(exec_mask, lane)
  pcode_vars, assigns = parse_pcode(pcode, make_srcs(lane))
  stores = [s for dest, val in assigns for s in make_stores(dest, val, lane, active, writes_return_data)]

  # FLAT/GLOBAL/SCRATCH: collect VDATA slices for loads
  if not is_lds and not is_atomic:
    _wdst = ctx.waccvgpr_dyn if use_acc else ctx.wvgpr_dyn
    for dword_idx, val in sorted(_collect_data_slices(assigns, 'VDATA', pcode_vars, op_name).items()):
      stores.append(_wdst(vdst_reg + _c(dword_idx), lane, val, exec_mask))

  return UOp.sink(UOp.group(*stores).end(lane), *ctx.inc_pc())

def _compile_mubuf(inst: irc.MUBUF, ctx: _Ctx) -> UOp:
  """CDNA MUBUF: linear buffer address = base + soffset + (stride * index) + vgpr_offset + inst_offset"""
  exec_mask, op_name = ctx.rexec(), _op_name(inst)
  use_acc, is_store, is_lds = bool(getattr(inst, 'acc', 0)), 'STORE' in op_name, bool(getattr(inst, 'lds', 0))
  n_dwords = 4 if 'X4' in op_name else 2 if 'X2' in op_name else 1

  # instruction fields
  vdata, vaddr = ctx.inst_field(type(inst).vdata), ctx.inst_field(type(inst).vaddr)
  srsrc, soffset = ctx.inst_field(type(inst).srsrc) * _c(4), ctx.inst_field(type(inst).soffset)
  offset, offen, idxen = ctx.inst_field(type(inst).offset), ctx.inst_field(type(inst).offen), ctx.inst_field(type(inst).idxen)

  # V# descriptor: base[0:1], num_records[2], stride=word3[13:0]
  base = _u64(ctx.rsgpr_dyn(srsrc), ctx.rsgpr_dyn(srsrc + _c(1))) & UOp.const(dtypes.uint64, 0xFFFFFFFFFFFF)
  num_records = ctx.rsgpr_dyn(srsrc + _c(2))
  stride = (ctx.rsgpr_dyn(srsrc + _c(3)) & _c(0x3FFF)).cast(dtypes.uint64)

  lane = ctx.range()
  active = _lane_active(exec_mask, lane)

  # soffset: sgpr if < 128, else inline constant
  soff = (soffset < _c(128)).where(ctx.rsgpr_dyn(soffset), soffset - _c(128)).cast(dtypes.uint64)
  # vaddr: index (if idxen) in vaddr, offset (if offen) in vaddr or vaddr+1
  index = idxen.ne(_c(0)).where(ctx.rvgpr_dyn(vaddr, lane), _c(0)).cast(dtypes.uint64)
  voff = offen.ne(_c(0)).where(ctx.rvgpr_dyn(idxen.ne(_c(0)).where(vaddr + _c(1), vaddr), lane), _c(0)).cast(dtypes.uint64)

  # buffer_offset for bounds check, final address
  buffer_offset = (stride * index + voff + offset.cast(dtypes.uint64)).cast(dtypes.uint32)
  in_bounds = active & buffer_offset.__lt__(num_records)
  addr = base + soff + buffer_offset.cast(dtypes.uint64)
  addr = in_bounds.where(addr, UOp.const(dtypes.uint64, 0))  # safe address when OOB
  mem = ctx.vmem

  stores: list[UOp] = []
  if is_lds and not is_store:
    # LDS load: buffer -> LDS (bypass VGPRs), LDS addr = M0[17:0] + lane * elem_size
    lds_base = ctx.rsgpr_dyn(_c(124)) & _c(0x3FFFF)
    lds_addr = lds_base + lane.cast(dtypes.uint32) * _c(n_dwords * 4)
    for i in range(n_dwords):
      word_addr = (addr + UOp.const(dtypes.uint64, i * 4)) >> UOp.const(dtypes.uint64, 2)
      val = in_bounds.where(mem.index(word_addr.cast(dtypes.int64), ptr=True).load(), _c(0))
      lds_idx = ((lds_addr + _c(i * 4)) >> _c(2)).cast(dtypes.int)
      stores.append(ctx.lds.index(lds_idx, active).store(active.where(val, ctx.lds.index(lds_idx, active))))
  elif is_store:
    for i in range(n_dwords):
      word_addr = (addr + UOp.const(dtypes.uint64, i * 4)) >> UOp.const(dtypes.uint64, 2)
      idx = mem.index(word_addr.cast(dtypes.int64), in_bounds)
      val = (ctx.raccvgpr_dyn if use_acc else ctx.rvgpr_dyn)(vdata + _c(i), lane)
      stores.append(idx.store(in_bounds.where(_to_u32(val), idx)))
  else:
    for i in range(n_dwords):
      word_addr = (addr + UOp.const(dtypes.uint64, i * 4)) >> UOp.const(dtypes.uint64, 2)
      val = in_bounds.where(mem.index(word_addr.cast(dtypes.int64), in_bounds, ptr=True).load(), _c(0))
      stores.append((ctx.waccvgpr_dyn if use_acc else ctx.wvgpr_dyn)(vdata + _c(i), lane, val, exec_mask))
  return UOp.sink(UOp.group(*stores).end(lane), *ctx.inc_pc())

# Dispatch table: instruction type -> handler function
_INST_HANDLERS: dict[type, Callable[..., UOp]] = {
  ir3.SOPP: _compile_sopp, ir3.SMEM: _compile_smem, ir3.SOP1: _compile_sop, ir3.SOP2: _compile_sop, ir3.SOPC: _compile_sop, ir3.SOPK: _compile_sop,
  ir3.VOP1: _compile_vop12, ir3.VOP1_SDST: _compile_vop12, ir3.VOP2: _compile_vop12, ir3.VOPC: _compile_vopc, ir3.VOP3: _compile_vop3,
  ir3.VOP3_SDST: _compile_vop3, ir3.VOP3SD: _compile_vop3sd, ir3.VOP3P: _compile_vop3p, ir3.VOPD: _compile_vopd,
  ir3.DS: _compile_mem_op, ir3.FLAT: _compile_mem_op, ir3.GLOBAL: _compile_mem_op, ir3.SCRATCH: _compile_mem_op,
  # RDNA4 instruction classes
  ir4.SOPP: _compile_sopp, ir4.SMEM: _compile_smem, ir4.SOP1: _compile_sop, ir4.SOP2: _compile_sop, ir4.SOPC: _compile_sop, ir4.SOPK: _compile_sop,
  ir4.VOP1: _compile_vop12, ir4.VOP1_SDST: _compile_vop12, ir4.VOP2: _compile_vop12, ir4.VOPC: _compile_vopc, ir4.VOP3: _compile_vop3,
  ir4.VOP3_SDST: _compile_vop3, ir4.VOP3SD: _compile_vop3sd, ir4.VOP3P: _compile_vop3p, ir4.VOPD: _compile_vopd,
  ir4.DS: _compile_mem_op, ir4.VFLAT: _compile_mem_op, ir4.VGLOBAL: _compile_mem_op, ir4.VSCRATCH: _compile_mem_op,
  # CDNA instruction classes
  irc.SOPP: _compile_sopp, irc.SMEM: _compile_smem, irc.SOP1: _compile_sop, irc.SOP2: _compile_sop, irc.SOPC: _compile_sop, irc.SOPK: _compile_sop,
  irc.VOP1: _compile_vop12, irc.VOP2: _compile_vop12, irc.VOPC: _compile_vopc, irc.VOP3: _compile_vop3,
  irc.VOP3_SDST: _compile_vop3, irc.VOP3SD: _compile_vop3sd, irc.VOP3P: _compile_vop3p,
  irc.VOP1_SDWA: _compile_sdwa, irc.VOP2_SDWA: _compile_sdwa, irc.VOP2_SDWA_SDST: _compile_sdwa, irc.VOPC_SDWA_SDST: _compile_sdwa,
  irc.DS: _compile_mem_op, irc.FLAT: _compile_mem_op, irc.GLOBAL: _compile_mem_op, irc.SCRATCH: _compile_mem_op,
  irc.MUBUF: _compile_mubuf,
}

# ═══════════════════════════════════════════════════════════════════════════════
# PROGRAM DECODE AND COMPILATION
# ═══════════════════════════════════════════════════════════════════════════════

_canonical_runner_cache: list[tuple[type, int, int, int, object]] = []  # [(inst_type, base, mask, size, runner), ...]

@functools.cache
def _get_runner(inst_bytes: bytes, arch: str = "rdna3"):
  """Build and compile instruction to CompiledRunner. Cached by instruction bytes, with canonical dedup."""
  inst = decode_inst(inst_bytes, arch)
  inst_size = inst.size()
  inst_int = int.from_bytes(inst_bytes[:inst_size], 'little')

  # Check if instruction matches any cached canonical pattern (must also match instruction type to avoid variant conflicts)
  for inst_type, base, mask, size, runner in _canonical_runner_cache:
    if type(inst) is inst_type and inst_size == size and (inst_int & mask) == base: return runner

  # Look up handler by type, falling back to base classes for _LIT variants
  handler = _INST_HANDLERS.get(type(inst))
  if handler is None:
    for cls in type(inst).__mro__:
      if cls in _INST_HANDLERS:
        handler = _INST_HANDLERS[cls]
        break
  if handler is None: raise RuntimeError(f"[emu] unimplemented instruction type: {type(inst).__name__} {_op_name(inst)}")

  ctx = _Ctx(inst_size, _wave_size(arch))
  sink = handler(inst, ctx)
  base, mask, size = ctx.canonical_mask(inst_bytes)
  canonical_name = f"{_op_name(inst).lower()}_{base.to_bytes(size, 'little').hex()}"
  sink = sink.replace(arg=KernelInfo(name=canonical_name)).rtag(1)

  # NOTE: renderer output is not reproducible because of _MXCSRContext. PROFILE=0 prevents emulator instruction runners from polluting profiling.
  with Context(NOOPT=1, CHECK_OOB=0, TUPLE_ORDER=0, EMULATED_DTYPES="", CAPTURE_PROCESS_REPLAY=0, PROFILE=0):
    runner = get_runner('CPU', sink)
  _canonical_runner_cache.append((type(inst), base, mask, size, runner))
  return runner

_BARRIER_OPS = {ir3.SOPPOp.S_BARRIER, irc.SOPPOp.S_BARRIER}
if hasattr(ir4.SOPPOp, 'S_BARRIER_WAIT'): _BARRIER_OPS.add(ir4.SOPPOp.S_BARRIER_WAIT)
_BARRIER_SOP1_OPS: set = set()
if hasattr(ir4.SOP1Op, 'S_BARRIER_SIGNAL'): _BARRIER_SOP1_OPS.add(ir4.SOP1Op.S_BARRIER_SIGNAL)
_BRANCH_OPS: set[int] = {op.value for op in (ir3.SOPPOp.S_BRANCH, ir3.SOPPOp.S_CBRANCH_SCC0, ir3.SOPPOp.S_CBRANCH_SCC1,
  ir3.SOPPOp.S_CBRANCH_VCCZ, ir3.SOPPOp.S_CBRANCH_VCCNZ, ir3.SOPPOp.S_CBRANCH_EXECZ, ir3.SOPPOp.S_CBRANCH_EXECNZ)}

def _decode_at(pc: int, arch: str):
  """Decode and compile instruction at absolute address pc. Returns (runner, decoded_inst)."""
  inst_bytes = bytes((ctypes.c_char * 16).from_address(pc).raw)
  inst = decode_inst(inst_bytes, arch)
  try: return _get_runner(bytes(inst_bytes[:inst.size() + 4]), arch), inst
  except Exception as e:
    try: inst_str = repr(inst)
    except Exception: inst_str = f"<{type(inst).__name__}>"
    raise RuntimeError(f"[emu] Failed to compile {inst_str}: {type(e).__name__}: {e}") from e

# ═══════════════════════════════════════════════════════════════════════════════
# WAVE STATE
# ═══════════════════════════════════════════════════════════════════════════════

# Inline float constants (as bit patterns) for GPU instructions
F32_INLINE = {240: 0x3f000000, 241: 0xbf000000, 242: 0x3f800000, 243: 0xbf800000,  # 0.5, -0.5, 1.0, -1.0
              244: 0x40000000, 245: 0xc0000000, 246: 0x40800000, 247: 0xc0800000, 248: 0x3e22f983}  # 2.0, -2.0, 4.0, -4.0, 1/(2*pi)

class WaveState:
  __slots__ = ('vgpr_buf', 'sgpr_buf', 'accvgpr_buf', '_vgpr_mv', '_sgpr_mv', 'n_lanes', 'wave_size')

  def __init__(self, n_lanes: int, wave_size: int = 32):
    self.n_lanes, self.wave_size = n_lanes, wave_size
    vgpr_size = 256 * wave_size
    self.vgpr_buf = Buffer('CPU', vgpr_size, dtypes.uint32).ensure_allocated()
    self.sgpr_buf = Buffer('CPU', SGPR_COUNT, dtypes.uint32).ensure_allocated()
    # CDNA (wave64) has separate ACCVGPR file; RDNA shares with VGPR
    if wave_size == 64:
      self.accvgpr_buf = Buffer('CPU', vgpr_size, dtypes.uint32).ensure_allocated()
      ctypes.memset(self.accvgpr_buf._buf.va_addr, 0, vgpr_size * 4)
    else:
      self.accvgpr_buf = self.vgpr_buf
    self._vgpr_mv = self.vgpr_buf.as_memoryview(force_zero_copy=True).cast('I')
    self._sgpr_mv = self.sgpr_buf.as_memoryview(force_zero_copy=True).cast('I')
    # Zero memory using ctypes memset (much faster than Python loops)
    ctypes.memset(self.vgpr_buf._buf.va_addr, 0, vgpr_size * 4)
    ctypes.memset(self.sgpr_buf._buf.va_addr, 0, SGPR_COUNT * 4)
    # Pre-populate inline constants at indices 128-255
    for i in range(65): self._write_sgpr(128 + i, i)  # 128-192: integers 0-64
    for i in range(16): self._write_sgpr(193 + i, (-(i + 1)) & MASK32)  # 193-208: -1 to -16
    for off, val in F32_INLINE.items(): self._write_sgpr(off, val)  # 240-248: float constants
    # EXEC mask: for 64-lane waves, set both EXEC_LO and EXEC_HI
    if wave_size == 64:
      self._write_sgpr(EXEC_LO.offset, (1 << min(n_lanes, 32)) - 1)
      self._write_sgpr(EXEC_LO.offset + 1, (1 << max(n_lanes - 32, 0)) - 1 if n_lanes > 32 else 0)
    else:
      self._write_sgpr(EXEC_LO.offset, (1 << n_lanes) - 1)
    self._write_sgpr(PC_LO_IDX, 0)
    self._write_sgpr(PC_HI_IDX, 0)

  def _write_sgpr(self, idx: int, val: int): self._sgpr_mv[idx] = val & MASK32
  def _read_sgpr(self, idx: int) -> int: return self._sgpr_mv[idx]
  def _write_vgpr(self, reg: int, lane: int, val: int): self._vgpr_mv[reg * self.wave_size + lane] = val & MASK32
  def _read_vgpr(self, reg: int, lane: int) -> int: return self._vgpr_mv[reg * self.wave_size + lane]

  @property
  def pc(self) -> int: return self._read_sgpr(PC_LO_IDX) | (self._read_sgpr(PC_HI_IDX) << 32)
  @pc.setter
  def pc(self, val: int):
    self._write_sgpr(PC_LO_IDX, val & MASK32)
    self._write_sgpr(PC_HI_IDX, (val >> 32) & MASK32)

# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def _init_wave(lib: int, wave_start: int, total_threads: int, lx: int, ly: int, lz: int, args_ptr: int, rsrc2: int,
               scratch_size: int, arch: str, gidx: int, gidy: int, gidz: int, user_data: list[int]|None,
               wave_size: int = 32) -> WaveState:
  """Initialize a single wavefront and return WaveState."""
  n_lanes = min(wave_size, total_threads - wave_start)
  st = WaveState(n_lanes, wave_size)
  st.pc = lib
  if user_data:
    for i, val in enumerate(user_data): st._write_sgpr(i, val)
  else:
    st._write_sgpr(0, args_ptr & MASK32)
    st._write_sgpr(1, (args_ptr >> 32) & MASK32)
  if arch == "rdna4":
    # workgroup IDs only exist in ttmp registers, not normal SGPRs
    st._write_sgpr(ttmp[7].offset, (gidy & 0xFFFF) | ((gidz & 0xFFFF) << 16))
    st._write_sgpr(ttmp[9].offset, gidx)
  else:
    sgpr_idx = (rsrc2 & hsa.AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT) >> hsa.AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT_SHIFT
    for enabled, gid in [(hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X, gidx),
                         (hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y, gidy),
                         (hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z, gidz)]:
      if rsrc2 & enabled:
        st._write_sgpr(sgpr_idx, gid)
        sgpr_idx += 1
  for lane in range(n_lanes):
    tid = wave_start + lane
    st._write_vgpr(0, lane, ((tid // (lx * ly)) << 20) | (((tid // lx) % ly) << 10) | (tid % lx))
  st._write_sgpr(SCRATCH_STRIDE_IDX, scratch_size)
  # Store HW register values at SGPR[SGPR_COUNT-16 .. SGPR_COUNT-1] for s_getreg_b32 emulation.
  # HW_ID (hwRegId=4): WAVE_ID[3:0], SIMD_ID[5:4], PIPE_ID[7:6], CU_ID[11:8], ...
  wave_idx = wave_start // wave_size  # wave index within this workgroup (0, 1, 2, 3 for 256 threads / 64 wave_size)
  hw_id = (wave_idx & 0xF) | ((wave_idx & 0x3) << 4)  # WAVE_ID = wave_idx, SIMD_ID = wave_idx % 4
  st._write_sgpr(SGPR_COUNT - 16 + 4, hw_id)  # HW_REGISTERS[4] = HW_ID
  return st

def run_asm(lib: int, lib_sz: int, gx: int, gy: int, gz: int, lx: int, ly: int, lz: int, args_ptr: int, rsrc2: int = 0x19c,
            scratch_size: int = 0, arch: str = "rdna3", user_data: list[int]|None = None) -> int:
  """Execute AMD assembly program. scratch_size is private_segment_fixed_size from kernel descriptor (per-lane)."""
  from tinygrad.renderer.amd.dsl import Inst
  program: dict[int, tuple[Callable, list[int], bool, Inst]] = {}  # pc -> (fxn, globals, is_barrier, inst)
  lds_size = ((rsrc2 & hsa.AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE) >> hsa.AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE_SHIFT) * 512
  total_threads = lx * ly * lz
  wave_size = _wave_size(arch)

  # Use Buffer objects with external_ptr=0 for vmem
  vmem_buf = Buffer('CPU', 1 << 40, dtypes.uint32, options=BufferSpec(external_ptr=0)).ensure_allocated()
  lds_buf = Buffer('CPU', max(lds_size // 4, 1), dtypes.uint32).ensure_allocated()
  scratch_buf = Buffer('CPU', scratch_size * wave_size, dtypes.uint8).ensure_allocated() if scratch_size else None

  # Initialize SQTT encoder — emits packets inline as instructions execute (only when profiling)
  if PROFILE:
    sqtt_emit, sqtt_finish, sqtt_finalize = _init_sqtt_encoder(lib)

  def _ensure_compiled(pc: int) -> tuple[Callable, list[int], bool, Inst]:
    if pc not in program:
      prev_len = len(_canonical_runner_cache)
      runner, inst = _decode_at(pc, arch)
      is_barrier = (isinstance(inst, (ir3.SOPP, ir4.SOPP, irc.SOPP)) and inst.op in _BARRIER_OPS) or \
                   (isinstance(inst, (ir4.SOP1,)) and inst.op in _BARRIER_SOP1_OPS)
      program[pc] = (runner._prg.fxn, runner.p.globals, is_barrier, inst)
      if DEBUG >= 3:
        msg = f"[emu] PC={pc - lib}: {inst!r}"
        print(colored(msg, 'green') if len(_canonical_runner_cache) > prev_len else msg)
    return program[pc]

  # Set DAZ+FTZ during emulator execution, restore afterward to avoid breaking hypothesis tests
  # Only trace the first workgroup (like real HW traces one CU/SIMD), subsequent workgroups run but don't add to trace
  tracing = bool(PROFILE)

  with _MXCSRContext():
    for gidz in range(gz):
      for gidy in range(gy):
        for gidx in range(gx):
          # Initialize all wavefronts for this workgroup
          waves: list[tuple[WaveState, list]] = []
          for wave_start in range(0, total_threads, wave_size):
            st = _init_wave(lib, wave_start, total_threads, lx, ly, lz, args_ptr, rsrc2, scratch_size, arch, gidx, gidy, gidz, user_data,
                            wave_size)
            c_bufs = [ctypes.c_uint64(st.sgpr_buf._buf.va_addr), ctypes.c_uint64(st.vgpr_buf._buf.va_addr),
                      ctypes.c_uint64(vmem_buf._buf.va_addr), ctypes.c_uint64(lds_buf._buf.va_addr),
                      ctypes.c_uint64(scratch_buf._buf.va_addr if scratch_buf else 0),
                      ctypes.c_uint64(st.accvgpr_buf._buf.va_addr)]
            waves.append((st, c_bufs))

          # Execute wavefronts with barrier synchronization
          # Each wave runs until it hits s_barrier or s_endpgm. When all waves have stopped, release barrier waves.
          done = [False] * len(waves)
          for total_inst in range(10_000_000):
            if all(done): break
            for wi, (st, c_bufs) in enumerate(waves):
              if done[wi]: continue
              # Run this wave until barrier or endpgm
              for _ in range(1_000_000):
                pc = st.pc
                if pc == ENDPGM_PC:
                  done[wi] = True
                  if tracing: sqtt_finish(wi)
                  break
                fxn, globals_list, is_barrier, inst = _ensure_compiled(pc)
                if DEBUG >= 5: print(f"  exec gid=({gidx},{gidy},{gidz}) w={wi} PC={pc - lib}: {inst!r}", flush=True)
                fxn(*[c_bufs[g] for g in globals_list])
                if tracing:
                  inst_op = inst.op.value if hasattr(inst, 'op') else 0
                  sqtt_emit(wi, inst, (st.pc != ENDPGM_PC and st.pc != pc + inst.size()) if inst_op in _BRANCH_OPS else None)
                if is_barrier: break  # s_barrier hit: PC already advanced past it, pause this wave
              else: raise RuntimeError("exceeded 1M instructions in single wave, likely infinite loop")
            # All waves have either hit barrier or endpgm — release barrier waves for next round
          else: raise RuntimeError("exceeded 10M total scheduling rounds")
          tracing = False  # only trace the first workgroup

          # Reset LDS for next workgroup
          if lds_size > 0: ctypes.memset(lds_buf._buf.va_addr, 0, max(lds_size, 4))

  if PROFILE:
    blob, cycles = sqtt_finalize()
    sqtt_traces.append(blob)
    sqtt_cycle_counts.append(cycles)
  return 0
