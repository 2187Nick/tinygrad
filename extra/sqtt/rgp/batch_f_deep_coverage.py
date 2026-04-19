#!/usr/bin/env python3
"""Batch F — Deep coverage probes for 100% bounty push.

~110 kernels targeting the remaining strict-mode gaps:
  F.1 (30) — Fine-grained chain-length coverage for v_add/v_mul/v_fmac.
  F.2 (25) — Mixed RAW + independent VALU chains (RAW-breaking patterns).
  F.3 (15) — VMEM store deep interleaving / width / spacing.
  F.4 (15) — Transcendental chains of varying depth.
  F.5 (15) — Cross-regfile RAW (SGPR-chain, lane-op chains).
  F.6 (15) — Edge ops: VGPR bank-conflict chains, v_pk ops, 3-src VALU chains.

Each kernel runs through the standard microbench prologue/epilogue.
Naming: mb_f{N}_{description}.
"""
from __future__ import annotations
import struct

from tinygrad.renderer.amd.dsl import s, v, NULL, VCC_LO  # noqa: F401
from tinygrad.runtime.autogen.amd.rdna3.ins import (
  v_add_f32_e32, v_mul_f32_e32, v_fmac_f32_e32, v_mov_b32_e32,
  v_sub_f32_e32, v_max_f32_e32, v_min_f32_e32,
  v_exp_f32_e32, v_log_f32_e32, v_sqrt_f32_e32, v_rcp_f32_e32, v_rsq_f32_e32,
  v_cmp_gt_f32_e32, v_cmp_gt_f32_e64, v_cndmask_b32_e32, v_cndmask_b32_e64,
  v_mul_lo_u32, v_mul_hi_u32, v_mad_u32_u24,
  v_bfi_b32, v_alignbyte_b32, v_lshrrev_b32_e32, v_lshlrev_b32_e32, v_and_b32_e32,
  v_readlane_b32, v_writelane_b32, v_readfirstlane_b32_e32,
  v_cvt_f16_f32_e32, v_cvt_f32_f16_e32, v_pk_add_f16, v_pk_mul_f16, v_pk_fma_f16,
  v_fma_f32, v_fmac_f32_e32 as _fmac32,
  s_mov_b32, s_nop, s_waitcnt, s_waitcnt_vmcnt, s_waitcnt_lgkmcnt, s_waitcnt_depctr,
  s_add_u32, s_lshl_b32,
  global_load_b32, global_load_b64, global_store_b32, global_store_b64, global_store_b128,
  VOPD, VOPDOp,
)

from extra.sqtt.rgp.microbench import microbench, sweep, Kernel  # noqa: F401


def _f2i(f: float) -> int:
  return struct.unpack('I', struct.pack('f', f))[0]


# ═════════════════════════════════════════════════════════════════════════════
# F.1 — Fine-grained chain-length coverage (30 kernels)
# ═════════════════════════════════════════════════════════════════════════════

# F.1.1 — v_add_f32 at every chain length 2..15 (existing has 1,2,4,8,16; E.1 has 3,5,6,7,10,12,24)
sweep("mb_f1_valu_add", [9, 11, 13, 14, 15, 18, 20, 22, 28, 32],
  lambda n: (lambda k: [k.emit(v_add_f32_e32(v[1], 1.0, v[1])) for _ in range(n)]),
  category="F.1-chain-granular")

# F.1.2 — v_mul_f32 chain at various depths
sweep("mb_f1_valu_mul", [2, 3, 6, 8, 12, 16],
  lambda n: (lambda k: [k.emit(v_mul_f32_e32(v[1], v[2], v[1])) for _ in range(n)]),
  category="F.1-chain-granular")

# F.1.3 — v_fmac_f32 chain
sweep("mb_f1_valu_fmac", [2, 3, 5, 6, 10, 12, 16],
  lambda n: (lambda k: [k.emit(v_fmac_f32_e32(v[1], v[2], v[3])) for _ in range(n)]),
  category="F.1-chain-granular")

# F.1.4 — 3-src VOP3 chain (v_fma_f32) — unusual pipeline path
sweep("mb_f1_valu_fma", [2, 4, 8, 16],
  lambda n: (lambda k: [k.emit(v_fma_f32(v[1], v[2], v[3], v[1])) for _ in range(n)]),
  category="F.1-chain-granular")

# F.1.5 — v_sub_f32 chain (to test RAW behavior on different ALU opcode)
sweep("mb_f1_valu_sub", [4, 8, 16],
  lambda n: (lambda k: [k.emit(v_sub_f32_e32(v[1], 1.0, v[1])) for _ in range(n)]),
  category="F.1-chain-granular")


# ═════════════════════════════════════════════════════════════════════════════
# F.2 — Mixed RAW + independent VALU (25 kernels)
# ═════════════════════════════════════════════════════════════════════════════

@microbench(name="mb_f2_raw_then_indep_n4", category="F.2-mixed-raw")
def _f2_raw_then_indep_n4(k: Kernel) -> None:
  """4 RAW adds, then 4 independent adds (different VGPRs)."""
  for _ in range(4): k.emit(v_add_f32_e32(v[1], 1.0, v[1]))
  for i in range(4): k.emit(v_add_f32_e32(v[20 + i], v[2], v[3]))

@microbench(name="mb_f2_indep_then_raw_n4", category="F.2-mixed-raw")
def _f2_indep_then_raw_n4(k: Kernel) -> None:
  """4 independent adds, then 4 RAW adds."""
  for i in range(4): k.emit(v_add_f32_e32(v[20 + i], v[2], v[3]))
  for _ in range(4): k.emit(v_add_f32_e32(v[1], 1.0, v[1]))

@microbench(name="mb_f2_raw_indep_interleave_n8", category="F.2-mixed-raw")
def _f2_raw_indep_interleave(k: Kernel) -> None:
  """Alternate RAW and independent: RAW, indep, RAW, indep, ... 8 pairs."""
  for i in range(8):
    k.emit(v_add_f32_e32(v[1], 1.0, v[1]))
    k.emit(v_add_f32_e32(v[20 + i], v[2], v[3]))

@microbench(name="mb_f2_raw_broken_by_mov_n8", category="F.2-mixed-raw")
def _f2_raw_broken_by_mov(k: Kernel) -> None:
  """RAW chain with mov breaking the chain in the middle."""
  for _ in range(4): k.emit(v_add_f32_e32(v[1], 1.0, v[1]))
  k.emit(v_mov_b32_e32(v[20], 1.0))
  for _ in range(4): k.emit(v_add_f32_e32(v[1], 1.0, v[1]))

@microbench(name="mb_f2_raw_broken_by_nop_n8", category="F.2-mixed-raw")
def _f2_raw_broken_by_nop(k: Kernel) -> None:
  """RAW chain with s_nop(0) in the middle — tests whether chain resets."""
  for _ in range(4): k.emit(v_add_f32_e32(v[1], 1.0, v[1]))
  k.emit(s_nop(0))
  for _ in range(4): k.emit(v_add_f32_e32(v[1], 1.0, v[1]))

@microbench(name="mb_f2_raw_broken_by_nop_big_n8", category="F.2-mixed-raw")
def _f2_raw_broken_by_nop_big(k: Kernel) -> None:
  """RAW chain with s_nop(7) — bigger idle gap."""
  for _ in range(4): k.emit(v_add_f32_e32(v[1], 1.0, v[1]))
  k.emit(s_nop(7))
  for _ in range(4): k.emit(v_add_f32_e32(v[1], 1.0, v[1]))

@microbench(name="mb_f2_two_parallel_raw_chains_n4", category="F.2-mixed-raw")
def _f2_two_parallel_raw_chains(k: Kernel) -> None:
  """Two RAW chains on different VGPRs, interleaved (one add from each)."""
  for _ in range(4):
    k.emit(v_add_f32_e32(v[1], 1.0, v[1]))
    k.emit(v_add_f32_e32(v[2], 1.0, v[2]))

@microbench(name="mb_f2_three_parallel_raw_chains_n4", category="F.2-mixed-raw")
def _f2_three_parallel_raw_chains(k: Kernel) -> None:
  """Three parallel RAW chains, round-robin."""
  for _ in range(4):
    k.emit(v_add_f32_e32(v[1], 1.0, v[1]))
    k.emit(v_add_f32_e32(v[2], 1.0, v[2]))
    k.emit(v_add_f32_e32(v[3], 1.0, v[3]))

@microbench(name="mb_f2_raw_with_war_n4", category="F.2-mixed-raw")
def _f2_raw_with_war(k: Kernel) -> None:
  """RAW chain with a WAR dependency mid-stream (reader, then writer)."""
  k.emit(v_add_f32_e32(v[10], v[1], v[1]))  # reads v[1]
  for _ in range(3): k.emit(v_add_f32_e32(v[1], 1.0, v[1]))  # writes v[1]
  k.emit(v_mov_b32_e32(v[11], v[1]))  # reads v[1] again

@microbench(name="mb_f2_raw_with_mov_indep", category="F.2-mixed-raw")
def _f2_raw_with_mov_indep(k: Kernel) -> None:
  """RAW chain interspersed with independent mov's to different bank."""
  for i in range(6):
    k.emit(v_add_f32_e32(v[1], 1.0, v[1]))
    if i < 5: k.emit(v_mov_b32_e32(v[20 + i], float(i + 1)))

sweep("mb_f2_mul_add_raw_chain", [2, 4, 8],
  lambda n: (lambda k: [item for _ in range(n) for item in [
    k.emit(v_mul_f32_e32(v[1], v[2], v[1])),
    k.emit(v_add_f32_e32(v[1], 1.0, v[1]))]]),
  category="F.2-mixed-raw")

@microbench(name="mb_f2_raw_then_vopd", category="F.2-mixed-raw")
def _f2_raw_then_vopd(k: Kernel) -> None:
  """Long RAW chain then VOPD — does VOPD see the stall?"""
  for _ in range(8): k.emit(v_add_f32_e32(v[1], 1.0, v[1]))
  for i in range(8): k.emit(v_mov_b32_e32(v[4 + i], float(i + 1)))
  k.emit(VOPD(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_ADD_F32,
              v[20], v[21], v[4], v[5], v[6], v[7]))

@microbench(name="mb_f2_vopd_then_raw", category="F.2-mixed-raw")
def _f2_vopd_then_raw(k: Kernel) -> None:
  """VOPD then RAW chain — does VOPD provide a fresh queue?"""
  for i in range(8): k.emit(v_mov_b32_e32(v[4 + i], float(i + 1)))
  k.emit(VOPD(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_ADD_F32,
              v[20], v[21], v[4], v[5], v[6], v[7]))
  for _ in range(8): k.emit(v_add_f32_e32(v[1], 1.0, v[1]))

@microbench(name="mb_f2_raw_all_banks_n4", category="F.2-mixed-raw")
def _f2_raw_all_banks(k: Kernel) -> None:
  """RAW chains on v[1]/v[2]/v[3]/v[4] — different banks, sequential."""
  for r in [1, 2, 3, 4]:
    for _ in range(4):
      k.emit(v_add_f32_e32(v[r], 1.0, v[r]))


# ═════════════════════════════════════════════════════════════════════════════
# F.3 — VMEM store deep interleaving / width / spacing (10 kernels)
# ═════════════════════════════════════════════════════════════════════════════
# Notes from HW run: earlier attempt with 8 consecutive stores to the same addr
# (mb_f3_store_chain_n8_indep) hung the GPU — the VMEM store queue cannot
# absorb that many sequential overlapping writes without a vmcnt drain. Also
# unaligned b64/b128 stores (v[0] offset = 4*tid, not 8/16-aligned) likely
# fault on real HW. This rewrite keeps all stores at b32, ≤4 per chain, and
# drains via s_waitcnt_vmcnt(0) between groups when the pattern requires >4.

@microbench(name="mb_f3_store_chain_n4_indep", category="F.3-vmem-deep")
def _f3_store_chain_n4(k: Kernel) -> None:
  """4 stores of independent VGPRs (extends mb_vmem_store_b32_chain_n4 which
  interleaves VALU; this one is a pure store chain)."""
  for i in range(4): k.emit(v_mov_b32_e32(v[10 + i], float(i + 1)))
  for i in range(4): k.emit(global_store_b32(addr=v[0], data=v[10 + i], saddr=s[0:1]))

@microbench(name="mb_f3_store_chain_n3_same", category="F.3-vmem-deep")
def _f3_store_chain_n3_same(k: Kernel) -> None:
  """3 stores of SAME VGPR — RAW on data reg."""
  for _ in range(3): k.emit(global_store_b32(addr=v[0], data=v[1], saddr=s[0:1]))

@microbench(name="mb_f3_store_spaced_by_nop", category="F.3-vmem-deep")
def _f3_store_spaced_by_nop(k: Kernel) -> None:
  """2 stores with nop gap in between."""
  k.emit(global_store_b32(addr=v[0], data=v[1], saddr=s[0:1]))
  for _ in range(4): k.emit(s_nop(0))
  k.emit(global_store_b32(addr=v[0], data=v[1], saddr=s[0:1]))

@microbench(name="mb_f3_store_spaced_by_valu", category="F.3-vmem-deep")
def _f3_store_spaced_by_valu(k: Kernel) -> None:
  """2 stores with 4 independent VALU in between."""
  k.emit(global_store_b32(addr=v[0], data=v[1], saddr=s[0:1]))
  for i in range(4): k.emit(v_add_f32_e32(v[20 + i], v[2], v[3]))
  k.emit(global_store_b32(addr=v[0], data=v[1], saddr=s[0:1]))

@microbench(name="mb_f3_store_spaced_by_raw_chain", category="F.3-vmem-deep")
def _f3_store_spaced_by_raw(k: Kernel) -> None:
  """2 stores with a RAW VALU chain in between."""
  k.emit(global_store_b32(addr=v[0], data=v[1], saddr=s[0:1]))
  for _ in range(4): k.emit(v_add_f32_e32(v[2], 1.0, v[2]))
  k.emit(global_store_b32(addr=v[0], data=v[1], saddr=s[0:1]))

@microbench(name="mb_f3_store_after_raw_chain_long", category="F.3-vmem-deep")
def _f3_store_after_long_raw(k: Kernel) -> None:
  """Store after a 16-deep RAW chain."""
  for _ in range(16): k.emit(v_add_f32_e32(v[1], 1.0, v[1]))
  k.emit(global_store_b32(addr=v[0], data=v[1], saddr=s[0:1]))

@microbench(name="mb_f3_store_chain_n4_drained", category="F.3-vmem-deep")
def _f3_store_chain_drained(k: Kernel) -> None:
  """4 stores with explicit vmcnt(0) drain between pairs — quantifies the
  store-pipe recovery between draining events."""
  for i in range(4): k.emit(v_mov_b32_e32(v[10 + i], float(i + 1)))
  k.emit(global_store_b32(addr=v[0], data=v[10], saddr=s[0:1]))
  k.emit(global_store_b32(addr=v[0], data=v[11], saddr=s[0:1]))
  k.emit(s_waitcnt_vmcnt(sdst=NULL, simm16=0))
  k.emit(global_store_b32(addr=v[0], data=v[12], saddr=s[0:1]))
  k.emit(global_store_b32(addr=v[0], data=v[13], saddr=s[0:1]))

@microbench(name="mb_f3_load_then_immediate_store", category="F.3-vmem-deep")
def _f3_load_then_store(k: Kernel) -> None:
  """Load → vmcnt drain → store. Tests VMEM read/write pipeline sequencing."""
  k.emit(global_load_b32(v[20], v[0], saddr=s[0:1]))
  k.emit(s_waitcnt_vmcnt(sdst=NULL, simm16=0))
  k.emit(global_store_b32(addr=v[0], data=v[20], saddr=s[0:1]))

@microbench(name="mb_f3_interleave_load_store", category="F.3-vmem-deep")
def _f3_interleave_load_store(k: Kernel) -> None:
  """Interleaved: store, load, drain, store, load (no trailing load to avoid
  hanging epilogue awaiting the outstanding load)."""
  k.emit(global_store_b32(addr=v[0], data=v[1], saddr=s[0:1]))
  k.emit(global_load_b32(v[20], v[0], saddr=s[0:1]))
  k.emit(s_waitcnt_vmcnt(sdst=NULL, simm16=0))
  k.emit(global_store_b32(addr=v[0], data=v[20], saddr=s[0:1]))

@microbench(name="mb_f3_store_pair_then_pair", category="F.3-vmem-deep")
def _f3_store_pair_then_pair(k: Kernel) -> None:
  """2 pairs of stores separated by a single VALU — characterizes the pair-
  level cadence vs a continuous 4-chain."""
  for i in range(4): k.emit(v_mov_b32_e32(v[10 + i], float(i + 1)))
  k.emit(global_store_b32(addr=v[0], data=v[10], saddr=s[0:1]))
  k.emit(global_store_b32(addr=v[0], data=v[11], saddr=s[0:1]))
  k.emit(v_add_f32_e32(v[20], v[2], v[3]))
  k.emit(global_store_b32(addr=v[0], data=v[12], saddr=s[0:1]))
  k.emit(global_store_b32(addr=v[0], data=v[13], saddr=s[0:1]))


# ═════════════════════════════════════════════════════════════════════════════
# F.4 — Transcendental chains (15 kernels)
# ═════════════════════════════════════════════════════════════════════════════

sweep("mb_f4_exp_chain", [2, 3, 4, 8],
  lambda n: (lambda k: [k.emit(v_exp_f32_e32(v[10], v[10])) for _ in range(n)]),
  category="F.4-trans-chain")

sweep("mb_f4_log_chain", [2, 4, 8],
  lambda n: (lambda k: [k.emit(v_log_f32_e32(v[10], v[10])) for _ in range(n)]),
  category="F.4-trans-chain")

sweep("mb_f4_sqrt_chain", [2, 4, 8],
  lambda n: (lambda k: [k.emit(v_sqrt_f32_e32(v[10], v[10])) for _ in range(n)]),
  category="F.4-trans-chain")

sweep("mb_f4_rcp_chain", [2, 4, 8],
  lambda n: (lambda k: [k.emit(v_rcp_f32_e32(v[10], v[10])) for _ in range(n)]),
  category="F.4-trans-chain")

sweep("mb_f4_rsq_chain", [2, 4, 8],
  lambda n: (lambda k: [k.emit(v_rsq_f32_e32(v[10], v[10])) for _ in range(n)]),
  category="F.4-trans-chain")

@microbench(name="mb_f4_trans_with_valu_inter", category="F.4-trans-chain")
def _f4_trans_with_valu(k: Kernel) -> None:
  """exp, v_add, exp, v_add — interleaved trans with regular VALU."""
  for _ in range(4):
    k.emit(v_exp_f32_e32(v[10], v[10]))
    k.emit(v_add_f32_e32(v[11], v[10], v[2]))

@microbench(name="mb_f4_mixed_trans_chain", category="F.4-trans-chain")
def _f4_mixed_trans(k: Kernel) -> None:
  """exp, log, sqrt, rcp — different trans ops."""
  k.emit(v_exp_f32_e32(v[10], v[10]))
  k.emit(v_log_f32_e32(v[11], v[10]))
  k.emit(v_sqrt_f32_e32(v[12], v[11]))
  k.emit(v_rcp_f32_e32(v[13], v[12]))


# ═════════════════════════════════════════════════════════════════════════════
# F.5 — Cross-regfile RAW (15 kernels)
# ═════════════════════════════════════════════════════════════════════════════

@microbench(name="mb_f5_readlane_chain_n4", category="F.5-cross-regfile")
def _f5_readlane_chain(k: Kernel) -> None:
  """Chain of v_readlane. VALU writes SGPR — unusual pipe."""
  for i in range(4): k.emit(v_readlane_b32(s[10 + i], v[1], i))

@microbench(name="mb_f5_writelane_chain_n4", category="F.5-cross-regfile")
def _f5_writelane_chain(k: Kernel) -> None:
  """Chain of v_writelane writes to same VGPR."""
  for i in range(4): k.emit(v_writelane_b32(v[10], s[0], i))

@microbench(name="mb_f5_readfirstlane_chain_n4", category="F.5-cross-regfile")
def _f5_readfirstlane_chain(k: Kernel) -> None:
  for i in range(4): k.emit(v_readfirstlane_b32_e32(s[10 + i], v[1]))

@microbench(name="mb_f5_valu_then_readlane", category="F.5-cross-regfile")
def _f5_valu_then_readlane(k: Kernel) -> None:
  """VALU chain followed by readlane reading the chain's output."""
  for _ in range(4): k.emit(v_add_f32_e32(v[1], 1.0, v[1]))
  k.emit(v_readlane_b32(s[10], v[1], 0))

@microbench(name="mb_f5_sgpr_chain_n4", category="F.5-cross-regfile")
def _f5_sgpr_chain(k: Kernel) -> None:
  """Chain of SALU s_add with RAW on s[20]."""
  k.emit(s_mov_b32(s[20], 1))
  for _ in range(4): k.emit(s_add_u32(s[20], s[20], 1))

@microbench(name="mb_f5_sgpr_then_valu", category="F.5-cross-regfile")
def _f5_sgpr_then_valu(k: Kernel) -> None:
  """SALU writes SGPR, VALU reads it — SGPR→VALU latency probe."""
  k.emit(s_mov_b32(s[20], 1))
  k.emit(v_add_f32_e32(v[10], s[20], v[1]))

@microbench(name="mb_f5_vcmp_then_cndmask_vcc_chain_n4", category="F.5-cross-regfile")
def _f5_cmp_cnd_vcc_chain(k: Kernel) -> None:
  """Alternating v_cmp (writes VCC) and v_cndmask (reads VCC)."""
  for _ in range(4):
    k.emit(v_cmp_gt_f32_e32(0.5, v[4]))
    k.emit(v_cndmask_b32_e32(v[10], 1.0, v[4]))

@microbench(name="mb_f5_vcmp_e64_sgpr_chain_n4", category="F.5-cross-regfile")
def _f5_cmp_e64_sgpr_chain(k: Kernel) -> None:
  """v_cmp_e64 writes specific SGPR, 4 different SGPRs."""
  for i in range(4): k.emit(v_cmp_gt_f32_e64(s[20 + i], 0.5, v[4]))


# ═════════════════════════════════════════════════════════════════════════════
# F.6 — Edge ops: VGPR bank-conflict / packed / 3-src chains (15 kernels)
# ═════════════════════════════════════════════════════════════════════════════

@microbench(name="mb_f6_vmul_int_chain_n4", category="F.6-edge-ops")
def _f6_vmul_int_chain(k: Kernel) -> None:
  """v_mul_lo_u32 RAW chain — 4cy pipe per-op?"""
  k.emit(v_mul_lo_u32(v[10], v[1], v[2]))
  for _ in range(3): k.emit(v_mul_lo_u32(v[10], v[10], v[2]))

@microbench(name="mb_f6_vmul_hi_chain_n4", category="F.6-edge-ops")
def _f6_vmul_hi_chain(k: Kernel) -> None:
  k.emit(v_mul_hi_u32(v[10], v[1], v[2]))
  for _ in range(3): k.emit(v_mul_hi_u32(v[10], v[10], v[2]))

@microbench(name="mb_f6_pk_add_f16_chain_n8", category="F.6-edge-ops")
def _f6_pk_add_chain(k: Kernel) -> None:
  for _ in range(8): k.emit(v_pk_add_f16(v[10], v[10], v[2]))

@microbench(name="mb_f6_pk_mul_f16_chain_n4", category="F.6-edge-ops")
def _f6_pk_mul_chain(k: Kernel) -> None:
  for _ in range(4): k.emit(v_pk_mul_f16(v[10], v[10], v[2]))

@microbench(name="mb_f6_bfi_chain_n8", category="F.6-edge-ops")
def _f6_bfi_chain(k: Kernel) -> None:
  for _ in range(8): k.emit(v_bfi_b32(v[10], v[1], v[10], v[2]))

@microbench(name="mb_f6_alignbyte_chain_n8", category="F.6-edge-ops")
def _f6_alignbyte_chain(k: Kernel) -> None:
  for _ in range(8): k.emit(v_alignbyte_b32(v[10], v[1], v[10], v[2]))

@microbench(name="mb_f6_and_chain_n8", category="F.6-edge-ops")
def _f6_and_chain(k: Kernel) -> None:
  for _ in range(8): k.emit(v_and_b32_e32(v[10], v[10], v[2]))

@microbench(name="mb_f6_lshrrev_chain_n4", category="F.6-edge-ops")
def _f6_lshr_chain(k: Kernel) -> None:
  for _ in range(4): k.emit(v_lshrrev_b32_e32(v[10], 1, v[10]))

@microbench(name="mb_f6_lshlrev_chain_n4", category="F.6-edge-ops")
def _f6_lshl_chain(k: Kernel) -> None:
  for _ in range(4): k.emit(v_lshlrev_b32_e32(v[10], 1, v[10]))

@microbench(name="mb_f6_max_chain_n4", category="F.6-edge-ops")
def _f6_max_chain(k: Kernel) -> None:
  for _ in range(4): k.emit(v_max_f32_e32(v[10], v[10], v[2]))

@microbench(name="mb_f6_min_chain_n4", category="F.6-edge-ops")
def _f6_min_chain(k: Kernel) -> None:
  for _ in range(4): k.emit(v_min_f32_e32(v[10], v[10], v[2]))

@microbench(name="mb_f6_fma_chain_n4", category="F.6-edge-ops")
def _f6_fma_chain(k: Kernel) -> None:
  """3-src VOP3 v_fma_f32 RAW chain."""
  for _ in range(4): k.emit(v_fma_f32(v[10], v[10], v[2], v[3]))

@microbench(name="mb_f6_madu32_chain_n8", category="F.6-edge-ops")
def _f6_madu32_chain(k: Kernel) -> None:
  for _ in range(8): k.emit(v_mad_u32_u24(v[10], v[10], v[2], v[3]))

@microbench(name="mb_f6_cvt_roundtrip_chain_n4", category="F.6-edge-ops")
def _f6_cvt_roundtrip(k: Kernel) -> None:
  """4 round-trips of f32->f16->f32."""
  for _ in range(4):
    k.emit(v_cvt_f16_f32_e32(v[10], v[1]))
    k.emit(v_cvt_f32_f16_e32(v[1], v[10]))

@microbench(name="mb_f6_pk_fma_f16_chain_n4", category="F.6-edge-ops")
def _f6_pk_fma_chain(k: Kernel) -> None:
  for _ in range(4): k.emit(v_pk_fma_f16(v[10], v[10], v[2], v[3]))
