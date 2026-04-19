#!/usr/bin/env python3
"""Batch E — Wave-variance + difficult-instruction probes.

~30 surgical probes aimed at the remaining strict-mode gap (per-wave dt
variance that MODAL hides). Each kernel isolates one axis of the HW
arbitration behavior the emulator does not yet model.

  E.1 (7 kernels) — VGPR-RAW chain-length boundary
    Where does "wave 0 wins bypass, waves 1-15 stall" start? Existing
    captures show n=4 → mostly wave 0..9 win; n=8 → wave 0..3 win;
    n=16 → only wave 0 wins. Fills the gap at n=3,5,6,7,10,12,24 so
    we can fit a per-wave credit model.

  E.2 (6 kernels) — VMEM store stagger
    mb_vmem_store_b32_chain_n4 [11,13,15] shows VALU→store pre-store
    VALU dt grows from 8 (wave 0) to 39 (wave 15). Isolate: solo store,
    back-to-back stores (no VALU), short/long chains with independent
    VGPRs.

  E.3 (5 kernels) — Wave-launch startup stagger
    Post-vmcnt-drain first-VALU shows W0=1, W2=23, W3=41 (mb_vcmp_cndmask_k8
    [5]). Isolate the stagger across different first-VALU types so we can
    calibrate WAVE_LAUNCH_STAGGER properly (is it per-wave linear, or
    bucketed? does it depend on what the first VALU is?).

  E.4 (10 kernels) — Under-characterized instructions
    SMEM load sizes, cross-lane ops (readlane/writelane/readfirstlane),
    integer mul (v_mul_lo_u32), VOP3 bit-manipulation (v_bfi/v_alignbyte),
    packed half (v_pk_add_f16), and f16 conversions. These paths feed
    real kernels (matmul, reductions) and their wave-variance signatures
    are unknown.

Naming: mb_e{N}_{what}. Standard microbench prologue/epilogue.
"""
from __future__ import annotations
import struct

from tinygrad.renderer.amd.dsl import s, v, NULL  # noqa: F401
from tinygrad.runtime.autogen.amd.rdna3.ins import (
  v_add_f32_e32, v_mul_f32_e32, v_fmac_f32_e32, v_mov_b32_e32,
  v_cmp_gt_f32_e32, v_cndmask_b32_e32,
  v_mul_lo_u32, v_mul_hi_u32, v_mad_u32_u24,
  v_bfi_b32, v_alignbyte_b32,
  v_readlane_b32, v_writelane_b32, v_readfirstlane_b32_e32,
  v_cvt_f16_f32_e32, v_cvt_f32_f16_e32, v_pk_add_f16, v_pk_mul_f16,
  v_add_nc_u32_e32, v_lshlrev_b32_e32,
  s_mov_b32, s_nop, s_waitcnt, s_waitcnt_vmcnt, s_waitcnt_lgkmcnt,
  s_load_b32, s_load_b64, s_load_b128, s_load_b256,
  global_load_b32, global_store_b32,
  VOPD, VOPDOp,
)

from extra.sqtt.rgp.microbench import microbench, sweep, Kernel  # noqa: F401


def _f2i(f: float) -> int:
  return struct.unpack('I', struct.pack('f', f))[0]


# ═════════════════════════════════════════════════════════════════════════════
# E.1 — VGPR-RAW chain-length boundary  (fills gaps in existing n=1,2,4,8,16)
# ═════════════════════════════════════════════════════════════════════════════
# Hypothesis: HW uses per-wave issue credits that drain with chain depth.
# Short chains don't exhaust → more waves win bypass.
# Long chains exhaust → only wave 0 wins.
# These 7 lengths plus the existing 5 let us fit a credit-depletion model.

sweep(
  "mb_e1_valu_add", [3, 5, 6, 7, 10, 12, 24],
  lambda n: (lambda k: [k.emit(v_add_f32_e32(v[1], 1.0, v[1])) for _ in range(n)]),
  category="E.1-chain-boundary",
)


# ═════════════════════════════════════════════════════════════════════════════
# E.2 — VMEM-store stagger probes
# ═════════════════════════════════════════════════════════════════════════════

@microbench(name="mb_e2_store_solo", category="E.2-store-stagger")
def _e2_store_solo(k: Kernel) -> None:
  """Single store, no preceding VALU. Tests cold-store wave-stagger."""
  k.emit(global_store_b32(addr=v[0], data=v[1], saddr=s[0:1]))

@microbench(name="mb_e2_store_pair_same_vgpr", category="E.2-store-stagger")
def _e2_store_pair_same(k: Kernel) -> None:
  """Two back-to-back stores of same VGPR (RAW on data). Store→store spacing."""
  k.emit(global_store_b32(addr=v[0], data=v[1], saddr=s[0:1]))
  k.emit(global_store_b32(addr=v[0], data=v[1], saddr=s[0:1]))

@microbench(name="mb_e2_store_pair_indep_vgpr", category="E.2-store-stagger")
def _e2_store_pair_indep(k: Kernel) -> None:
  """Two stores, independent data VGPRs. No RAW — pure store-pipe spacing."""
  k.emit(v_mov_b32_e32(v[10], 1.0))
  k.emit(v_mov_b32_e32(v[11], 2.0))
  k.emit(global_store_b32(addr=v[0], data=v[10], saddr=s[0:1]))
  k.emit(global_store_b32(addr=v[0], data=v[11], saddr=s[0:1]))

@microbench(name="mb_e2_store_chain_indep_n3", category="E.2-store-stagger")
def _e2_store_chain_indep_n3(k: Kernel) -> None:
  """3 stores of independent VGPRs (no RAW). Stride = store-pipe width."""
  for i, f in enumerate([1.0, 2.0, 3.0]):
    k.emit(v_mov_b32_e32(v[10 + i], f))
  for i in range(3):
    k.emit(global_store_b32(addr=v[0], data=v[10 + i], saddr=s[0:1]))

@microbench(name="mb_e2_store_chain_indep_n5", category="E.2-store-stagger")
def _e2_store_chain_indep_n5(k: Kernel) -> None:
  """5 stores of independent VGPRs."""
  for i in range(5):
    k.emit(v_mov_b32_e32(v[10 + i], float(i + 1)))
  for i in range(5):
    k.emit(global_store_b32(addr=v[0], data=v[10 + i], saddr=s[0:1]))

@microbench(name="mb_e2_store_after_longnop", category="E.2-store-stagger")
def _e2_store_after_longnop(k: Kernel) -> None:
  """Store after 16× s_nop(0). Does the store-pipe drain with idle time?"""
  for _ in range(16): k.emit(s_nop(0))
  k.emit(global_store_b32(addr=v[0], data=v[1], saddr=s[0:1]))


# ═════════════════════════════════════════════════════════════════════════════
# E.3 — Wave-launch startup stagger
# ═════════════════════════════════════════════════════════════════════════════
# After the standard prologue's s_waitcnt_vmcnt(NULL), the first body VALU
# shows wave-linear stagger (W0=1, W2=23, W3=41 in mb_vcmp_cndmask_k8). These
# isolate what the stagger depends on.

@microbench(name="mb_e3_post_waitcnt_vmov", category="E.3-launch-stagger")
def _e3_post_waitcnt_vmov(k: Kernel) -> None:
  """Post-vmcnt-drain: single v_mov to fresh VGPR. Baseline wave-stagger."""
  k.emit(v_mov_b32_e32(v[10], 1.0))

@microbench(name="mb_e3_post_waitcnt_vadd_fresh", category="E.3-launch-stagger")
def _e3_post_waitcnt_vadd_fresh(k: Kernel) -> None:
  """Post-drain: v_add writing fresh VGPR (no RAW). Does it differ from v_mov?"""
  k.emit(v_add_f32_e32(v[10], 1.0, v[1]))

@microbench(name="mb_e3_post_waitcnt_vmov_then_chain", category="E.3-launch-stagger")
def _e3_post_waitcnt_vmov_then_chain(k: Kernel) -> None:
  """Post-drain: 8 back-to-back v_movs to independent VGPRs. Watch how the
  stagger propagates through indep-VGPR chain."""
  for i in range(8):
    k.emit(v_mov_b32_e32(v[10 + i], float(i + 1)))

@microbench(name="mb_e3_post_waitcnt_vopd", category="E.3-launch-stagger")
def _e3_post_waitcnt_vopd(k: Kernel) -> None:
  """Post-drain: VOPD. Does VOPD first-op hit the same launch stagger?"""
  for i in range(8):
    k.emit(v_mov_b32_e32(v[4 + i], float(i + 1)))
  k.emit(VOPD(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_ADD_F32,
              v[20], v[21], v[4], v[5], v[6], v[7]))

@microbench(name="mb_e3_extra_waitcnt_nop16_vmov", category="E.3-launch-stagger")
def _e3_extra_waitcnt_nop16_vmov(k: Kernel) -> None:
  """After prologue drain, 16 s_nop(0) then v_mov. Does the extra idle time
  give later waves time to catch up (reduce stagger)?"""
  for _ in range(16): k.emit(s_nop(0))
  k.emit(v_mov_b32_e32(v[10], 1.0))


# ═════════════════════════════════════════════════════════════════════════════
# E.4 — Under-characterized instruction probes
# ═════════════════════════════════════════════════════════════════════════════

# E.4.1 — Integer multiplies (multi-cycle VALU — 4cy per AMD ISA spec)
@microbench(name="mb_e4_vmul_lo_u32_n1", category="E.4-int-mul")
def _e4_vmul_lo_u32_n1(k: Kernel) -> None:
  k.emit(v_mul_lo_u32(v[10], v[1], v[1]))

@microbench(name="mb_e4_vmul_lo_u32_n4", category="E.4-int-mul")
def _e4_vmul_lo_u32_n4(k: Kernel) -> None:
  """4× v_mul_lo_u32 RAW on v[10]. Tests multi-cycle VALU issue cadence."""
  k.emit(v_mul_lo_u32(v[10], v[1], v[1]))
  for _ in range(3): k.emit(v_mul_lo_u32(v[10], v[10], v[1]))

@microbench(name="mb_e4_vmad_u32_u24_n4", category="E.4-int-mul")
def _e4_vmad_u32_u24_n4(k: Kernel) -> None:
  """v_mad_u32_u24: 3-source integer mad, unsigned 24-bit inputs."""
  k.emit(v_mad_u32_u24(v[10], v[1], v[1], v[1]))
  for _ in range(3): k.emit(v_mad_u32_u24(v[10], v[10], v[1], v[1]))

# E.4.2 — Cross-lane ops
@microbench(name="mb_e4_readlane_n1", category="E.4-cross-lane")
def _e4_readlane_n1(k: Kernel) -> None:
  """v_readlane: VALU lane→SGPR broadcast. Unusual pipe (VALU writes SGPR)."""
  k.emit(v_readlane_b32(s[10], v[1], 0))

@microbench(name="mb_e4_writelane_n1", category="E.4-cross-lane")
def _e4_writelane_n1(k: Kernel) -> None:
  """v_writelane: SGPR→VGPR single-lane. Writes v[10].lane[0]."""
  k.emit(v_writelane_b32(v[10], s[0], 0))

@microbench(name="mb_e4_readfirstlane_n1", category="E.4-cross-lane")
def _e4_readfirstlane_n1(k: Kernel) -> None:
  """v_readfirstlane: single-lane VGPR→SGPR (mask & first lane)."""
  k.emit(v_readfirstlane_b32_e32(s[10], v[1]))

# E.4.3 — VOP3 bit manipulation
@microbench(name="mb_e4_bfi_n4", category="E.4-bit-manip")
def _e4_bfi_n4(k: Kernel) -> None:
  """v_bfi_b32: 3-source bitfield insert. RAW chain on v[10]."""
  k.emit(v_bfi_b32(v[10], v[1], v[1], v[1]))
  for _ in range(3): k.emit(v_bfi_b32(v[10], v[10], v[1], v[1]))

@microbench(name="mb_e4_alignbyte_n4", category="E.4-bit-manip")
def _e4_alignbyte_n4(k: Kernel) -> None:
  """v_alignbyte_b32: 3-src (hi, lo, shift). RAW chain on v[10]."""
  k.emit(v_alignbyte_b32(v[10], v[1], v[1], v[1]))
  for _ in range(3): k.emit(v_alignbyte_b32(v[10], v[10], v[1], v[1]))

# E.4.4 — Packed half
@microbench(name="mb_e4_pk_add_f16_n4", category="E.4-packed-half")
def _e4_pk_add_f16_n4(k: Kernel) -> None:
  """v_pk_add_f16: packed fp16 add. RAW chain on v[10]."""
  k.emit(v_pk_add_f16(v[10], v[1], v[1]))
  for _ in range(3): k.emit(v_pk_add_f16(v[10], v[10], v[1]))

@microbench(name="mb_e4_cvt_f32_f16_chain_n4", category="E.4-packed-half")
def _e4_cvt_f32_f16_chain_n4(k: Kernel) -> None:
  """f32→f16→f32 round-trip conversion chain. Tests transcendental-like pipe."""
  k.emit(v_cvt_f16_f32_e32(v[10], v[1]))
  k.emit(v_cvt_f32_f16_e32(v[11], v[10]))
  k.emit(v_cvt_f16_f32_e32(v[12], v[11]))
  k.emit(v_cvt_f32_f16_e32(v[13], v[12]))
