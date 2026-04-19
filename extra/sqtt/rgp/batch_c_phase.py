#!/usr/bin/env python3
"""Batch C — phase-state and VOPD-chain edge-case microbenches.

28 targeted probes aimed at the 11 mismatches remaining after the 2026-04-19
phase-model extensions. Each kernel isolates exactly one axis of the HW
behavior we're trying to pin down:

  C.1 (8 kernels) — VOPD-chain-at-1cy: V_DUAL_MOV with SGPR/literal sources,
                    which HW pipelines at 1cy (where [8] pattern).
  C.2 (6 kernels) — Depctr→cmp→cndmask→VOPD phase-state: various chain
                    depths/compositions to calibrate the +2cy VOPD warm-up
                    and the GAP=1 applicability window.
  C.3 (6 kernels) — Post-DRAM-idle VOPD: VOPDs after a long waitcnt/vmcnt
                    stall vs fresh state (exp_chain [26] pattern).
  C.4 (4 kernels) — Cndmask first-read / subsequent-read taper: fresh SGPR
                    reads in a phase-shifted chain of varying depth.
  C.5 (4 kernels) — b128 VMEM-store VALU-forwarding with different VGPR
                    stride patterns (where [18] pattern).

All kernels follow the standard microbench prologue/epilogue. Each body
seeds required state explicitly so the timing measurement is isolated.
"""
from __future__ import annotations

from tinygrad.renderer.amd.dsl import s, v, NULL, VCC_LO  # noqa: F401
from tinygrad.runtime.autogen.amd.rdna3.ins import (
  v_add_f32_e32, v_mul_f32_e32, v_mov_b32_e32,
  v_exp_f32_e32, v_log_f32_e32,
  v_cmp_gt_f32_e32, v_cmp_gt_f32_e64, v_cndmask_b32_e32, v_cndmask_b32_e64,
  s_mov_b32, s_nop, s_waitcnt, s_waitcnt_depctr, s_waitcnt_vmcnt,
  global_store_b128,
  VOPD, VOPD_LIT, VOPDOp,
)

from extra.sqtt.rgp.microbench import microbench, Kernel  # noqa: F401


# ── Shared seeders ───────────────────────────────────────────────────────────

def _seed_vgprs_8(k: "Kernel") -> None:
  """Seed v[4..11] with non-zero float values so VOPDs have valid inputs."""
  for i in range(8): k.emit(v_mov_b32_e32(v[4 + i], float(i + 1)))

def _seed_sgprs(k: "Kernel") -> None:
  """Seed s[4..7] via v_cmp chain writing explicit SGPRs."""
  k.emit(v_cmp_gt_f32_e64(s[4], 0.5, v[4]))
  k.emit(v_cmp_gt_f32_e64(s[5], 0.5, v[5]))
  k.emit(v_cmp_gt_f32_e64(s[6], 0.5, v[6]))
  k.emit(v_cmp_gt_f32_e64(s[7], 0.5, v[7]))


# ── C.1 — VOPD chain-at-1cy with SGPR / literal sources ─────────────────────
# Target: exp_chain [51] VOPD_LIT, where [8] V_DUAL_MOV with SGPR source
# Hypothesis: VOPDs that don't read VGPRs pipeline at 1cy (not 4cy).

@microbench(name="mb_vopd_dualmov_sgpr_pair", category="C.1-vopd-no-vgpr")
def _c1_vopd_dualmov_sgpr_pair(k: Kernel) -> None:
  _seed_sgprs(k)
  # Two VOPDs each doing V_DUAL_MOV with SGPR sources only — no VGPR reads
  k.emit(VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_MOV_B32, v[20], v[21], s[4], s[5]))
  k.emit(VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_MOV_B32, v[22], v[23], s[6], s[7]))

@microbench(name="mb_vopd_dualmov_sgpr_chain_n4", category="C.1-vopd-no-vgpr")
def _c1_vopd_dualmov_sgpr_chain_n4(k: Kernel) -> None:
  _seed_sgprs(k)
  # 4× V_DUAL_MOV with SGPR sources — chain at 1cy per HW
  k.emit(VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_MOV_B32, v[20], v[21], s[4], s[5]))
  k.emit(VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_MOV_B32, v[22], v[23], s[6], s[7]))
  k.emit(VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_MOV_B32, v[24], v[25], s[4], s[5]))
  k.emit(VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_MOV_B32, v[26], v[27], s[6], s[7]))

def _f2i(f: float) -> int:
  import struct
  return struct.unpack('I', struct.pack('f', f))[0]

@microbench(name="mb_vopd_dualmov_lit_pair", category="C.1-vopd-no-vgpr")
def _c1_vopd_dualmov_lit_pair(k: Kernel) -> None:
  """Two VOPD_LIT (FMAAK+MOV) back-to-back with literal. Tests VOPD_LIT chain spacing."""
  _seed_vgprs_8(k)
  for pair in [(20, 21), (22, 23)]:
    k.emit(VOPD_LIT(
      opx=VOPDOp.V_DUAL_FMAAK_F32, opy=VOPDOp.V_DUAL_MOV_B32,
      vdstx=v[pair[0]],   srcx0=v[6], vsrcx1=v[8],
      vdsty=v[pair[1]],   srcy0=v[7], vsrcy1=v[9],
      literal=_f2i(2.0),
    ))

@microbench(name="mb_vopd_mov_add_mix", category="C.1-vopd-no-vgpr")
def _c1_vopd_mov_add_mix(k: Kernel) -> None:
  _seed_vgprs_8(k)
  _seed_sgprs(k)
  # First VOPD: mov+mov (no VGPR read). Second: add+add (VGPR read).
  # Expect first→second gap = 4cy (standard), not 1cy.
  k.emit(VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_MOV_B32, v[20], v[21], s[4], s[5]))
  k.emit(VOPD(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_ADD_F32, v[22], v[23], v[4], v[5], v[6], v[7]))

@microbench(name="mb_vopd_add_mov_mix", category="C.1-vopd-no-vgpr")
def _c1_vopd_add_mov_mix(k: Kernel) -> None:
  _seed_vgprs_8(k)
  _seed_sgprs(k)
  # First VOPD: add+add (VGPR read). Second: mov+mov (no VGPR read).
  # Expect first→second gap: if "no-VGPR-read = 1cy" rule applies to consumer side.
  k.emit(VOPD(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_ADD_F32, v[20], v[21], v[4], v[5], v[6], v[7]))
  k.emit(VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_MOV_B32, v[22], v[23], s[4], s[5]))

@microbench(name="mb_vopd_dualmov_all_lit_chain_n4", category="C.1-vopd-no-vgpr")
def _c1_vopd_dualmov_all_lit_chain_n4(k: Kernel) -> None:
  """4-chain of VOPD_LIT FMAAK+MOV — 1cy per Batch B confirmed. Tests chain pipelining."""
  _seed_vgprs_8(k)
  for reg_pair in [(20, 21), (22, 23), (24, 25), (26, 27)]:
    k.emit(VOPD_LIT(
      opx=VOPDOp.V_DUAL_FMAAK_F32, opy=VOPDOp.V_DUAL_MOV_B32,
      vdstx=v[reg_pair[0]], srcx0=v[6], vsrcx1=v[8],
      vdsty=v[reg_pair[1]], srcy0=v[7], vsrcy1=v[9],
      literal=_f2i(2.0),
    ))

@microbench(name="mb_vopd_literal_then_vgpr", category="C.1-vopd-no-vgpr")
def _c1_vopd_literal_then_vgpr(k: Kernel) -> None:
  """LIT-source VOPD then VGPR-source VOPD. How much does the transition cost?"""
  _seed_vgprs_8(k)
  k.emit(VOPD_LIT(
    opx=VOPDOp.V_DUAL_FMAAK_F32, opy=VOPDOp.V_DUAL_MOV_B32,
    vdstx=v[20], srcx0=v[6], vsrcx1=v[8],
    vdsty=v[21], srcy0=v[7], vsrcy1=v[9],
    literal=_f2i(2.0),
  ))
  k.emit(VOPD(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_ADD_F32, v[22], v[23], v[4], v[5], v[6], v[7]))

@microbench(name="mb_vopd_vgpr_then_literal", category="C.1-vopd-no-vgpr")
def _c1_vopd_vgpr_then_literal(k: Kernel) -> None:
  """VGPR-source VOPD then LIT-source VOPD. Reverse direction."""
  _seed_vgprs_8(k)
  k.emit(VOPD(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_ADD_F32, v[20], v[21], v[4], v[5], v[6], v[7]))
  k.emit(VOPD_LIT(
    opx=VOPDOp.V_DUAL_FMAAK_F32, opy=VOPDOp.V_DUAL_MOV_B32,
    vdstx=v[22], srcx0=v[6], vsrcx1=v[8],
    vdsty=v[23], srcy0=v[7], vsrcy1=v[9],
    literal=_f2i(2.0),
  ))


# ── C.2 — Depctr→cmp→cndmask→VOPD phase-state chain sweeps ──────────────────
# Target: exp_chain [37], [61] — VOPD after phase-shifted cndmask chain.
# Hypothesis: post-depctr the VOPD closing the chain pays +2cy. Sweep the
# chain depth to see if the rule is depth-dependent.

def _depctr_sequence(k: Kernel) -> None:
  """Emit a dummy trans + depctr so the chain is in post-depctr phase."""
  k.emit(v_exp_f32_e32(v[10], v[4]))
  k.emit(s_waitcnt_depctr(0xFFF))

@microbench(name="mb_c2_depctr_cmp2_cnd2_vopd", category="C.2-phase-chain")
def _c2_depctr_cmp2_cnd2_vopd(k: Kernel) -> None:
  _seed_vgprs_8(k)
  _depctr_sequence(k)
  # 2 cmp writes, 2 cndmask reads, then VOPD closing
  k.emit(v_cmp_gt_f32_e64(s[4], 0.5, v[4]))
  k.emit(v_cmp_gt_f32_e64(s[5], 0.5, v[5]))
  k.emit(v_cndmask_b32_e64(v[20], 1.0, v[4], s[4]))
  k.emit(v_cndmask_b32_e64(v[21], 1.0, v[5], s[5]))
  k.emit(VOPD(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_ADD_F32, v[22], v[23], v[4], v[5], v[6], v[7]))

@microbench(name="mb_c2_depctr_cmp3_cnd3_vopd", category="C.2-phase-chain")
def _c2_depctr_cmp3_cnd3_vopd(k: Kernel) -> None:
  _seed_vgprs_8(k)
  _depctr_sequence(k)
  k.emit(v_cmp_gt_f32_e64(s[4], 0.5, v[4]))
  k.emit(v_cmp_gt_f32_e64(s[5], 0.5, v[5]))
  k.emit(v_cmp_gt_f32_e64(s[6], 0.5, v[6]))
  k.emit(v_cndmask_b32_e64(v[20], 1.0, v[4], s[4]))
  k.emit(v_cndmask_b32_e64(v[21], 1.0, v[5], s[5]))
  k.emit(v_cndmask_b32_e64(v[22], 1.0, v[6], s[6]))
  k.emit(VOPD(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_ADD_F32, v[24], v[25], v[4], v[5], v[6], v[7]))

@microbench(name="mb_c2_depctr_cmp4_cnd4_vopd", category="C.2-phase-chain")
def _c2_depctr_cmp4_cnd4_vopd(k: Kernel) -> None:
  _seed_vgprs_8(k)
  _depctr_sequence(k)
  k.emit(v_cmp_gt_f32_e64(s[4], 0.5, v[4]))
  k.emit(v_cmp_gt_f32_e64(s[5], 0.5, v[5]))
  k.emit(v_cmp_gt_f32_e64(s[6], 0.5, v[6]))
  k.emit(v_cmp_gt_f32_e64(s[7], 0.5, v[7]))
  k.emit(v_cndmask_b32_e64(v[20], 1.0, v[4], s[4]))
  k.emit(v_cndmask_b32_e64(v[21], 1.0, v[5], s[5]))
  k.emit(v_cndmask_b32_e64(v[22], 1.0, v[6], s[6]))
  k.emit(v_cndmask_b32_e64(v[23], 1.0, v[7], s[7]))
  k.emit(VOPD(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_ADD_F32, v[24], v[25], v[4], v[5], v[6], v[7]))

@microbench(name="mb_c2_nodepctr_cmp4_cnd4_vopd", category="C.2-phase-chain")
def _c2_nodepctr_cmp4_cnd4_vopd(k: Kernel) -> None:
  """Control: same chain WITHOUT depctr — should show no +2cy VOPD penalty."""
  _seed_vgprs_8(k)
  k.emit(v_cmp_gt_f32_e64(s[4], 0.5, v[4]))
  k.emit(v_cmp_gt_f32_e64(s[5], 0.5, v[5]))
  k.emit(v_cmp_gt_f32_e64(s[6], 0.5, v[6]))
  k.emit(v_cmp_gt_f32_e64(s[7], 0.5, v[7]))
  k.emit(v_cndmask_b32_e64(v[20], 1.0, v[4], s[4]))
  k.emit(v_cndmask_b32_e64(v[21], 1.0, v[5], s[5]))
  k.emit(v_cndmask_b32_e64(v[22], 1.0, v[6], s[6]))
  k.emit(v_cndmask_b32_e64(v[23], 1.0, v[7], s[7]))
  k.emit(VOPD(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_ADD_F32, v[24], v[25], v[4], v[5], v[6], v[7]))

@microbench(name="mb_c2_depctr_cmp_vcc_cnd_vopd", category="C.2-phase-chain")
def _c2_depctr_cmp_vcc_cnd_vopd(k: Kernel) -> None:
  """Mixed VCC-first (like exp_chain [33]) followed by SGPR reads."""
  _seed_vgprs_8(k)
  _depctr_sequence(k)
  k.emit(v_cmp_gt_f32_e32(0.5, v[4]))  # writes VCC
  k.emit(v_cmp_gt_f32_e64(s[4], 0.5, v[5]))
  k.emit(v_cmp_gt_f32_e64(s[5], 0.5, v[6]))
  k.emit(v_cndmask_b32_e32(v[20], 1.0, v[4]))  # reads VCC
  k.emit(v_cndmask_b32_e64(v[21], 1.0, v[4], s[4]))
  k.emit(v_cndmask_b32_e64(v[22], 1.0, v[5], s[5]))
  k.emit(VOPD(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_ADD_F32, v[24], v[25], v[4], v[5], v[6], v[7]))

@microbench(name="mb_c2_depctr_vopd_pair_after_cnd", category="C.2-phase-chain")
def _c2_depctr_vopd_pair_after_cnd(k: Kernel) -> None:
  """Post-depctr cmp+cndmask then VOPD-pair: first pays +2cy, 2nd pays 2cy after."""
  _seed_vgprs_8(k)
  _depctr_sequence(k)
  k.emit(v_cmp_gt_f32_e64(s[4], 0.5, v[4]))
  k.emit(v_cmp_gt_f32_e64(s[5], 0.5, v[5]))
  k.emit(v_cndmask_b32_e64(v[20], 1.0, v[4], s[4]))
  k.emit(v_cndmask_b32_e64(v[21], 1.0, v[5], s[5]))
  k.emit(VOPD(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_ADD_F32, v[22], v[23], v[4], v[5], v[6], v[7]))
  k.emit(VOPD(VOPDOp.V_DUAL_MUL_F32, VOPDOp.V_DUAL_MUL_F32, v[24], v[25], v[4], v[5], v[6], v[7]))


# ── C.3 — Post-DRAM-idle VOPD ─────────────────────────────────────────────
# Target: exp_chain [26] — VOPD after a DRAM wait (HW=1cy, emu predicts 3).
# Hypothesis: after a long waitcnt drain the VOPD state is fully reset and
# the next VOPD runs at 1cy regardless of prior VOPD activity.

@microbench(name="mb_c3_vmem_wait_then_vopd", category="C.3-post-dram")
def _c3_vmem_wait_then_vopd(k: Kernel) -> None:
  _seed_vgprs_8(k)
  # Earlier VOPD to set pipe state
  k.emit(VOPD(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_ADD_F32, v[20], v[21], v[4], v[5], v[6], v[7]))
  # vmcnt wait with no outstanding vmem ops — still counts as drain
  k.emit(s_waitcnt_vmcnt(NULL, 0))
  # VOPD after drain — expect HW 1cy
  k.emit(VOPD(VOPDOp.V_DUAL_MUL_F32, VOPDOp.V_DUAL_MUL_F32, v[22], v[23], v[4], v[5], v[6], v[7]))

@microbench(name="mb_c3_snop_long_then_vopd", category="C.3-post-dram")
def _c3_snop_long_then_vopd(k: Kernel) -> None:
  _seed_vgprs_8(k)
  k.emit(VOPD(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_ADD_F32, v[20], v[21], v[4], v[5], v[6], v[7]))
  k.emit(s_nop(15))  # 16cy idle
  k.emit(VOPD(VOPDOp.V_DUAL_MUL_F32, VOPDOp.V_DUAL_MUL_F32, v[22], v[23], v[4], v[5], v[6], v[7]))

@microbench(name="mb_c3_snop_short_then_vopd", category="C.3-post-dram")
def _c3_snop_short_then_vopd(k: Kernel) -> None:
  _seed_vgprs_8(k)
  k.emit(VOPD(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_ADD_F32, v[20], v[21], v[4], v[5], v[6], v[7]))
  k.emit(s_nop(3))   # 4cy idle
  k.emit(VOPD(VOPDOp.V_DUAL_MUL_F32, VOPDOp.V_DUAL_MUL_F32, v[22], v[23], v[4], v[5], v[6], v[7]))

@microbench(name="mb_c3_cndmask_after_idle_then_vopd", category="C.3-post-dram")
def _c3_cndmask_after_idle_then_vopd(k: Kernel) -> None:
  """Mimics exp_chain [23-26]: cndmask → idle → VOPD pattern."""
  _seed_vgprs_8(k)
  _seed_sgprs(k)
  k.emit(v_cndmask_b32_e64(v[20], 1.0, v[4], s[4]))
  k.emit(v_cndmask_b32_e64(v[21], 1.0, v[5], s[5]))
  k.emit(s_waitcnt(0))  # empty waitcnt, still drains
  k.emit(VOPD(VOPDOp.V_DUAL_MUL_F32, VOPDOp.V_DUAL_MUL_F32, v[22], v[23], v[4], v[5], v[6], v[7]))

@microbench(name="mb_c3_vopd_wait_vopd_chain", category="C.3-post-dram")
def _c3_vopd_wait_vopd_chain(k: Kernel) -> None:
  _seed_vgprs_8(k)
  # Chain: VOPD → wait → 4× VOPD
  k.emit(VOPD(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_ADD_F32, v[20], v[21], v[4], v[5], v[6], v[7]))
  k.emit(s_waitcnt(0))
  for dest_pair in [(22, 23), (24, 25), (26, 27), (28, 29)]:
    k.emit(VOPD(VOPDOp.V_DUAL_MUL_F32, VOPDOp.V_DUAL_MUL_F32,
                v[dest_pair[0]], v[dest_pair[1]], v[4], v[5], v[6], v[7]))

@microbench(name="mb_c3_depctr_then_vopd_fresh", category="C.3-post-dram")
def _c3_depctr_then_vopd_fresh(k: Kernel) -> None:
  """Depctr with no pending trans — plain drain. Then VOPD — no phase shift."""
  _seed_vgprs_8(k)
  k.emit(VOPD(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_ADD_F32, v[20], v[21], v[4], v[5], v[6], v[7]))
  k.emit(s_waitcnt_depctr(0xFFF))
  k.emit(VOPD(VOPDOp.V_DUAL_MUL_F32, VOPDOp.V_DUAL_MUL_F32, v[22], v[23], v[4], v[5], v[6], v[7]))


# ── C.4 — Cndmask taper positions in phase-shifted chain ────────────────────
# Target: exp_chain [34], [56], [57] — first cndmask-reads-SGPR timing.
# Sweep chain depth (1, 2, 3, 4) to characterize the read-latency curve.

@microbench(name="mb_c4_depctr_chain_n1", category="C.4-cndmask-taper")
def _c4_depctr_chain_n1(k: Kernel) -> None:
  _seed_vgprs_8(k)
  _depctr_sequence(k)
  k.emit(v_cmp_gt_f32_e64(s[4], 0.5, v[4]))
  k.emit(v_cndmask_b32_e64(v[20], 1.0, v[4], s[4]))

@microbench(name="mb_c4_depctr_chain_n2", category="C.4-cndmask-taper")
def _c4_depctr_chain_n2(k: Kernel) -> None:
  _seed_vgprs_8(k)
  _depctr_sequence(k)
  k.emit(v_cmp_gt_f32_e64(s[4], 0.5, v[4]))
  k.emit(v_cmp_gt_f32_e64(s[5], 0.5, v[5]))
  k.emit(v_cndmask_b32_e64(v[20], 1.0, v[4], s[4]))
  k.emit(v_cndmask_b32_e64(v[21], 1.0, v[5], s[5]))

@microbench(name="mb_c4_depctr_chain_n3", category="C.4-cndmask-taper")
def _c4_depctr_chain_n3(k: Kernel) -> None:
  _seed_vgprs_8(k)
  _depctr_sequence(k)
  k.emit(v_cmp_gt_f32_e64(s[4], 0.5, v[4]))
  k.emit(v_cmp_gt_f32_e64(s[5], 0.5, v[5]))
  k.emit(v_cmp_gt_f32_e64(s[6], 0.5, v[6]))
  k.emit(v_cndmask_b32_e64(v[20], 1.0, v[4], s[4]))
  k.emit(v_cndmask_b32_e64(v[21], 1.0, v[5], s[5]))
  k.emit(v_cndmask_b32_e64(v[22], 1.0, v[6], s[6]))

@microbench(name="mb_c4_depctr_chain_n4_vcc_first", category="C.4-cndmask-taper")
def _c4_depctr_chain_n4_vcc_first(k: Kernel) -> None:
  """Exact exp_chain [29-36] structure: VCC-write cmp + 3 SGPR-write cmps, VCC-read cndmask + 3 SGPR-read cndmasks."""
  _seed_vgprs_8(k)
  _depctr_sequence(k)
  k.emit(v_cmp_gt_f32_e32(0.5, v[4]))
  k.emit(v_cmp_gt_f32_e64(s[4], 0.5, v[5]))
  k.emit(v_cmp_gt_f32_e64(s[5], 0.5, v[6]))
  k.emit(v_cmp_gt_f32_e64(s[6], 0.5, v[7]))
  k.emit(v_cndmask_b32_e32(v[20], 1.0, v[4]))
  k.emit(v_cndmask_b32_e64(v[21], 1.0, v[5], s[4]))
  k.emit(v_cndmask_b32_e64(v[22], 1.0, v[6], s[5]))
  k.emit(v_cndmask_b32_e64(v[23], 1.0, v[7], s[6]))


# ── C.5 — b128 VMEM-store VALU-forwarding variants ──────────────────────────
# Target: where [18] global_store_b128 forwarding pattern.

@microbench(name="mb_c5_b128_store_4mov_seed", category="C.5-vmem-b128")
def _c5_b128_store_4mov_seed(k: Kernel) -> None:
  """4× v_mov writing consecutive VGPRs then store 128b — standard pattern."""
  for i in range(4):
    k.emit(v_mov_b32_e32(v[10 + i], float(i + 1)))
  k.emit(global_store_b128(v[0], v[0], v[10:13], s[0:1]))

@microbench(name="mb_c5_b128_store_vopd_seed", category="C.5-vmem-b128")
def _c5_b128_store_vopd_seed(k: Kernel) -> None:
  """2× VOPD writing 4 consecutive VGPRs then store 128b."""
  k.emit(VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_MOV_B32, v[10], v[11], 0, 1.0))
  k.emit(VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_MOV_B32, v[12], v[13], 0, 1.0))
  k.emit(global_store_b128(v[0], v[0], v[10:13], s[0:1]))

@microbench(name="mb_c5_b128_store_interleave", category="C.5-vmem-b128")
def _c5_b128_store_interleave(k: Kernel) -> None:
  """Interleaved VALU writes: v[10], v[12], v[11], v[13] — tests gather-order forwarding."""
  k.emit(v_mov_b32_e32(v[10], 1.0))
  k.emit(v_mov_b32_e32(v[12], 2.0))
  k.emit(v_mov_b32_e32(v[11], 3.0))
  k.emit(v_mov_b32_e32(v[13], 4.0))
  k.emit(global_store_b128(v[0], v[0], v[10:13], s[0:1]))

@microbench(name="mb_c5_b128_store_after_cndmask", category="C.5-vmem-b128")
def _c5_b128_store_after_cndmask(k: Kernel) -> None:
  """Matches where[8-18]: VOPD+cmp+cndmask chain then b128 store."""
  _seed_vgprs_8(k)
  _seed_sgprs(k)
  k.emit(VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_MOV_B32, v[10], v[11], s[4], s[5]))
  k.emit(VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_MOV_B32, v[12], v[13], s[6], s[7]))
  k.emit(v_cndmask_b32_e64(v[10], 1.0, v[4], s[4]))
  k.emit(v_cndmask_b32_e64(v[11], 1.0, v[5], s[5]))
  k.emit(v_cndmask_b32_e64(v[12], 1.0, v[6], s[6]))
  k.emit(v_cndmask_b32_e64(v[13], 1.0, v[7], s[7]))
  k.emit(global_store_b128(v[0], v[0], v[10:13], s[0:1]))
