#!/usr/bin/env python3
"""Batch A — Single-instruction VALU microbenchmarks (families A.1/A.2/A.3).

24 one-knob probes targeting VALU issue timing on RDNA3:
  A.1 — v_add_f32 RAW-chain sweep (5 kernels)
  A.2 — v_mul / v_fmac / VOPD pairings (9 kernels)
  A.3 — Transcendental v_exp/v_log/v_sqrt/v_rcp/v_rsq (10 kernels)

Each kernel uses the standard microbench prologue (s_load_b64 ->
s_waitcnt_lgkmcnt -> v_lshlrev_b32_e32 -> global_load_b32 -> s_waitcnt_vmcnt)
and epilogue (global_store_b32 -> s_endpgm) from extra.sqtt.rgp.microbench.

VOPD register selection:
  vdstx is even, vdsty is odd (VOPD bank-group parity requirement).
  srcx / srcy banks (addr % 4) are disjoint pairwise to avoid VGPR-bank
  conflicts. We reuse the _probe_vopd layout from test/amd/test_custom_kernel.py:
    vopd(x, y) writes v[x], v[y] and reads v[x+2], v[y+2], v[x+4], v[y+4].
"""
from __future__ import annotations

from tinygrad.renderer.amd.dsl import s, v, NULL  # noqa: F401  (re-exports)
from tinygrad.runtime.autogen.amd.rdna3.ins import (
  v_add_f32_e32, v_mul_f32_e32, v_fmac_f32_e32, v_mov_b32_e32,
  v_exp_f32_e32, v_log_f32_e32, v_sqrt_f32_e32, v_rcp_f32_e32, v_rsq_f32_e32,
  VOPD, VOPDOp,
)
import tinygrad.runtime.autogen.amd.rdna3.ins as r3  # noqa: F401

from extra.sqtt.rgp.microbench import microbench, sweep, Kernel  # noqa: F401


# ── A.1 — v_add_f32 back-to-back RAW on v[1] (5 kernels) ─────────────────────
# N back-to-back `v_add_f32 v[1], 1.0, v[1]`.  The prologue's global_load has
# already warmed v[1], so these are pure VALU chains with a RAW dep per step.
# Expected HW: dt=1 per instruction (burst-issue throughput = 1 cy).
sweep(
  "mb_valu_add", [1, 2, 4, 8, 16],
  lambda n: (lambda k: [k.emit(v_add_f32_e32(v[1], 1.0, v[1])) for _ in range(n)]),
  category="valu-add",
)


# ── A.2 — v_mul / v_fmac / VOPD family (9 kernels) ───────────────────────────

# A.2.1 — N back-to-back `v_mul_f32 v[1], v[1], v[2]`.  RAW on v[1].
@microbench(name="mb_valu_mul_n1", category="valu-mul")
def _mb_valu_mul_n1(k: "Kernel") -> None:
  k.emit(v_mul_f32_e32(v[1], v[1], v[2]))

@microbench(name="mb_valu_mul_n4", category="valu-mul")
def _mb_valu_mul_n4(k: "Kernel") -> None:
  for _ in range(4): k.emit(v_mul_f32_e32(v[1], v[1], v[2]))

# A.2.2 — N back-to-back `v_fmac_f32 v[1], v[2], v[3]`.  v_fmac reads+writes
# v[1] so this is a RAW chain on the accumulator.
@microbench(name="mb_valu_fmac_n1", category="valu-fmac")
def _mb_valu_fmac_n1(k: "Kernel") -> None:
  k.emit(v_fmac_f32_e32(v[1], v[2], v[3]))

@microbench(name="mb_valu_fmac_n4", category="valu-fmac")
def _mb_valu_fmac_n4(k: "Kernel") -> None:
  for _ in range(4): k.emit(v_fmac_f32_e32(v[1], v[2], v[3]))

@microbench(name="mb_valu_fmac_n8", category="valu-fmac")
def _mb_valu_fmac_n8(k: "Kernel") -> None:
  for _ in range(8): k.emit(v_fmac_f32_e32(v[1], v[2], v[3]))

# A.2.3 — VOPDs.
# Register layout chosen to satisfy RDNA3 VOPD bank rules:
#   vdstx=v[4] (bank 0, even), vdsty=v[5] (bank 1, odd)           -> dst parity ok,
#     dst banks {0, 1} disjoint modulo 4.
#   srcx0=v[6] (bank 2), srcy0=v[7] (bank 3)                      -> disjoint
#   vsrcx1=v[8] (bank 0), vsrcy1=v[9] (bank 1)                    -> disjoint
# This is the same layout used by _probe_vopd() in test/amd/test_custom_kernel.py.

def _seed_vopd_sources(k: "Kernel") -> None:
  """Init v[4..11] to independent non-zero values so VOPDs have real inputs."""
  for i in range(8): k.emit(v_mov_b32_e32(v[4 + i], float(i + 1)))

def _vopd(opx, opy, x=4, y=5):
  """Emit a VOPD writing v[x], v[y] reading v[x+2], v[y+2], v[x+4], v[y+4]."""
  return VOPD(
    opx=opx, opy=opy,
    vdstx=v[x],   srcx0=v[x + 2], vsrcx1=v[x + 4],
    vdsty=v[y],   srcy0=v[y + 2], vsrcy1=v[y + 4],
  )

# 2 VOPD v_dual_fmac_f32 + v_dual_mul_f32 (same pair, no RAW across pairs):
# diagnoses C6/C12 (VOPD back-to-back spacing).
@microbench(name="mb_vopd_fmac_mul_n2", category="vopd")
def _mb_vopd_fmac_mul_n2(k: "Kernel") -> None:
  _seed_vopd_sources(k)
  for _ in range(2):
    k.emit(_vopd(VOPDOp.V_DUAL_FMAC_F32, VOPDOp.V_DUAL_MUL_F32))

# 4 VOPDs of the same pairing — exposes sustained VOPD throughput.
@microbench(name="mb_vopd_fmac_mul_n4", category="vopd")
def _mb_vopd_fmac_mul_n4(k: "Kernel") -> None:
  _seed_vopd_sources(k)
  for _ in range(4):
    k.emit(_vopd(VOPDOp.V_DUAL_FMAC_F32, VOPDOp.V_DUAL_MUL_F32))

# 2 VOPD v_dual_cndmask_b32 pairs.  cndmask implicitly reads VCC — prologue
# hasn't written VCC, but the encoding is still well-formed and the emulator
# will simulate cndmask timing from the issue-slot perspective, which is what
# we're measuring here.
@microbench(name="mb_vopd_cndmask_n2", category="vopd")
def _mb_vopd_cndmask_n2(k: "Kernel") -> None:
  _seed_vopd_sources(k)
  for _ in range(2):
    k.emit(_vopd(VOPDOp.V_DUAL_CNDMASK_B32, VOPDOp.V_DUAL_CNDMASK_B32))

# 4 VOPDs alternating add_mul / mul_sub.  Measures whether the scheduler
# treats different opcode pairings identically for back-to-back issue (C6/C12).
@microbench(name="mb_vopd_mixed_n4", category="vopd")
def _mb_vopd_mixed_n4(k: "Kernel") -> None:
  _seed_vopd_sources(k)
  pairs = [
    (VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_MUL_F32),
    (VOPDOp.V_DUAL_MUL_F32, VOPDOp.V_DUAL_SUB_F32),
    (VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_MUL_F32),
    (VOPDOp.V_DUAL_MUL_F32, VOPDOp.V_DUAL_SUB_F32),
  ]
  for (ox, oy) in pairs:
    k.emit(_vopd(ox, oy))


# ── A.3 — Transcendental (10 kernels) ────────────────────────────────────────
# v_exp/v_log/v_sqrt/v_rcp/v_rsq all go through the trans pipe.  A single
# isolated trans op issues at dt=1 relative to its predecessor (VALU issue cost),
# but a RAW consumer after one trans op stalls ~4 cy (trans latency).

@microbench(name="mb_trans_exp_n1", category="trans")
def _mb_trans_exp_n1(k: "Kernel") -> None:
  k.emit(v_exp_f32_e32(v[1], v[1]))

@microbench(name="mb_trans_log_n1", category="trans")
def _mb_trans_log_n1(k: "Kernel") -> None:
  k.emit(v_log_f32_e32(v[1], v[1]))

@microbench(name="mb_trans_sqrt_n1", category="trans")
def _mb_trans_sqrt_n1(k: "Kernel") -> None:
  k.emit(v_sqrt_f32_e32(v[1], v[1]))

@microbench(name="mb_trans_rcp_n1", category="trans")
def _mb_trans_rcp_n1(k: "Kernel") -> None:
  k.emit(v_rcp_f32_e32(v[1], v[1]))

@microbench(name="mb_trans_rsq_n1", category="trans")
def _mb_trans_rsq_n1(k: "Kernel") -> None:
  k.emit(v_rsq_f32_e32(v[1], v[1]))

# 4 back-to-back v_exp with RAW on v[1].  Expect dt=1 for the first, then ~4
# each for the RAW-dependent successors (trans-pipe occupancy).
@microbench(name="mb_trans_exp_n4", category="trans")
def _mb_trans_exp_n4(k: "Kernel") -> None:
  for _ in range(4): k.emit(v_exp_f32_e32(v[1], v[1]))

# 4 back-to-back v_log with RAW on v[1].
@microbench(name="mb_trans_log_n4", category="trans")
def _mb_trans_log_n4(k: "Kernel") -> None:
  for _ in range(4): k.emit(v_log_f32_e32(v[1], v[1]))

# v_exp -> v_log RAW on v[1].  Diagnoses trans-to-trans RAW pipeline cost.
@microbench(name="mb_trans_mixed_exp_log", category="trans")
def _mb_trans_mixed_exp_log(k: "Kernel") -> None:
  k.emit(v_exp_f32_e32(v[1], v[1]))
  k.emit(v_log_f32_e32(v[1], v[1]))

# v_exp v[1] -> v_add v[2] -> v_exp v[1].  The middle v_add has no trans dep,
# so the two trans ops should overlap with it in the trans+VALU pipelines.
@microbench(name="mb_trans_exp_valu_exp", category="trans")
def _mb_trans_exp_valu_exp(k: "Kernel") -> None:
  k.emit(v_exp_f32_e32(v[1], v[1]))
  k.emit(v_add_f32_e32(v[2], 1.0, v[2]))
  k.emit(v_exp_f32_e32(v[1], v[1]))

# v_add; v_exp; v_add; v_add; v_exp — cold vs warm trans pipeline (C8).
# The first v_exp hits the cold trans pipe; the second one has a shorter gap.
@microbench(name="mb_trans_cold_vs_warm", category="trans")
def _mb_trans_cold_vs_warm(k: "Kernel") -> None:
  k.emit(v_add_f32_e32(v[2], 1.0, v[2]))
  k.emit(v_exp_f32_e32(v[3], v[3]))
  k.emit(v_add_f32_e32(v[4], 1.0, v[4]))
  k.emit(v_add_f32_e32(v[5], 1.0, v[5]))
  k.emit(v_exp_f32_e32(v[6], v[6]))


__all__: list[str] = []  # registration happens as a side effect of import
