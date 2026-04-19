#!/usr/bin/env python3
"""Batch D — VOPD→non-VOPD transition tail microbenches.

18 surgical probes to characterize what the pipeline does AFTER a VOPD (or
VOPD chain) ends and a non-VOPD instruction appears. The emu currently
treats VOPD pipe_avail as a single-state gate — but HW behavior depends
on BOTH the producer (VOPD variant) AND the consumer (what type of inst
follows).

Context: Batch C revealed that VOPDs chain at 1cy, but attempts to land
this uniformly regressed exp_chain [51]→[52] VOPD_LIT→v_cmp (HW=4cy)
vs exp_chain [28]→[29] VOPD→v_cmp (HW=1cy). The structural context
matters. This batch isolates what makes transitions 1cy vs 4cy.

Categories:
  D.1 (8 kernels) — Single VOPD then various non-VOPD consumer types.
  D.2 (4 kernels) — Single VOPD_LIT then various non-VOPD consumers.
  D.3 (6 kernels) — Chained VOPDs (2× or 4×) then non-VOPD consumer.

Naming: mb_d{N}_{producer}_then_{consumer}
"""
from __future__ import annotations
import struct

from tinygrad.renderer.amd.dsl import s, v, NULL, VCC_LO  # noqa: F401
from tinygrad.runtime.autogen.amd.rdna3.ins import (
  v_add_f32_e32, v_mul_f32_e32, v_mov_b32_e32,
  v_cmp_gt_f32_e32, v_cmp_gt_f32_e64, v_cndmask_b32_e32, v_cndmask_b32_e64,
  s_mov_b32, s_nop, s_waitcnt,
  VOPD, VOPD_LIT, VOPDOp,
)

from extra.sqtt.rgp.microbench import microbench, Kernel  # noqa: F401


def _f2i(f: float) -> int:
  return struct.unpack('I', struct.pack('f', f))[0]


def _seed(k: "Kernel") -> None:
  """Seed v[4..11] with non-zero floats. Matches earlier batches."""
  for i in range(8): k.emit(v_mov_b32_e32(v[4 + i], float(i + 1)))


def _vopd_add(k: "Kernel", dstx: int, dsty: int) -> None:
  """Emit a VOPD V_DUAL_ADD_F32 writing v[dstx]/v[dsty], reading v[4..7]."""
  k.emit(VOPD(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_ADD_F32,
              v[dstx], v[dsty], v[4], v[5], v[6], v[7]))


def _vopd_lit(k: "Kernel", dstx: int, dsty: int) -> None:
  """Emit a VOPD_LIT FMAAK+MOV pair."""
  k.emit(VOPD_LIT(
    opx=VOPDOp.V_DUAL_FMAAK_F32, opy=VOPDOp.V_DUAL_MOV_B32,
    vdstx=v[dstx], srcx0=v[6], vsrcx1=v[8],
    vdsty=v[dsty], srcy0=v[7], vsrcy1=v[9],
    literal=_f2i(2.0),
  ))


# ── D.1 — VOPD then non-VOPD (8 kernels) ─────────────────────────────────────

@microbench(name="mb_d1_vopd_then_vcmp_e32", category="D.1-vopd-tail")
def _d1_vopd_then_vcmp_e32(k: Kernel) -> None:
  """VOPD → v_cmp (e32, writes VCC). HW gap?"""
  _seed(k)
  _vopd_add(k, 20, 21)
  k.emit(v_cmp_gt_f32_e32(0.5, v[4]))

@microbench(name="mb_d1_vopd_then_vcmp_e64", category="D.1-vopd-tail")
def _d1_vopd_then_vcmp_e64(k: Kernel) -> None:
  """VOPD → v_cmp (e64, writes explicit SGPR)."""
  _seed(k)
  _vopd_add(k, 20, 21)
  k.emit(v_cmp_gt_f32_e64(s[4], 0.5, v[4]))

@microbench(name="mb_d1_vopd_then_cndmask_vcc", category="D.1-vopd-tail")
def _d1_vopd_then_cndmask_vcc(k: Kernel) -> None:
  """VOPD → v_cndmask (reads VCC). VCC was set in prologue by initial v_cmp."""
  _seed(k)
  k.emit(v_cmp_gt_f32_e32(0.5, v[4]))   # VCC <- v[4] > 0.5
  _vopd_add(k, 20, 21)
  k.emit(v_cndmask_b32_e32(v[22], 1.0, v[4]))

@microbench(name="mb_d1_vopd_then_vmov", category="D.1-vopd-tail")
def _d1_vopd_then_vmov(k: Kernel) -> None:
  """VOPD → v_mov (literal)."""
  _seed(k)
  _vopd_add(k, 20, 21)
  k.emit(v_mov_b32_e32(v[22], 3.0))

@microbench(name="mb_d1_vopd_then_vadd", category="D.1-vopd-tail")
def _d1_vopd_then_vadd(k: Kernel) -> None:
  """VOPD → plain v_add."""
  _seed(k)
  _vopd_add(k, 20, 21)
  k.emit(v_add_f32_e32(v[22], v[4], v[5]))

@microbench(name="mb_d1_vopd_then_snop", category="D.1-vopd-tail")
def _d1_vopd_then_snop(k: Kernel) -> None:
  """VOPD → s_nop(3) (IB drain). Tests VOPD tail vs SALU drain."""
  _seed(k)
  _vopd_add(k, 20, 21)
  k.emit(s_nop(3))

@microbench(name="mb_d1_vopd_then_salu", category="D.1-vopd-tail")
def _d1_vopd_then_salu(k: Kernel) -> None:
  """VOPD → SALU (s_mov)."""
  _seed(k)
  _vopd_add(k, 20, 21)
  k.emit(s_mov_b32(s[5], 0))

@microbench(name="mb_d1_vopd_then_waitcnt", category="D.1-vopd-tail")
def _d1_vopd_then_waitcnt(k: Kernel) -> None:
  """VOPD → s_waitcnt()."""
  _seed(k)
  _vopd_add(k, 20, 21)
  k.emit(s_waitcnt(0))


# ── D.2 — VOPD_LIT then non-VOPD (4 kernels) ────────────────────────────────

@microbench(name="mb_d2_vopd_lit_then_vcmp_e32", category="D.2-vopd-lit-tail")
def _d2_vopd_lit_then_vcmp_e32(k: Kernel) -> None:
  """VOPD_LIT → v_cmp (e32). Mirrors exp_chain [51]→[52] pattern."""
  _seed(k)
  _vopd_lit(k, 20, 21)
  k.emit(v_cmp_gt_f32_e32(0.5, v[4]))

@microbench(name="mb_d2_vopd_lit_then_vcmp_e64", category="D.2-vopd-lit-tail")
def _d2_vopd_lit_then_vcmp_e64(k: Kernel) -> None:
  """VOPD_LIT → v_cmp (e64)."""
  _seed(k)
  _vopd_lit(k, 20, 21)
  k.emit(v_cmp_gt_f32_e64(s[4], 0.5, v[4]))

@microbench(name="mb_d2_vopd_lit_then_vmov", category="D.2-vopd-lit-tail")
def _d2_vopd_lit_then_vmov(k: Kernel) -> None:
  """VOPD_LIT → v_mov."""
  _seed(k)
  _vopd_lit(k, 20, 21)
  k.emit(v_mov_b32_e32(v[22], 3.0))

@microbench(name="mb_d2_vopd_lit_then_vadd", category="D.2-vopd-lit-tail")
def _d2_vopd_lit_then_vadd(k: Kernel) -> None:
  """VOPD_LIT → plain v_add."""
  _seed(k)
  _vopd_lit(k, 20, 21)
  k.emit(v_add_f32_e32(v[22], v[4], v[5]))


# ── D.3 — Chained VOPDs then non-VOPD (6 kernels) ────────────────────────────

@microbench(name="mb_d3_vopd_chain2_then_vcmp", category="D.3-chain-tail")
def _d3_vopd_chain2_then_vcmp(k: Kernel) -> None:
  """2× VOPD → v_cmp. Does chain depth affect the tail?"""
  _seed(k)
  _vopd_add(k, 20, 21)
  _vopd_add(k, 22, 23)
  k.emit(v_cmp_gt_f32_e32(0.5, v[4]))

@microbench(name="mb_d3_vopd_chain4_then_vcmp", category="D.3-chain-tail")
def _d3_vopd_chain4_then_vcmp(k: Kernel) -> None:
  """4× VOPD → v_cmp. Tests longer chain tail."""
  _seed(k)
  _vopd_add(k, 20, 21)
  _vopd_add(k, 22, 23)
  _vopd_add(k, 24, 25)
  _vopd_add(k, 26, 27)
  k.emit(v_cmp_gt_f32_e32(0.5, v[4]))

@microbench(name="mb_d3_vopd_lit_chain2_then_vcmp", category="D.3-chain-tail")
def _d3_vopd_lit_chain2_then_vcmp(k: Kernel) -> None:
  """2× VOPD_LIT → v_cmp. Most direct match for exp_chain [48-52]."""
  _seed(k)
  _vopd_lit(k, 20, 21)
  _vopd_lit(k, 22, 23)
  k.emit(v_cmp_gt_f32_e32(0.5, v[4]))

@microbench(name="mb_d3_vopd_lit_chain4_then_vcmp", category="D.3-chain-tail")
def _d3_vopd_lit_chain4_then_vcmp(k: Kernel) -> None:
  """4× VOPD_LIT → v_cmp."""
  _seed(k)
  _vopd_lit(k, 20, 21)
  _vopd_lit(k, 22, 23)
  _vopd_lit(k, 24, 25)
  _vopd_lit(k, 26, 27)
  k.emit(v_cmp_gt_f32_e32(0.5, v[4]))

@microbench(name="mb_d3_mixed_chain_then_vcmp", category="D.3-chain-tail")
def _d3_mixed_chain_then_vcmp(k: Kernel) -> None:
  """VOPD, VOPD_LIT, VOPD, VOPD_LIT → v_cmp. Alternating variants."""
  _seed(k)
  _vopd_add(k, 20, 21)
  _vopd_lit(k, 22, 23)
  _vopd_add(k, 24, 25)
  _vopd_lit(k, 26, 27)
  k.emit(v_cmp_gt_f32_e32(0.5, v[4]))

@microbench(name="mb_d3_vopd_chain2_then_vmov", category="D.3-chain-tail")
def _d3_vopd_chain2_then_vmov(k: Kernel) -> None:
  """2× VOPD → v_mov. Consumer is a simple v_mov, not v_cmp."""
  _seed(k)
  _vopd_add(k, 20, 21)
  _vopd_add(k, 22, 23)
  k.emit(v_mov_b32_e32(v[24], 3.0))
