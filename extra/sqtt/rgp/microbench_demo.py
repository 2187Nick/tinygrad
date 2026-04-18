#!/usr/bin/env python3
"""Demo microbenchmarks to validate the microbench framework end-to-end.

This file is intentionally tiny — just enough to prove the API works. The full
taxonomy (300-600 microbenches) is the taxonomy agent's job; they should add
new modules alongside this one or extend it.

Registers on import:
  - mb_sanity_n1 .. mb_sanity_n4 : N back-to-back v_add_f32_e32
  - mb_nop0_solo, mb_nop1_solo, mb_nop3_solo, mb_nop5_solo, mb_nop7_solo,
    mb_nop10_solo, mb_nop15_solo : solo s_nop(N) kernels
  - mb_vopd_indep_n4 : 4 independent VOPD (v_dual_mul_f32) in a row
"""
from tinygrad.renderer.amd.dsl import s, v, NULL  # noqa: F401
from tinygrad.runtime.autogen.amd.rdna3.ins import (
  v_add_f32_e32, s_nop, v_mov_b32_e32,
  VOPD, VOPDOp,
)

from extra.sqtt.rgp.microbench import microbench, sweep, Kernel

# ── Sanity: N back-to-back v_add_f32_e32 on v[1] ─────────────────────────────
# The prologue has warmed v[1] via global_load, so these are pure VALU chains
# with a RAW dep on v[1]. Use `sweep` to stamp out n=1..4.
sweep(
  "mb_sanity", range(1, 5),
  lambda n: (lambda k: [k.emit(v_add_f32_e32(v[1], 1.0, v[1])) for _ in range(n)]),
  category="single-inst",
)

# ── s_nop(N) solo ────────────────────────────────────────────────────────────
# A single s_nop sandwiched between a warm v[1] and the store epilogue.
# Different N values probe the s_nop(N) cost curve in isolation.
for _n in (0, 1, 3, 5, 7, 10, 15):
  # Closures over _n require a default-arg trick to avoid late-binding.
  def _body(k: "Kernel", _n=_n) -> None:
    k.emit(s_nop(_n))
    # Touch v[1] so the compiler/emulator doesn't elide the store path.
    k.emit(v_add_f32_e32(v[1], 1.0, v[1]))
  microbench(name=f"mb_nop{_n}_solo", category="s_nop")(_body)

# ── 4 independent VOPDs ──────────────────────────────────────────────────────
# Uses v_dual_mul_f32 x v_dual_mul_f32 with disjoint vdst_x / vdst_y targets, so
# each pair writes different VGPRs. The prologue leaves v[1] warm; we init
# v[4..11] first, then chain 4 VOPDs with no RAW overlap (each VOPD writes a
# fresh pair of VGPRs).
#
# VOPD constraints on RDNA3:
#   - vdst_x % 2 must equal 0, vdst_y % 2 must equal 1 (even/odd split)
#   - vdst_x % 4 != vdst_y % 4 (bank-group disjointness)
#   - src0x, src0y, vsrc1x, vsrc1y: bank-conflict rules
# We use v[4,6,8,10] for X (even, bank 0/2/0/2) and v[5,7,9,11] for Y
# (odd, bank 1/3/1/3). srcs come from v[12..19] to avoid hitting any dst.
@microbench(name="mb_vopd_indep_n4", category="vopd")
def _mb_vopd_indep_n4(k: "Kernel") -> None:
  # Seed sources — 8 VGPRs with independent values.
  for i in range(8): k.emit(v_mov_b32_e32(v[12+i], float(i+1)))
  # 4 back-to-back VOPD pairs, each writing a fresh (even,odd) target.
  pairs = [(4, 5), (6, 7), (8, 9), (10, 11)]
  for (x, y) in pairs:
    k.emit(VOPD(
      opx=VOPDOp.V_DUAL_MUL_F32, opy=VOPDOp.V_DUAL_MUL_F32,
      vdstx=v[x], srcx0=v[12], vsrcx1=v[14],
      vdsty=v[y], srcy0=v[13], vsrcy1=v[15],
    ))
  # Pull one of the results into v[1] so the store path uses it.
  k.emit(v_add_f32_e32(v[1], v[4], v[1]))
