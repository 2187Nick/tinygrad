#!/usr/bin/env python3
"""Batch A microbenchmarks, families A.4, A.5, A.6.

Register 26 one-knob microbenches covering v_cmp (A.4 — 8 kernels),
v_cndmask (A.5 — 8 kernels), and SALU opcodes (A.6 — 10 kernels). Each
kernel isolates one HW timing behavior; see MICROBENCH_TAXONOMY.md for the
expected dt-sequences per kernel.

Conventions:
  - Standard prologue emits: s_load_b64 -> waitcnt -> v_lshlrev(v[0], 2)
    -> global_load(v[1]) -> waitcnt_vmcnt. On entry, s[0:1] holds the
    kernel-arg pointer and v[0]/v[1] are live.
  - Standard epilogue emits: global_store_b32 v[1] -> s_endpgm.
  - We use s[4:] for scratch SGPRs so we never clobber s[0:1].
  - Before any v_cndmask_b32_e64 that reads s[k], a v_cmp_gt_f32_e64 MUST
    have written s[k], otherwise the timing reflects stale state rather
    than the pattern we're measuring.
"""
from __future__ import annotations

from tinygrad.renderer.amd.dsl import s, v, NULL  # noqa: F401
from tinygrad.runtime.autogen.amd.rdna3.ins import (
  v_mov_b32_e32, v_add_f32_e32,
  v_cmp_gt_f32_e32, v_cmp_gt_f32_e64,
  v_cndmask_b32_e32, v_cndmask_b32_e64,
  s_mov_b32, s_cmp_eq_i32, s_cbranch_scc0, s_cbranch_scc1,
  s_bitcmp1_b32, s_and_b32, s_nop,
)

from extra.sqtt.rgp.microbench import microbench, sweep, Kernel


# =============================================================================
# A.4 — v_cmp family (8 kernels)
# =============================================================================
# `v_cmp_gt_f32_e32(lit, v[n])` writes VCC. `v_cmp_gt_f32_e64(s[k], lit, v[n])`
# writes s[k] (SGPR). Per taxonomy: n1/n4/n8 back-to-back, mixed VCC/SGPR,
# literal variants, and reading from distinct VGPRs.

def _seed_vgprs(k: "Kernel", idxs):
  """v_mov_b32_e32 a float constant into each v[i] so the v_cmp sources are
  valid. Uses 1.0, 2.0, ... to keep them distinct."""
  for n, i in enumerate(idxs):
    k.emit(v_mov_b32_e32(v[i], float(n + 1)))


@microbench(name="mb_vcmp_vcc_n1", category="vcmp")
def _mb_vcmp_vcc_n1(k: "Kernel") -> None:
  # v[1] is already warm from prologue. 1 v_cmp writing VCC.
  k.emit(v_cmp_gt_f32_e32(0.5, v[1]))


@microbench(name="mb_vcmp_vcc_n4", category="vcmp")
def _mb_vcmp_vcc_n4(k: "Kernel") -> None:
  # 4 v_cmp_e32 writing VCC back-to-back — same source VGPRs ok, VCC is the
  # destination under test.
  _seed_vgprs(k, [2, 3, 4, 5])
  for i in (2, 3, 4, 5):
    k.emit(v_cmp_gt_f32_e32(0.5, v[i]))


# A.4 sweep over SGPR-destination chain depth: n=1, 4, 8.
def _mb_vcmp_sgpr_body(n):
  def _body(k: "Kernel") -> None:
    # seed n distinct VGPRs, then write s[4..4+n-1]
    _seed_vgprs(k, list(range(2, 2 + n)))
    for i in range(n):
      k.emit(v_cmp_gt_f32_e64(s[4 + i], 0.5, v[2 + i]))
  return _body

sweep("mb_vcmp_sgpr", [1, 4, 8], _mb_vcmp_sgpr_body, category="vcmp")


@microbench(name="mb_vcmp_vcc_then_sgpr", category="vcmp")
def _mb_vcmp_vcc_then_sgpr(k: "Kernel") -> None:
  # VCC-writing e32 then SGPR-writing e64 -> measures mixed SGPR/VCC port.
  _seed_vgprs(k, [2, 3])
  k.emit(v_cmp_gt_f32_e32(0.5, v[2]))          # writes VCC
  k.emit(v_cmp_gt_f32_e64(s[4], 0.5, v[3]))    # writes s[4]


@microbench(name="mb_vcmp_literal", category="vcmp")
def _mb_vcmp_literal(k: "Kernel") -> None:
  # v_cmp_gt_f32_e32 v[0], 1.0 — LIT-encoded literal. v[0] is live from prologue
  # (it holds the element offset; its bit pattern is a valid f32 sample).
  k.emit(v_cmp_gt_f32_e32(1.0, v[0]))


@microbench(name="mb_vcmp_chain_different_regs", category="vcmp")
def _mb_vcmp_chain_different_regs(k: "Kernel") -> None:
  # 4 v_cmps, one per distinct VGPR — isolates the per-source-VGPR path from
  # the SGPR-destination chain depth effect.
  _seed_vgprs(k, [2, 3, 4, 5])
  for i in (2, 3, 4, 5):
    k.emit(v_cmp_gt_f32_e32(0.5, v[i]))


# =============================================================================
# A.5 — v_cndmask family (8 kernels)
# =============================================================================
# `v_cndmask_b32_e32(v[d], lit, v[s])` reads VCC. `v_cndmask_b32_e64(v[d],
# lit, v[s], s[k])` reads s[k]. ANY cndmask that reads s[k] must be preceded
# by a v_cmp_gt_f32_e64 writing s[k], else the HW timing reflects a cold
# SGPR (which is a different test).

@microbench(name="mb_cndmask_vcc_n1", category="cndmask")
def _mb_cndmask_vcc_n1(k: "Kernel") -> None:
  # prime VCC, then 1 cndmask consuming it
  _seed_vgprs(k, [2, 3])
  k.emit(v_cmp_gt_f32_e32(0.5, v[2]))                    # writes VCC
  k.emit(v_cndmask_b32_e32(v[4], 0.0, v[3]))             # reads VCC


@microbench(name="mb_cndmask_vcc_n4", category="cndmask")
def _mb_cndmask_vcc_n4(k: "Kernel") -> None:
  # 4 cndmasks reading VCC, distinct dst VGPRs
  _seed_vgprs(k, [2, 3])
  k.emit(v_cmp_gt_f32_e32(0.5, v[2]))                    # writes VCC
  for d in (4, 5, 6, 7):
    k.emit(v_cndmask_b32_e32(v[d], 0.0, v[3]))           # reads VCC


@microbench(name="mb_cndmask_sgpr_n1", category="cndmask")
def _mb_cndmask_sgpr_n1(k: "Kernel") -> None:
  # one cndmask reading s[4]; write s[4] first with v_cmp_e64
  _seed_vgprs(k, [2, 3])
  k.emit(v_cmp_gt_f32_e64(s[4], 0.5, v[2]))              # writes s[4]
  k.emit(v_cndmask_b32_e64(v[4], 0.0, v[3], s[4]))       # reads s[4]


@microbench(name="mb_cndmask_sgpr_fresh_n4", category="cndmask")
def _mb_cndmask_sgpr_fresh_n4(k: "Kernel") -> None:
  # 4 cndmasks, each reading a different, freshly-written SGPR. Taxonomy
  # expects dt=2,1,1,1 (first-cold SGPR slip).
  _seed_vgprs(k, [2, 3, 4, 5])
  for i in range(4):
    k.emit(v_cmp_gt_f32_e64(s[4 + i], 0.5, v[2 + i]))    # write s[4..7]
  for i in range(4):
    k.emit(v_cndmask_b32_e64(v[6 + i], 0.0, v[2 + i], s[4 + i]))


@microbench(name="mb_cndmask_sgpr_stale_n4", category="cndmask")
def _mb_cndmask_sgpr_stale_n4(k: "Kernel") -> None:
  # Same as fresh_n4 but insert ~40 cycles of s_nop between the v_cmp chain
  # and the cndmasks. RDNA3 s_nop(N) = N+1 cycles, so 3x s_nop(15) = 48 cy
  # exceeds the taxonomy's 40-cycle target. Expectation: dt=1,1,1,1 (no
  # first-cold-SGPR slip because the SGPRs are no longer "fresh").
  _seed_vgprs(k, [2, 3, 4, 5])
  for i in range(4):
    k.emit(v_cmp_gt_f32_e64(s[4 + i], 0.5, v[2 + i]))
  k.emit(s_nop(15))
  k.emit(s_nop(15))
  k.emit(s_nop(15))
  for i in range(4):
    k.emit(v_cndmask_b32_e64(v[6 + i], 0.0, v[2 + i], s[4 + i]))


@microbench(name="mb_cndmask_sgpr_followed_by_vcmp", category="cndmask")
def _mb_cndmask_sgpr_followed_by_vcmp(k: "Kernel") -> None:
  # 4 fresh-SGPR cndmasks, then a vcmp — probes the F2-tail-slip effect
  # where the vcmp that follows a cndmask chain picks up extra latency.
  _seed_vgprs(k, [2, 3, 4, 5])
  for i in range(4):
    k.emit(v_cmp_gt_f32_e64(s[4 + i], 0.5, v[2 + i]))
  for i in range(4):
    k.emit(v_cndmask_b32_e64(v[6 + i], 0.0, v[2 + i], s[4 + i]))
  # Tail vcmp writing a new SGPR (s[8]) — distinct from s[4..7] so there's
  # no RAW/WAW with the cndmask chain.
  k.emit(v_cmp_gt_f32_e64(s[8], 0.5, v[2]))


@microbench(name="mb_cndmask_tail_gt4", category="cndmask")
def _mb_cndmask_tail_gt4(k: "Kernel") -> None:
  # 5 fresh-SGPR cndmasks on s[4..8] — probes the F1/F3 tail-slip (the 5th
  # cndmask picks up +1 cy after the deep-queue retire).
  _seed_vgprs(k, [2, 3, 4, 5, 6])
  for i in range(5):
    k.emit(v_cmp_gt_f32_e64(s[4 + i], 0.5, v[2 + i]))
  for i in range(5):
    k.emit(v_cndmask_b32_e64(v[7 + i], 0.0, v[2 + i], s[4 + i]))


@microbench(name="mb_cndmask_dst_fresh_vgpr_high", category="cndmask")
def _mb_cndmask_dst_fresh_vgpr_high(k: "Kernel") -> None:
  # 6 cndmasks targeting v[10..15] (never previously touched) to test
  # the F2 fresh-VGPR hypothesis. Use a VCC source so we don't muddle the
  # test with per-SGPR fresh state; 1 v_cmp primes VCC once.
  _seed_vgprs(k, [2])
  k.emit(v_cmp_gt_f32_e32(0.5, v[2]))                    # writes VCC
  for d in range(10, 16):
    k.emit(v_cndmask_b32_e32(v[d], 0.0, v[2]))           # reads VCC


# =============================================================================
# A.6 — SALU opcodes (10 kernels)
# =============================================================================
# SALU timing is qualitatively different from VALU. `s_mov`, `s_cmp`, and
# the SCC-based branch family show cycle-level beat/phase interactions
# that the E1/E2/E3 mismatches call out.

@microbench(name="mb_salu_smov_n1", category="salu")
def _mb_salu_smov_n1(k: "Kernel") -> None:
  k.emit(s_mov_b32(s[4], 0))


@microbench(name="mb_salu_smov_n4", category="salu")
def _mb_salu_smov_n4(k: "Kernel") -> None:
  for i in range(4):
    k.emit(s_mov_b32(s[4 + i], i))


@microbench(name="mb_salu_smov_followed_by_nop0", category="salu")
def _mb_salu_smov_followed_by_nop0(k: "Kernel") -> None:
  # `s_mov; s_nop(0)` — the taxonomy's E-family scalar-warmup probe.
  k.emit(s_mov_b32(s[4], 0))
  k.emit(s_nop(0))


@microbench(name="mb_salu_scmp_tight", category="salu")
def _mb_salu_scmp_tight(k: "Kernel") -> None:
  # s_mov; s_cmp; s_cbranch (tight) — E1 fast-path. SCC=0 after s_cmp
  # because s[4]=0 and we compare with 1, so scc1 branch is not taken.
  k.emit(s_mov_b32(s[4], 0))
  k.emit(s_cmp_eq_i32(s[4], 1))                # SCC = (0 == 1) = 0
  k.emit(s_cbranch_scc1(), target="salu_end")  # not taken
  k.label("salu_end")


@microbench(name="mb_salu_scmp_spaced_nop0", category="salu")
def _mb_salu_scmp_spaced_nop0(k: "Kernel") -> None:
  # s_mov; s_nop(0); s_cmp; s_cbranch — E1 slow path.
  k.emit(s_mov_b32(s[4], 0))
  k.emit(s_nop(0))
  k.emit(s_cmp_eq_i32(s[4], 1))
  k.emit(s_cbranch_scc1(), target="salu_end")
  k.label("salu_end")


@microbench(name="mb_salu_scmp_spaced_nop0x2", category="salu")
def _mb_salu_scmp_spaced_nop0x2(k: "Kernel") -> None:
  k.emit(s_mov_b32(s[4], 0))
  for _ in range(2):
    k.emit(s_nop(0))
  k.emit(s_cmp_eq_i32(s[4], 1))
  k.emit(s_cbranch_scc1(), target="salu_end")
  k.label("salu_end")


@microbench(name="mb_salu_scmp_spaced_nop0x3", category="salu")
def _mb_salu_scmp_spaced_nop0x3(k: "Kernel") -> None:
  k.emit(s_mov_b32(s[4], 0))
  for _ in range(3):
    k.emit(s_nop(0))
  k.emit(s_cmp_eq_i32(s[4], 1))
  k.emit(s_cbranch_scc1(), target="salu_end")
  k.label("salu_end")


@microbench(name="mb_salu_scbranch_taken", category="salu")
def _mb_salu_scbranch_taken(k: "Kernel") -> None:
  # Force SCC=1: s[4]=0; cmp(s[4] == 0) -> SCC=1; scc1 branch is taken.
  # The cbranch target skips nothing (the label is right after) so the
  # probe measures only the *taken-branch* latency, not the skipped body.
  k.emit(s_mov_b32(s[4], 0))
  k.emit(s_cmp_eq_i32(s[4], 0))                # SCC = (0 == 0) = 1
  k.emit(s_cbranch_scc1(), target="salu_end")  # taken
  k.label("salu_end")


@microbench(name="mb_salu_sbitcmp_branch", category="salu")
def _mb_salu_sbitcmp_branch(k: "Kernel") -> None:
  # s_bitcmp1_b32: SCC = bit(ssrc0, ssrc1). Seed s[4]=1, bit 0 is set, so
  # s_bitcmp1_b32(s[4], 0) sets SCC=1 -> scc1 branch taken.
  k.emit(s_mov_b32(s[4], 1))
  k.emit(s_bitcmp1_b32(s[4], 0))
  k.emit(s_cbranch_scc1(), target="salu_end")
  k.label("salu_end")


@microbench(name="mb_salu_sand_scmp_branch", category="salu")
def _mb_salu_sand_scmp_branch(k: "Kernel") -> None:
  # s_and_b32 writes SCC = (result != 0). s[4]=3, s[5]=1 -> s[6]=1, SCC=1,
  # then branch on scc1 taken.
  k.emit(s_mov_b32(s[4], 3))
  k.emit(s_mov_b32(s[5], 1))
  k.emit(s_and_b32(s[6], s[4], s[5]))          # s[6]=1, SCC=1
  k.emit(s_cbranch_scc1(), target="salu_end")
  k.label("salu_end")
