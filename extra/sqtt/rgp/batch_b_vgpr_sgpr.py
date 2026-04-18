#!/usr/bin/env python3
"""Batch B microbenchmarks, families B.2 and B.4.

Register 39 one-knob microbenches covering VGPR bank-port conflicts (B.2 - 24
kernels) and v_cmp / v_cndmask SGPR-forwarding behavior (B.4 - 15 kernels).
Each kernel isolates one HW timing behavior; see MICROBENCH_TAXONOMY.md
(sections B.2 and B.4) for the expected dt-sequences per kernel.

Conventions (match batch_a_cmp_salu.py):
  - Standard prologue emits: s_load_b64 -> waitcnt -> v_lshlrev(v[0], 2)
    -> global_load(v[1]) -> waitcnt_vmcnt. On entry, s[0:1] holds the
    kernel-arg pointer, v[0] holds the per-thread byte offset (MUST NOT be
    clobbered — the epilogue's global_store uses it as addr), and v[1]
    holds the loaded value (MUST NOT be clobbered — the epilogue stores it).
  - Standard epilogue emits: global_store_b32 v[1] -> s_endpgm.
  - Scratch SGPRs start at s[4] so we never clobber s[0:1].
  - Scratch VGPRs live in v[4..31]. Bank ids repeat mod 4:
      bank 0: v[4],  v[8],  v[12], ...
      bank 1: v[5],  v[9],  v[13], ...
      bank 2: v[6],  v[10], v[14], ...
      bank 3: v[7],  v[11], v[15], ...
    so `_BANK_REG[b]` yields a freshly-seedable representative for bank b.
  - Before any v_cndmask_b32_e64 that reads s[k], a v_cmp_gt_f32_e64 MUST
    have written s[k], otherwise the timing reflects stale state rather
    than the pattern we're measuring.
  - VGPRs used as v_cmp / VALU sources are seeded with v_mov_b32_e32 so
    each probe has valid, distinct inputs.

B.2 notes:
  - RDNA3 has 4 VGPR read banks: bank(v[n]) = n % 4.
  - Each probe walks a specific (src_bank, dst_bank, src-count) cell.
  - v_fma_f32 is VOP3 (3-src) called positionally: v_fma_f32(dst, s0, s1, s2).
  - v_fmac_f32_e32 is VOP2 with implicit accumulator: it reads *and* writes
    dst, so v_fmac v[D],v[s0],v[s1] reads v[s0], v[s1], and v[D].

B.4 notes:
  - Cold SGPRs cost one extra cycle on the first cndmask read, and the
    deep-queue retire adds one extra cycle to the 5th+ cndmask (F1/F3).
  - s_nop(N) costs N+1 cycles; we sweep nop spacing to draw the
    warm -> full-drain boundary for the v_cmp -> v_cndmask forwarding path.
"""
from __future__ import annotations

from tinygrad.renderer.amd.dsl import s, v, NULL  # noqa: F401
from tinygrad.runtime.autogen.amd.rdna3.ins import (
  v_mov_b32_e32, v_add_f32_e32, v_fmac_f32_e32, v_fma_f32,
  v_cmp_gt_f32_e32, v_cmp_gt_f32_e64,
  v_cndmask_b32_e32, v_cndmask_b32_e64,
  s_nop,
)

from extra.sqtt.rgp.microbench import microbench, Kernel


# ── shared helpers ───────────────────────────────────────────────────────────

# One source-VGPR per bank, all in the safe scratch window (>= v[4]).
_BANK_REG = {0: 4, 1: 5, 2: 6, 3: 7}


def _seed_vgprs(k: "Kernel", idxs) -> None:
  """v_mov_b32_e32 a float constant into each v[i] so VALU / v_cmp sources
  are valid. Uses 1.0, 2.0, ... to keep them distinct. Caller must not
  include v[0] or v[1] — those are live-in from the standard prologue
  (offset / loaded value) and live-out to the standard epilogue."""
  for n, i in enumerate(idxs):
    assert i >= 4, f"v[{i}] overlaps with the prologue-owned v[0]/v[1]; use v[4..]"
    k.emit(v_mov_b32_e32(v[i], float(n + 1)))


# =============================================================================
# B.2 - VGPR bank-port conflict probes (24 kernels)
# =============================================================================
# bank(v[n]) = n % 4. Each probe holds either src-bank(s), dst-bank, or the
# 2-VALU bank-pair pattern constant so that a timing mismatch unambiguously
# points to one model dimension.

# B.2.1 — same-src-bank (bank-0 read pressure) --------------------------------
# `v_add v[D], v[4], v[4]` x4: both srcs bank 0 (v[4]). Dst alternates banks
# so the write-port isn't also contending. Measures same-bank dual-read cost.
@microbench(name="mb_vgpr_bank_same_src_n4", category="valu_bank")
def _mb_vgpr_bank_same_src_n4(k: "Kernel") -> None:
  _seed_vgprs(k, [4])
  for d in (16, 17, 18, 19):
    k.emit(v_add_f32_e32(v[d], v[4], v[4]))


# `v_add v[D], v[5], v[6]` x4: src0 bank 1, src1 bank 2 — all distinct from
# each other and from whatever bank dst lands in. No read-port conflict.
@microbench(name="mb_vgpr_bank_diff_src_n4", category="valu_bank")
def _mb_vgpr_bank_diff_src_n4(k: "Kernel") -> None:
  _seed_vgprs(k, [5, 6])
  for d in (16, 17, 18, 19):
    k.emit(v_add_f32_e32(v[d], v[5], v[6]))


# `v_fmac v[8], v[4], v[4]`: src0=src1=v[4] (bank 0) and acc v[8] (also
# bank 0). Three bank-0 reads per cycle.
@microbench(name="mb_vgpr_bank_src2_same_n4", category="valu_bank")
def _mb_vgpr_bank_src2_same_n4(k: "Kernel") -> None:
  _seed_vgprs(k, [4, 8])
  for _ in range(4):
    k.emit(v_fmac_f32_e32(v[8], v[4], v[4]))


# B.2.2 — 3-src (v_fma_f32) bank combinations --------------------------------
# dst = v[16] (bank 0). Sources vary across the bank-occupancy axes.

@microbench(name="mb_vgpr_bank_3src_b0_b0_b0", category="valu_bank")
def _mb_vgpr_bank_3src_b0_b0_b0(k: "Kernel") -> None:
  _seed_vgprs(k, [4])
  k.emit(v_fma_f32(v[16], v[4], v[4], v[4]))


# Three distinct banks — baseline, no conflict expected.
@microbench(name="mb_vgpr_bank_3src_b0_b1_b2", category="valu_bank")
def _mb_vgpr_bank_3src_b0_b1_b2(k: "Kernel") -> None:
  _seed_vgprs(k, [4, 5, 6])
  k.emit(v_fma_f32(v[16], v[4], v[5], v[6]))


# Two in bank 0, one in bank 1 — partial conflict.
@microbench(name="mb_vgpr_bank_3src_b0_b0_b1", category="valu_bank")
def _mb_vgpr_bank_3src_b0_b0_b1(k: "Kernel") -> None:
  _seed_vgprs(k, [4, 5])
  k.emit(v_fma_f32(v[16], v[4], v[4], v[5]))


# Same-bank triple reads for each of the other three banks (fills out the
# bank symmetry axis). These diagnose whether the bank-conflict model is
# bank-0-biased.
@microbench(name="mb_vgpr_bank_3src_b1_b1_b1", category="valu_bank")
def _mb_vgpr_bank_3src_b1_b1_b1(k: "Kernel") -> None:
  _seed_vgprs(k, [5])
  k.emit(v_fma_f32(v[17], v[5], v[5], v[5]))


@microbench(name="mb_vgpr_bank_3src_b2_b2_b2", category="valu_bank")
def _mb_vgpr_bank_3src_b2_b2_b2(k: "Kernel") -> None:
  _seed_vgprs(k, [6])
  k.emit(v_fma_f32(v[18], v[6], v[6], v[6]))


@microbench(name="mb_vgpr_bank_3src_b3_b3_b3", category="valu_bank")
def _mb_vgpr_bank_3src_b3_b3_b3(k: "Kernel") -> None:
  _seed_vgprs(k, [7])
  k.emit(v_fma_f32(v[19], v[7], v[7], v[7]))


# B.2.3 — 2-VALU adjacent-cycle pair (bank 0 -> bank K) -----------------------
# 1st VALU reads bank 0; 2nd VALU reads bank K. Dsts are on distinct banks
# to remove write-port noise.

def _mb_vgpr_bank_pair(second_bank: int):
  s0 = _BANK_REG[0]          # bank-0 source for the 1st v_add
  s1 = _BANK_REG[second_bank]  # bank-K source for the 2nd v_add
  def _body(k: "Kernel") -> None:
    _seed_vgprs(k, sorted({s0, s1}))
    k.emit(v_add_f32_e32(v[16], v[s0], v[s0]))  # reads bank 0
    k.emit(v_add_f32_e32(v[20], v[s1], v[s1]))  # reads bank `second_bank`
  return _body

microbench(name="mb_vgpr_bank_pair_b0_b0", category="valu_bank")(_mb_vgpr_bank_pair(0))
microbench(name="mb_vgpr_bank_pair_b0_b1", category="valu_bank")(_mb_vgpr_bank_pair(1))
microbench(name="mb_vgpr_bank_pair_b0_b2", category="valu_bank")(_mb_vgpr_bank_pair(2))
microbench(name="mb_vgpr_bank_pair_b0_b3", category="valu_bank")(_mb_vgpr_bank_pair(3))


# B.2.4 — long chain all reading bank 0 / bank 1 (saturation) -----------------
# 8 back-to-back VALUs all reading v[4] (or v[5]). Each writes a distinct
# dst VGPR so there's no RAW dep between them — the only timing signal is
# the read-bank contention.
@microbench(name="mb_vgpr_bank_chain_b0_aaaa", category="valu_bank")
def _mb_vgpr_bank_chain_b0_aaaa(k: "Kernel") -> None:
  _seed_vgprs(k, [4])
  for d in range(16, 24):
    k.emit(v_add_f32_e32(v[d], v[4], v[4]))


@microbench(name="mb_vgpr_bank_chain_b1_aaaa", category="valu_bank")
def _mb_vgpr_bank_chain_b1_aaaa(k: "Kernel") -> None:
  _seed_vgprs(k, [5])
  for d in range(16, 24):
    k.emit(v_add_f32_e32(v[d], v[5], v[5]))


# B.2.5 — dst-bank chain (writes all in bank K) -------------------------------
# 4 VALUs whose dst VGPRs all share one bank. Srcs spread across banks so
# the only controlled variable is the dst bank. Tests whether writeback
# contention is modeled separately from read contention.

def _mb_vgpr_bank_dst_chain(dst_bank: int):
  """4 v_adds writing v[16+dst_bank], v[20+dst_bank], v[24+dst_bank],
  v[28+dst_bank] (all dst_bank % 4). Sources span banks 1,2,3,1."""
  def _body(k: "Kernel") -> None:
    _seed_vgprs(k, [5, 6, 7])
    dsts = [16 + dst_bank, 20 + dst_bank, 24 + dst_bank, 28 + dst_bank]
    srcs = [5, 6, 7, 5]
    for d, sr in zip(dsts, srcs):
      k.emit(v_add_f32_e32(v[d], v[sr], v[sr]))
  return _body

microbench(name="mb_vgpr_bank_dst_chain_b0", category="valu_bank")(_mb_vgpr_bank_dst_chain(0))
microbench(name="mb_vgpr_bank_dst_chain_b1", category="valu_bank")(_mb_vgpr_bank_dst_chain(1))
microbench(name="mb_vgpr_bank_dst_chain_b2", category="valu_bank")(_mb_vgpr_bank_dst_chain(2))
microbench(name="mb_vgpr_bank_dst_chain_b3", category="valu_bank")(_mb_vgpr_bank_dst_chain(3))


# B.2.6 — 2-src (VOP2) bank pair x4 -------------------------------------------
# 4 back-to-back VOP2 v_adds, each reading (src0 in bank 0) and (src1 in
# bank k). Distinct dst per instruction so no RAW between them.

def _mb_vgpr_bank_2src_pair(src1_bank: int):
  s0 = _BANK_REG[0]
  s1 = _BANK_REG[src1_bank]
  def _body(k: "Kernel") -> None:
    _seed_vgprs(k, sorted({s0, s1}))
    for d in (16, 17, 18, 19):
      k.emit(v_add_f32_e32(v[d], v[s0], v[s1]))
  return _body

microbench(name="mb_vgpr_bank_2src_b0b0_n4", category="valu_bank")(_mb_vgpr_bank_2src_pair(0))
microbench(name="mb_vgpr_bank_2src_b0b1_n4", category="valu_bank")(_mb_vgpr_bank_2src_pair(1))
microbench(name="mb_vgpr_bank_2src_b0b2_n4", category="valu_bank")(_mb_vgpr_bank_2src_pair(2))
microbench(name="mb_vgpr_bank_2src_b0b3_n4", category="valu_bank")(_mb_vgpr_bank_2src_pair(3))


# B.2.7 — writes to adjacent odd banks ----------------------------------------
# 4 v_adds whose dsts are v[17], v[19], v[21], v[23] — banks 1, 3, 1, 3
# (adjacent odd-bank stripe). Srcs stay in bank 2 to isolate the write-side
# path from the read path.
@microbench(name="mb_vgpr_bank_write_chain_b1357", category="valu_bank")
def _mb_vgpr_bank_write_chain_b1357(k: "Kernel") -> None:
  _seed_vgprs(k, [6])
  for d in (17, 19, 21, 23):
    k.emit(v_add_f32_e32(v[d], v[6], v[6]))


# =============================================================================
# B.4 - v_cmp / v_cndmask SGPR-forwarding probes (15 kernels)
# =============================================================================
# Every probe writes SGPRs via v_cmp_gt_f32_e64 and consumes them via
# v_cndmask_b32_e64. SGPR indices start at s[4] so we never clobber s[0:1]
# (the kernel-arg pointer that the epilogue's global_store reads). Source
# VGPRs start at v[4] to leave v[0]/v[1] alone.

# B.4.1 — k v_cmps followed by k v_cndmasks (k-sweep) -------------------------
# Each cndmask reads a fresh SGPR. First cndmask hits cold-SGPR slip (+1 cy);
# remaining cndmasks dt=1. At k=5+, the F1/F3 tail-slip adds another +1 cy
# to the 5th-and-beyond cndmask.

def _mb_vcmp_cndmask_k(k_count: int):
  def _body(k: "Kernel") -> None:
    # seed v[4..4+k_count-1] as v_cmp sources
    _seed_vgprs(k, list(range(4, 4 + k_count)))
    # write s[4..4+k_count-1]
    for i in range(k_count):
      k.emit(v_cmp_gt_f32_e64(s[4 + i], 0.5, v[4 + i]))
    # cndmasks each write a distinct high dst VGPR.
    for i in range(k_count):
      k.emit(v_cndmask_b32_e64(v[20 + i], 0.0, v[4 + i], s[4 + i]))
  return _body

microbench(name="mb_vcmp_cndmask_k1", category="cndmask")(_mb_vcmp_cndmask_k(1))
microbench(name="mb_vcmp_cndmask_k2", category="cndmask")(_mb_vcmp_cndmask_k(2))
microbench(name="mb_vcmp_cndmask_k4", category="cndmask")(_mb_vcmp_cndmask_k(4))
microbench(name="mb_vcmp_cndmask_k8", category="cndmask")(_mb_vcmp_cndmask_k(8))


# B.4.2 — spacing sweep: vcmp -> s_nop(N) -> cndmask --------------------------
# Draws the cold -> warm -> drained curve for v_cmp -> v_cndmask SGPR
# forwarding. s_nop(N) contributes N+1 cycles of gap on RDNA3.

@microbench(name="mb_vcmp_spaced_cndmask", category="cndmask")
def _mb_vcmp_spaced_cndmask(k: "Kernel") -> None:
  _seed_vgprs(k, [4])
  k.emit(v_cmp_gt_f32_e64(s[4], 0.5, v[4]))
  k.emit(s_nop(4))
  k.emit(v_cndmask_b32_e64(v[20], 0.0, v[4], s[4]))


@microbench(name="mb_vcmp_spaced_cndmask_nop8", category="cndmask")
def _mb_vcmp_spaced_cndmask_nop8(k: "Kernel") -> None:
  _seed_vgprs(k, [4])
  k.emit(v_cmp_gt_f32_e64(s[4], 0.5, v[4]))
  k.emit(s_nop(8))
  k.emit(v_cndmask_b32_e64(v[20], 0.0, v[4], s[4]))


@microbench(name="mb_vcmp_spaced_cndmask_nop12", category="cndmask")
def _mb_vcmp_spaced_cndmask_nop12(k: "Kernel") -> None:
  _seed_vgprs(k, [4])
  k.emit(v_cmp_gt_f32_e64(s[4], 0.5, v[4]))
  k.emit(s_nop(12))
  k.emit(v_cndmask_b32_e64(v[20], 0.0, v[4], s[4]))


# B.4.3 — tail-retire probes (F1/F3) -----------------------------------------
# After 4 cndmasks retire, does the 5th cndmask reading the *same* SGPR
# queue pick up a tail slip?
@microbench(name="mb_cndmask_tail_retire", category="cndmask")
def _mb_cndmask_tail_retire(k: "Kernel") -> None:
  _seed_vgprs(k, [4, 5, 6, 7])
  for i in range(4):
    k.emit(v_cmp_gt_f32_e64(s[4 + i], 0.5, v[4 + i]))
  for i in range(4):
    k.emit(v_cndmask_b32_e64(v[20 + i], 0.0, v[4 + i], s[4 + i]))
  # 5th cndmask reads s[4] again (already-retired SGPR).
  k.emit(v_cndmask_b32_e64(v[24], 0.0, v[4], s[4]))


# 5th cndmask reads a NEW SGPR (s[8]) that was written before the chain but
# not yet consumed.
@microbench(name="mb_cndmask_tail_new_sgpr", category="cndmask")
def _mb_cndmask_tail_new_sgpr(k: "Kernel") -> None:
  _seed_vgprs(k, [4, 5, 6, 7, 8])
  # write s[4..8] (5 vcmps)
  for i in range(5):
    k.emit(v_cmp_gt_f32_e64(s[4 + i], 0.5, v[4 + i]))
  # consume s[4..7] via cndmasks
  for i in range(4):
    k.emit(v_cndmask_b32_e64(v[20 + i], 0.0, v[4 + i], s[4 + i]))
  # 5th cndmask reads s[8] — freshly-written but untouched by the chain.
  k.emit(v_cndmask_b32_e64(v[24], 0.0, v[8], s[8]))


# 5th cndmask writes a FRESH, never-touched high VGPR (v[30] = bank 2).
@microbench(name="mb_cndmask_new_vgpr_tail", category="cndmask")
def _mb_cndmask_new_vgpr_tail(k: "Kernel") -> None:
  _seed_vgprs(k, [4, 5, 6, 7])
  for i in range(4):
    k.emit(v_cmp_gt_f32_e64(s[4 + i], 0.5, v[4 + i]))
  for i in range(4):
    k.emit(v_cndmask_b32_e64(v[20 + i], 0.0, v[4 + i], s[4 + i]))
  # Fresh VGPR destination (v[30] never written before).
  k.emit(v_cndmask_b32_e64(v[30], 0.0, v[4], s[4]))


# B.4.4 — trailing v_cmp after a cndmask chain -------------------------------
# 4 cndmasks consume s[4..7], then a fresh v_cmp writes s[8]. Measures the
# closing-stall cost after the cndmask queue drains.
@microbench(name="mb_vcmp_after_cndmask_chain", category="cndmask")
def _mb_vcmp_after_cndmask_chain(k: "Kernel") -> None:
  _seed_vgprs(k, [4, 5, 6, 7])
  for i in range(4):
    k.emit(v_cmp_gt_f32_e64(s[4 + i], 0.5, v[4 + i]))
  for i in range(4):
    k.emit(v_cndmask_b32_e64(v[20 + i], 0.0, v[4 + i], s[4 + i]))
  # Tail v_cmp writing a distinct SGPR — no RAW/WAW with the chain.
  k.emit(v_cmp_gt_f32_e64(s[8], 0.5, v[4]))


# B.4.5 — interleaved vcmp/cndmask/... ---------------------------------------
# Each cndmask sits immediately after its producing vcmp (RAW next-cy).
@microbench(name="mb_vcmp_interleave_cndmask", category="cndmask")
def _mb_vcmp_interleave_cndmask(k: "Kernel") -> None:
  _seed_vgprs(k, [4, 5, 6, 7])
  for i in range(4):
    k.emit(v_cmp_gt_f32_e64(s[4 + i], 0.5, v[4 + i]))
    k.emit(v_cndmask_b32_e64(v[20 + i], 0.0, v[4 + i], s[4 + i]))


# B.4.6 — e32 vs e64 encoding ------------------------------------------------
# 4 v_cmp_e32 (writing VCC) then 4 v_cmp_e64 (writing s[4..7]). Diagnoses
# whether VCC and per-SGPR paths share the same issue throughput.
@microbench(name="mb_vcmp_e32_vs_e64", category="cndmask")
def _mb_vcmp_e32_vs_e64(k: "Kernel") -> None:
  _seed_vgprs(k, [4, 5, 6, 7])
  for i in range(4):
    k.emit(v_cmp_gt_f32_e32(0.5, v[4 + i]))           # writes VCC
  for i in range(4):
    k.emit(v_cmp_gt_f32_e64(s[4 + i], 0.5, v[4 + i])) # writes s[4..7]


# B.4.7 — cndmask reading VCC then cndmask reading SGPR ----------------------
@microbench(name="mb_cndmask_read_vcc_then_sgpr", category="cndmask")
def _mb_cndmask_read_vcc_then_sgpr(k: "Kernel") -> None:
  _seed_vgprs(k, [4, 5])
  # Prime VCC (e32) and s[4] (e64) with distinct vcmps.
  k.emit(v_cmp_gt_f32_e32(0.5, v[4]))                # writes VCC
  k.emit(v_cmp_gt_f32_e64(s[4], 0.5, v[5]))          # writes s[4]
  # 1st cndmask reads VCC (e32); 2nd cndmask reads s[4] (e64).
  k.emit(v_cndmask_b32_e32(v[20], 0.0, v[5]))
  k.emit(v_cndmask_b32_e64(v[21], 0.0, v[5], s[4]))


# B.4.8 — SGPR k-sweep: K v_cmps writing s[4..4+K-1], then ONE cndmask
# reading the LAST SGPR (s[4+K-1]). K=4 is the headline value.
@microbench(name="mb_cndmask_sgpr_k_sweep", category="cndmask")
def _mb_cndmask_sgpr_k_sweep(k: "Kernel") -> None:
  k_count = 4
  _seed_vgprs(k, list(range(4, 4 + k_count)))
  for i in range(k_count):
    k.emit(v_cmp_gt_f32_e64(s[4 + i], 0.5, v[4 + i]))
  # single tail cndmask reads the LAST-written SGPR
  k.emit(v_cndmask_b32_e64(v[20], 0.0, v[4], s[4 + k_count - 1]))


__all__: list[str] = []  # registration happens as a side effect of import
