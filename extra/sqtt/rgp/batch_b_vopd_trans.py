#!/usr/bin/env python3
"""Batch B.3 + B.5 — VOPD eligibility and trans-pipe interaction microbenches.

55 one-knob probes targeting VOPD dual-issue and transcendental pipe behavior:
  B.3.a — VOPD opcode pair grid (20 kernels)
  B.3.b — VOPD dependency / bank / VCC matrix (20 kernels)
  B.5   — Trans pipe interactions (15 kernels)

All kernels follow the standard microbench prologue / epilogue from
extra.sqtt.rgp.microbench.  Each body seeds independent VGPRs (v[4..11]) and
SGPRs (s[4..7]) before the instructions under test so the issue stream is
purely VALU/VOPD/TRANS traffic.

VOPD register layout (satisfies RDNA3 bank rules):
  vdstx = v[4]  (bank 0, even)   vdsty  = v[5]  (bank 1, odd)
  srcx0 = v[6]  (bank 2)         srcy0  = v[7]  (bank 3)    — disjoint banks
  vsrcx1 = v[8] (bank 0)         vsrcy1 = v[9] (bank 1)     — disjoint banks
Additional dst blocks for chains use {v[10]/v[11], v[12]/v[13], v[14]/v[15]}.
"""
from __future__ import annotations

from tinygrad.renderer.amd.dsl import s, v, NULL, VCC_LO  # noqa: F401
from tinygrad.runtime.autogen.amd.rdna3.ins import (
  v_add_f32_e32, v_mul_f32_e32, v_fmac_f32_e32, v_sub_f32_e32, v_mov_b32_e32,
  v_exp_f32_e32, v_log_f32_e32, v_sqrt_f32_e32, v_rcp_f32_e32, v_rsq_f32_e32,
  v_cmp_gt_f32_e32, v_cmp_gt_f32_e64, v_cndmask_b32_e32,
  s_mov_b32, s_nop, s_waitcnt, s_waitcnt_depctr, s_waitcnt_vmcnt,
  global_load_b32,
  VOPD, VOPD_LIT, VOPDOp,
)

from extra.sqtt.rgp.microbench import microbench, Kernel  # noqa: F401


# ── Shared helpers ───────────────────────────────────────────────────────────

def _seed_vopd_sources(k: "Kernel") -> None:
  """Init v[4..11] to independent non-zero values so VOPDs have real inputs."""
  for i in range(8): k.emit(v_mov_b32_e32(v[4 + i], float(i + 1)))

def _seed_vopd_ext(k: "Kernel") -> None:
  """Seed v[4..15] so chain-of-4 VOPDs have disjoint dsts."""
  for i in range(12): k.emit(v_mov_b32_e32(v[4 + i], float(i + 1)))

def _seed_sgpr_cmps(k: "Kernel") -> None:
  """Produce VCC and s[4..7] from v_cmps so cndmask/sgpr probes have live srcs."""
  k.emit(v_cmp_gt_f32_e32(0.5, v[4]))                    # VCC <- v[4] > 0.5
  k.emit(v_cmp_gt_f32_e64(s[4], 0.5, v[5]))              # s[4]
  k.emit(v_cmp_gt_f32_e64(s[5], 0.5, v[6]))              # s[5]
  k.emit(v_cmp_gt_f32_e64(s[6], 0.5, v[7]))              # s[6]
  k.emit(v_cmp_gt_f32_e64(s[7], 0.5, v[8]))              # s[7]

def _vopd(opx, opy, x=4, y=5, sx0=6, sy0=7, vx1=8, vy1=9):
  """Emit a canonical VOPD pair honoring bank + parity rules."""
  return VOPD(
    opx=opx, opy=opy,
    vdstx=v[x],   srcx0=v[sx0], vsrcx1=v[vx1],
    vdsty=v[y],   srcy0=v[sy0], vsrcy1=v[vy1],
  )

def _f2i(f: float) -> int:
  """float32 -> int32 bit pattern (for VOPD_LIT literal=...)."""
  import struct
  return int.from_bytes(struct.pack("<f", float(f)), "little", signed=False)


# ═════════════════════════════════════════════════════════════════════════════
# B.3.a — VOPD opcode pair grid (20 kernels)
# Each kernel emits 2 back-to-back VOPDs of a single (opx, opy) pair.
# Expected: fused, dt=1,1 for most — diagnostic markers for VOPD eligibility.
# ═════════════════════════════════════════════════════════════════════════════

@microbench(name="mb_vopd_add_add", category="vopd")
def _mb_vopd_add_add(k: "Kernel") -> None:
  _seed_vopd_sources(k)
  for _ in range(2): k.emit(_vopd(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_ADD_F32))

@microbench(name="mb_vopd_mul_mul", category="vopd")
def _mb_vopd_mul_mul(k: "Kernel") -> None:
  _seed_vopd_sources(k)
  for _ in range(2): k.emit(_vopd(VOPDOp.V_DUAL_MUL_F32, VOPDOp.V_DUAL_MUL_F32))

@microbench(name="mb_vopd_fmac_fmac", category="vopd")
def _mb_vopd_fmac_fmac(k: "Kernel") -> None:
  _seed_vopd_sources(k)
  for _ in range(2): k.emit(_vopd(VOPDOp.V_DUAL_FMAC_F32, VOPDOp.V_DUAL_FMAC_F32))

@microbench(name="mb_vopd_cndmask_cndmask", category="vopd")
def _mb_vopd_cndmask_cndmask(k: "Kernel") -> None:
  _seed_vopd_sources(k)
  # cndmask reads VCC implicitly — write it first so sources are well-defined.
  k.emit(v_cmp_gt_f32_e32(0.5, v[4]))
  for _ in range(2):
    k.emit(_vopd(VOPDOp.V_DUAL_CNDMASK_B32, VOPDOp.V_DUAL_CNDMASK_B32))

@microbench(name="mb_vopd_add_mul", category="vopd")
def _mb_vopd_add_mul(k: "Kernel") -> None:
  _seed_vopd_sources(k)
  for _ in range(2): k.emit(_vopd(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_MUL_F32))

@microbench(name="mb_vopd_mul_add", category="vopd")
def _mb_vopd_mul_add(k: "Kernel") -> None:
  _seed_vopd_sources(k)
  for _ in range(2): k.emit(_vopd(VOPDOp.V_DUAL_MUL_F32, VOPDOp.V_DUAL_ADD_F32))

@microbench(name="mb_vopd_fmac_add", category="vopd")
def _mb_vopd_fmac_add(k: "Kernel") -> None:
  _seed_vopd_sources(k)
  for _ in range(2): k.emit(_vopd(VOPDOp.V_DUAL_FMAC_F32, VOPDOp.V_DUAL_ADD_F32))

@microbench(name="mb_vopd_fmac_mul", category="vopd")
def _mb_vopd_fmac_mul(k: "Kernel") -> None:
  _seed_vopd_sources(k)
  for _ in range(2): k.emit(_vopd(VOPDOp.V_DUAL_FMAC_F32, VOPDOp.V_DUAL_MUL_F32))

@microbench(name="mb_vopd_cndmask_add", category="vopd")
def _mb_vopd_cndmask_add(k: "Kernel") -> None:
  _seed_vopd_sources(k)
  k.emit(v_cmp_gt_f32_e32(0.5, v[4]))
  for _ in range(2): k.emit(_vopd(VOPDOp.V_DUAL_CNDMASK_B32, VOPDOp.V_DUAL_ADD_F32))

@microbench(name="mb_vopd_cndmask_mul", category="vopd")
def _mb_vopd_cndmask_mul(k: "Kernel") -> None:
  _seed_vopd_sources(k)
  k.emit(v_cmp_gt_f32_e32(0.5, v[4]))
  for _ in range(2): k.emit(_vopd(VOPDOp.V_DUAL_CNDMASK_B32, VOPDOp.V_DUAL_MUL_F32))

@microbench(name="mb_vopd_sub_sub", category="vopd")
def _mb_vopd_sub_sub(k: "Kernel") -> None:
  _seed_vopd_sources(k)
  for _ in range(2): k.emit(_vopd(VOPDOp.V_DUAL_SUB_F32, VOPDOp.V_DUAL_SUB_F32))

@microbench(name="mb_vopd_min_max", category="vopd")
def _mb_vopd_min_max(k: "Kernel") -> None:
  _seed_vopd_sources(k)
  for _ in range(2): k.emit(_vopd(VOPDOp.V_DUAL_MIN_F32, VOPDOp.V_DUAL_MAX_F32))

@microbench(name="mb_vopd_mov_add", category="vopd")
def _mb_vopd_mov_add(k: "Kernel") -> None:
  _seed_vopd_sources(k)
  for _ in range(2): k.emit(_vopd(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_ADD_F32))

@microbench(name="mb_vopd_mov_mov", category="vopd")
def _mb_vopd_mov_mov(k: "Kernel") -> None:
  _seed_vopd_sources(k)
  for _ in range(2): k.emit(_vopd(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_MOV_B32))

@microbench(name="mb_vopd_dot2_add", category="vopd")
def _mb_vopd_dot2_add(k: "Kernel") -> None:
  # v_dual_dot2acc_f32_f16 packs 2×f16 lanes in each src; HW may or may not fuse
  # it with a pure-f32 add — either way, we log the outcome.
  _seed_vopd_sources(k)
  for _ in range(2): k.emit(_vopd(VOPDOp.V_DUAL_DOT2ACC_F32_F16, VOPDOp.V_DUAL_ADD_F32))

@microbench(name="mb_vopd_add_int", category="vopd")
def _mb_vopd_add_int(k: "Kernel") -> None:
  # V_DUAL_ADD_NC_U32 (enum value 16) only fits the 5-bit opy slot; opx is
  # 4 bits. Pair it with a f32 opx so the integer lane is exercised.
  _seed_vopd_sources(k)
  for _ in range(2): k.emit(_vopd(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_ADD_NC_U32))

@microbench(name="mb_vopd_mixed_f32_f16", category="vopd")
def _mb_vopd_mixed_f32_f16(k: "Kernel") -> None:
  # dot2acc_f32_f16 reads its operands as packed f16 while add_f32 is pure f32.
  # This tests whether the scheduler rejects the mixed-precision pairing.
  _seed_vopd_sources(k)
  for _ in range(2): k.emit(_vopd(VOPDOp.V_DUAL_DOT2ACC_F32_F16, VOPDOp.V_DUAL_MUL_F32))

@microbench(name="mb_vopd_mul_fmac_bank_same", category="vopd")
def _mb_vopd_mul_fmac_bank_same(k: "Kernel") -> None:
  # Force srcx0 and srcy0 into the SAME VGPR bank (both reading v[6]).
  # HW should split / refuse to dual-issue.
  _seed_vopd_sources(k)
  for _ in range(2):
    k.emit(VOPD(
      opx=VOPDOp.V_DUAL_MUL_F32, opy=VOPDOp.V_DUAL_FMAC_F32,
      vdstx=v[4],   srcx0=v[6], vsrcx1=v[8],
      vdsty=v[5],   srcy0=v[6], vsrcy1=v[9],     # srcy0 = srcx0 = v[6]
    ))

@microbench(name="mb_vopd_lit_add", category="vopd")
def _mb_vopd_lit_add(k: "Kernel") -> None:
  # VOPD_LIT: v_dual_fmaak_f32 uses a 32-bit SIMM literal.  Pair it with a
  # v_dual_add_f32 (no literal on the Y lane).
  _seed_vopd_sources(k)
  for _ in range(2):
    k.emit(VOPD_LIT(
      opx=VOPDOp.V_DUAL_FMAAK_F32, opy=VOPDOp.V_DUAL_ADD_F32,
      vdstx=v[4],   srcx0=v[6], vsrcx1=v[8],
      vdsty=v[5],   srcy0=v[7], vsrcy1=v[9],
      literal=_f2i(2.0),
    ))

@microbench(name="mb_vopd_lit_lit", category="vopd")
def _mb_vopd_lit_lit(k: "Kernel") -> None:
  # Two VOPD_LIT back-to-back: FMAAK on X, MOV on Y (single shared literal).
  # Note: one instruction can carry at most one 32-bit literal, so this is a
  # chain of two independent VOPD_LITs (each with its own literal), not a
  # single packet with two literals.
  _seed_vopd_sources(k)
  for _ in range(2):
    k.emit(VOPD_LIT(
      opx=VOPDOp.V_DUAL_FMAAK_F32, opy=VOPDOp.V_DUAL_MOV_B32,
      vdstx=v[4],   srcx0=v[6], vsrcx1=v[8],
      vdsty=v[5],   srcy0=v[7], vsrcy1=v[9],
      literal=_f2i(3.0),
    ))


# ═════════════════════════════════════════════════════════════════════════════
# B.3.b — VOPD dependency / bank / VCC matrix (20 kernels)
# Same opcode family; vary RAW/WAR/bank/VCC relationships between 2-4 VOPDs.
# ═════════════════════════════════════════════════════════════════════════════

@microbench(name="mb_vopd_pair_no_raw", category="vopd")
def _mb_vopd_pair_no_raw(k: "Kernel") -> None:
  # Two VOPDs on disjoint register blocks: dsts v[4]/v[5] then v[10]/v[11],
  # sources also disjoint (v[6..9] vs v[12..15]).
  _seed_vopd_ext(k)
  k.emit(_vopd(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_MUL_F32, 4, 5, 6, 7, 8, 9))
  k.emit(_vopd(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_MUL_F32, 10, 11, 12, 13, 14, 15))

@microbench(name="mb_vopd_pair_raw_x", category="vopd")
def _mb_vopd_pair_raw_x(k: "Kernel") -> None:
  # VOPD#2.srcx0 = VOPD#1.vdstx.  Expect dt=1,3 — cross-VOPD X-lane RAW.
  _seed_vopd_ext(k)
  k.emit(_vopd(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_MUL_F32, 4, 5, 6, 7, 8, 9))
  k.emit(VOPD(
    opx=VOPDOp.V_DUAL_ADD_F32, opy=VOPDOp.V_DUAL_MUL_F32,
    vdstx=v[10], srcx0=v[4],  vsrcx1=v[12],   # srcx0=v[4] = prior vdstx
    vdsty=v[11], srcy0=v[13], vsrcy1=v[14],
  ))

@microbench(name="mb_vopd_pair_raw_y", category="vopd")
def _mb_vopd_pair_raw_y(k: "Kernel") -> None:
  # VOPD#2.srcy0 = VOPD#1.vdsty.  Cross-VOPD Y-lane RAW.
  _seed_vopd_ext(k)
  k.emit(_vopd(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_MUL_F32, 4, 5, 6, 7, 8, 9))
  k.emit(VOPD(
    opx=VOPDOp.V_DUAL_ADD_F32, opy=VOPDOp.V_DUAL_MUL_F32,
    vdstx=v[10], srcx0=v[12], vsrcx1=v[14],
    vdsty=v[11], srcy0=v[5],  vsrcy1=v[13],   # srcy0=v[5] = prior vdsty
  ))

@microbench(name="mb_vopd_pair_raw_xy", category="vopd")
def _mb_vopd_pair_raw_xy(k: "Kernel") -> None:
  # Both X and Y lanes have a cross-VOPD RAW dependency.
  _seed_vopd_ext(k)
  k.emit(_vopd(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_MUL_F32, 4, 5, 6, 7, 8, 9))
  k.emit(VOPD(
    opx=VOPDOp.V_DUAL_ADD_F32, opy=VOPDOp.V_DUAL_MUL_F32,
    vdstx=v[10], srcx0=v[4],  vsrcx1=v[12],   # RAW on X
    vdsty=v[11], srcy0=v[5],  vsrcy1=v[13],   # RAW on Y
  ))

@microbench(name="mb_vopd_pair_war", category="vopd")
def _mb_vopd_pair_war(k: "Kernel") -> None:
  # Write-after-read: VOPD#2 writes registers that VOPD#1 read.
  _seed_vopd_ext(k)
  # VOPD#1: dst v[10]/v[11], reads v[6..9]
  k.emit(VOPD(
    opx=VOPDOp.V_DUAL_ADD_F32, opy=VOPDOp.V_DUAL_MUL_F32,
    vdstx=v[10], srcx0=v[6], vsrcx1=v[8],
    vdsty=v[11], srcy0=v[7], vsrcy1=v[9],
  ))
  # VOPD#2: overwrites v[6]/v[7] (which #1 read as srcx0/srcy0) — pure WAR.
  k.emit(VOPD(
    opx=VOPDOp.V_DUAL_ADD_F32, opy=VOPDOp.V_DUAL_MUL_F32,
    vdstx=v[6],  srcx0=v[12], vsrcx1=v[14],
    vdsty=v[7],  srcy0=v[13], vsrcy1=v[15],
  ))

@microbench(name="mb_vopd_chain_n4_raw", category="vopd")
def _mb_vopd_chain_n4_raw(k: "Kernel") -> None:
  # 4 VOPDs, each reading the prior VOPD's vdstx/vdsty as sources.
  _seed_vopd_ext(k)
  # VOPD#1: writes v[4]/v[5]
  k.emit(VOPD(
    opx=VOPDOp.V_DUAL_ADD_F32, opy=VOPDOp.V_DUAL_MUL_F32,
    vdstx=v[4],  srcx0=v[6], vsrcx1=v[8],
    vdsty=v[5],  srcy0=v[7], vsrcy1=v[9],
  ))
  # VOPD#2: writes v[10]/v[11], reads v[4] (X RAW) + v[5] (Y RAW).
  k.emit(VOPD(
    opx=VOPDOp.V_DUAL_ADD_F32, opy=VOPDOp.V_DUAL_MUL_F32,
    vdstx=v[10], srcx0=v[4],  vsrcx1=v[12],
    vdsty=v[11], srcy0=v[5],  vsrcy1=v[13],
  ))
  # VOPD#3: writes v[14]/v[15], reads v[10]/v[11].
  k.emit(VOPD(
    opx=VOPDOp.V_DUAL_ADD_F32, opy=VOPDOp.V_DUAL_MUL_F32,
    vdstx=v[14], srcx0=v[10], vsrcx1=v[12],
    vdsty=v[15], srcy0=v[11], vsrcy1=v[13],
  ))
  # VOPD#4: writes v[4]/v[5] again, reads v[14]/v[15].
  k.emit(VOPD(
    opx=VOPDOp.V_DUAL_ADD_F32, opy=VOPDOp.V_DUAL_MUL_F32,
    vdstx=v[4],  srcx0=v[14], vsrcx1=v[8],
    vdsty=v[5],  srcy0=v[15], vsrcy1=v[9],
  ))

@microbench(name="mb_vopd_chain_n4_no_raw", category="vopd")
def _mb_vopd_chain_n4_no_raw(k: "Kernel") -> None:
  # 4 VOPDs, all independent (no cross-VOPD RAW).  All read v[6..9], write
  # disjoint dst pairs v[4]/v[5], v[10]/v[11], v[12]/v[13], v[14]/v[15].
  _seed_vopd_ext(k)
  dsts = [(4, 5), (10, 11), (12, 13), (14, 15)]
  for (x, y) in dsts:
    k.emit(VOPD(
      opx=VOPDOp.V_DUAL_ADD_F32, opy=VOPDOp.V_DUAL_MUL_F32,
      vdstx=v[x], srcx0=v[6], vsrcx1=v[8],
      vdsty=v[y], srcy0=v[7], vsrcy1=v[9],
    ))

@microbench(name="mb_vopd_vcc_producer_then_cndmask", category="vopd")
def _mb_vopd_vcc_producer_then_cndmask(k: "Kernel") -> None:
  # v_cmp writes VCC, then a VOPD(cndmask, cndmask) consumes it.
  _seed_vopd_ext(k)
  k.emit(v_cmp_gt_f32_e32(0.5, v[6]))  # VCC
  k.emit(_vopd(VOPDOp.V_DUAL_CNDMASK_B32, VOPDOp.V_DUAL_CNDMASK_B32))

@microbench(name="mb_vopd_vcc_within_pair", category="vopd")
def _mb_vopd_vcc_within_pair(k: "Kernel") -> None:
  # VOPD#1's X lane writes VCC via an add-with-carry-out operation; VOPD#2 is
  # cndmask pair that reads VCC — cross-VOPD VCC RAW.  V_DUAL_ADD_NC_U32 does
  # NOT write VCC, so use a v_cmp VALU packet in between to emulate VCC
  # production mid-chain.  (There is no VOPD opcode that writes VCC directly
  # in RDNA3; this is closest to what compilers emit.)
  _seed_vopd_ext(k)
  k.emit(v_cmp_gt_f32_e32(0.5, v[6]))                  # set VCC
  k.emit(_vopd(VOPDOp.V_DUAL_CNDMASK_B32, VOPDOp.V_DUAL_CNDMASK_B32))  # reads VCC
  k.emit(v_cmp_gt_f32_e32(0.5, v[7]))                  # overwrite VCC
  k.emit(_vopd(VOPDOp.V_DUAL_CNDMASK_B32, VOPDOp.V_DUAL_CNDMASK_B32))  # reads new VCC

@microbench(name="mb_vopd_vcc_war", category="vopd")
def _mb_vopd_vcc_war(k: "Kernel") -> None:
  # cndmask VOPD reads VCC, then a later VALU v_cmp writes VCC (WAR).
  _seed_vopd_ext(k)
  k.emit(v_cmp_gt_f32_e32(0.5, v[6]))
  k.emit(_vopd(VOPDOp.V_DUAL_CNDMASK_B32, VOPDOp.V_DUAL_CNDMASK_B32))
  k.emit(v_cmp_gt_f32_e32(0.5, v[7]))                  # VCC WAR

@microbench(name="mb_vopd_bank_conflict_src", category="vopd")
def _mb_vopd_bank_conflict_src(k: "Kernel") -> None:
  # Both opx and opy read from the same VGPR bank (v[0] and v[4] both on
  # bank 0 mod 4).  Expect HW to refuse to dual-issue or split.
  _seed_vopd_ext(k)
  for _ in range(2):
    k.emit(VOPD(
      opx=VOPDOp.V_DUAL_ADD_F32, opy=VOPDOp.V_DUAL_MUL_F32,
      vdstx=v[4],  srcx0=v[0], vsrcx1=v[8],   # srcx0 bank 0
      vdsty=v[5],  srcy0=v[4], vsrcy1=v[9],   # srcy0 bank 0  (conflict)
    ))

@microbench(name="mb_vopd_bank_conflict_dst", category="vopd")
def _mb_vopd_bank_conflict_dst(k: "Kernel") -> None:
  # Dst-parity violation: both vdstx and vdsty even (v[4], v[6]).  Compilers
  # normally prevent this; emit it anyway to probe HW behavior.
  _seed_vopd_ext(k)
  for _ in range(2):
    k.emit(VOPD(
      opx=VOPDOp.V_DUAL_ADD_F32, opy=VOPDOp.V_DUAL_MUL_F32,
      vdstx=v[4],  srcx0=v[7], vsrcx1=v[8],
      vdsty=v[6],  srcy0=v[9], vsrcy1=v[11],  # vdsty=v[6] (even) — parity bad
    ))

@microbench(name="mb_vopd_post_depctr", category="vopd")
def _mb_vopd_post_depctr(k: "Kernel") -> None:
  # VOPD, s_waitcnt_depctr(0xffff), VOPD.  Probes C1 (depctr reset cost).
  _seed_vopd_ext(k)
  k.emit(_vopd(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_MUL_F32, 4, 5, 6, 7, 8, 9))
  k.emit(s_waitcnt_depctr(simm16=0xffff))
  k.emit(_vopd(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_MUL_F32, 10, 11, 12, 13, 14, 15))

@microbench(name="mb_vopd_post_waitcnt_vmcnt", category="vopd")
def _mb_vopd_post_waitcnt_vmcnt(k: "Kernel") -> None:
  # Re-issue a vmem load, wait for it, then a VOPD — tests VMEM-return -> VOPD.
  _seed_vopd_ext(k)
  k.emit(global_load_b32(v[2], v[0], saddr=s[0:1]))
  k.emit(s_waitcnt_vmcnt(sdst=NULL, simm16=0))
  k.emit(_vopd(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_MUL_F32))

@microbench(name="mb_vopd_post_trans", category="vopd")
def _mb_vopd_post_trans(k: "Kernel") -> None:
  # v_exp (trans pipe) -> VOPD.  The VOPD does NOT consume the trans dst,
  # so HW should be able to dual-issue it despite trans occupancy.
  _seed_vopd_ext(k)
  k.emit(v_exp_f32_e32(v[2], v[2]))
  k.emit(_vopd(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_MUL_F32))

@microbench(name="mb_vopd_pre_trans", category="vopd")
def _mb_vopd_pre_trans(k: "Kernel") -> None:
  # VOPD writes v[4] (vdstx); v_exp then reads v[4] — trans RAW on VOPD dst.
  _seed_vopd_ext(k)
  k.emit(_vopd(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_MUL_F32, 4, 5, 6, 7, 8, 9))
  k.emit(v_exp_f32_e32(v[2], v[4]))   # RAW on VOPD-dstx

@microbench(name="mb_vopd_sandwich_trans", category="vopd")
def _mb_vopd_sandwich_trans(k: "Kernel") -> None:
  # VOPD; v_exp (no RAW on VOPD dst); VOPD.  Probes trans mid-pair.
  _seed_vopd_ext(k)
  k.emit(_vopd(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_MUL_F32, 4, 5, 6, 7, 8, 9))
  k.emit(v_exp_f32_e32(v[2], v[2]))   # no dep on v[4]/v[5]
  k.emit(_vopd(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_MUL_F32, 10, 11, 12, 13, 14, 15))

@microbench(name="mb_vopd_lit_chain_n4", category="vopd")
def _mb_vopd_lit_chain_n4(k: "Kernel") -> None:
  # 4 VOPD_LIT back-to-back (V_DUAL_FMAAK_F32 on X, V_DUAL_MOV_B32 on Y).
  # VOPD_LIT -> VOPD_LIT is the 1cy fast path.
  _seed_vopd_ext(k)
  dsts = [(4, 5), (10, 11), (12, 13), (14, 15)]
  for (x, y) in dsts:
    k.emit(VOPD_LIT(
      opx=VOPDOp.V_DUAL_FMAAK_F32, opy=VOPDOp.V_DUAL_MOV_B32,
      vdstx=v[x], srcx0=v[6], vsrcx1=v[8],
      vdsty=v[y], srcy0=v[7], vsrcy1=v[9],
      literal=_f2i(2.5),
    ))

@microbench(name="mb_vopd_lit_then_nonlit", category="vopd")
def _mb_vopd_lit_then_nonlit(k: "Kernel") -> None:
  # VOPD_LIT → plain VOPD: does leaving the LIT fast-path cost anything?
  _seed_vopd_ext(k)
  k.emit(VOPD_LIT(
    opx=VOPDOp.V_DUAL_FMAAK_F32, opy=VOPDOp.V_DUAL_MOV_B32,
    vdstx=v[4], srcx0=v[6], vsrcx1=v[8],
    vdsty=v[5], srcy0=v[7], vsrcy1=v[9],
    literal=_f2i(1.25),
  ))
  k.emit(_vopd(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_MUL_F32, 10, 11, 12, 13, 14, 15))

@microbench(name="mb_vopd_nonlit_then_lit", category="vopd")
def _mb_vopd_nonlit_then_lit(k: "Kernel") -> None:
  # plain VOPD → VOPD_LIT.  Does entering the LIT path cost anything?
  _seed_vopd_ext(k)
  k.emit(_vopd(VOPDOp.V_DUAL_ADD_F32, VOPDOp.V_DUAL_MUL_F32, 4, 5, 6, 7, 8, 9))
  k.emit(VOPD_LIT(
    opx=VOPDOp.V_DUAL_FMAAK_F32, opy=VOPDOp.V_DUAL_MOV_B32,
    vdstx=v[10], srcx0=v[12], vsrcx1=v[14],
    vdsty=v[11], srcy0=v[13], vsrcy1=v[15],
    literal=_f2i(0.75),
  ))


# ═════════════════════════════════════════════════════════════════════════════
# B.5 — Trans pipe interactions (15 kernels)
# Measures trans-trans occupancy, trans-valu RAW, trans-snop interaction, and
# trans-VOPD interleave.
# ═════════════════════════════════════════════════════════════════════════════

@microbench(name="mb_trans_after_trans_0", category="trans")
def _mb_trans_after_trans_0(k: "Kernel") -> None:
  # v_exp; v_log with NO VALU between.  Expect 1,4 (trans occupancy).
  # Use distinct dsts to avoid RAW — pure occupancy probe.
  k.emit(v_exp_f32_e32(v[2], v[2]))
  k.emit(v_log_f32_e32(v[3], v[3]))

@microbench(name="mb_trans_after_trans_1", category="trans")
def _mb_trans_after_trans_1(k: "Kernel") -> None:
  # v_exp; v_add; v_log — 1 VALU between trans ops.
  k.emit(v_exp_f32_e32(v[2], v[2]))
  k.emit(v_add_f32_e32(v[4], 1.0, v[4]))
  k.emit(v_log_f32_e32(v[3], v[3]))

@microbench(name="mb_trans_after_trans_4", category="trans")
def _mb_trans_after_trans_4(k: "Kernel") -> None:
  # v_exp; v_add×4; v_log — 4 VALUs between trans ops (hides pipe occupancy).
  k.emit(v_exp_f32_e32(v[2], v[2]))
  for i in range(4): k.emit(v_add_f32_e32(v[4 + i], 1.0, v[4 + i]))
  k.emit(v_log_f32_e32(v[3], v[3]))

@microbench(name="mb_trans_after_trans_8", category="trans")
def _mb_trans_after_trans_8(k: "Kernel") -> None:
  # v_exp; v_add×8; v_log.
  k.emit(v_exp_f32_e32(v[2], v[2]))
  for i in range(8): k.emit(v_add_f32_e32(v[4 + (i % 8)], 1.0, v[4 + (i % 8)]))
  k.emit(v_log_f32_e32(v[3], v[3]))

@microbench(name="mb_trans_raw_exp_log", category="trans")
def _mb_trans_raw_exp_log(k: "Kernel") -> None:
  # v_exp v[1]; v_log v[1] — same dst, RAW on v[1].
  # Expect HW dt=31 (trans pipeline latency).
  k.emit(v_exp_f32_e32(v[1], v[1]))
  k.emit(v_log_f32_e32(v[1], v[1]))

@microbench(name="mb_trans_raw_valu", category="trans")
def _mb_trans_raw_valu(k: "Kernel") -> None:
  # v_exp v[1]; v_add v[2], v[1], v[1] — trans -> VALU RAW.
  k.emit(v_exp_f32_e32(v[1], v[1]))
  k.emit(v_add_f32_e32(v[2], v[1], v[1]))

@microbench(name="mb_trans_raw_with_depctr", category="trans")
def _mb_trans_raw_with_depctr(k: "Kernel") -> None:
  # v_exp; s_waitcnt_depctr(0xffff) (drains pending trans); v_add v[1].
  # depctr should absorb the trans latency so the add issues at dt=1.
  k.emit(v_exp_f32_e32(v[1], v[1]))
  k.emit(s_waitcnt_depctr(simm16=0xffff))
  k.emit(v_add_f32_e32(v[1], 1.0, v[1]))

@microbench(name="mb_trans_then_salu", category="trans")
def _mb_trans_then_salu(k: "Kernel") -> None:
  # v_sqrt; s_waitcnt (empty); s_mov — trans -> SALU cost.
  k.emit(v_sqrt_f32_e32(v[1], v[1]))
  k.emit(s_waitcnt(simm16=0))
  k.emit(s_mov_b32(s[4], 0))

@microbench(name="mb_trans_then_snop", category="trans")
def _mb_trans_then_snop(k: "Kernel") -> None:
  # v_sqrt; s_waitcnt; s_nop(15); v_add.  Diagnoses B3 (first s_nop post-trans).
  k.emit(v_sqrt_f32_e32(v[1], v[1]))
  k.emit(s_waitcnt(simm16=0))
  k.emit(s_nop(simm16=15))
  k.emit(v_add_f32_e32(v[2], 1.0, v[2]))

@microbench(name="mb_trans_chain4_then_snop", category="trans")
def _mb_trans_chain4_then_snop(k: "Kernel") -> None:
  # 4-trans chain, then waitcnt, then s_nop(15) — same probe as above but with
  # trans pipe fully warm.
  for _ in range(4): k.emit(v_exp_f32_e32(v[1], v[1]))
  k.emit(s_waitcnt(simm16=0))
  k.emit(s_nop(simm16=15))
  k.emit(v_add_f32_e32(v[2], 1.0, v[2]))

@microbench(name="mb_trans_pair_same_op", category="trans")
def _mb_trans_pair_same_op(k: "Kernel") -> None:
  # v_exp v[2]; v_exp v[3] — same op, distinct dsts (no RAW).
  # Expect 1,4 — pure trans-pipe occupancy.
  k.emit(v_exp_f32_e32(v[2], v[2]))
  k.emit(v_exp_f32_e32(v[3], v[3]))

@microbench(name="mb_trans_pair_diff_op", category="trans")
def _mb_trans_pair_diff_op(k: "Kernel") -> None:
  # v_exp v[2]; v_log v[3] — different trans ops, no RAW.
  # Expect 1,4 — trans pipeline is a single shared resource.
  k.emit(v_exp_f32_e32(v[2], v[2]))
  k.emit(v_log_f32_e32(v[3], v[3]))

@microbench(name="mb_trans_nested_valu", category="trans")
def _mb_trans_nested_valu(k: "Kernel") -> None:
  # v_exp; v_fmac; v_log; v_fmac — VALU interleaved between trans ops.
  k.emit(v_exp_f32_e32(v[2], v[2]))
  k.emit(v_fmac_f32_e32(v[4], v[5], v[6]))
  k.emit(v_log_f32_e32(v[3], v[3]))
  k.emit(v_fmac_f32_e32(v[7], v[8], v[9]))

@microbench(name="mb_trans_cold_warm_alternating", category="trans")
def _mb_trans_cold_warm_alternating(k: "Kernel") -> None:
  # Alternating cold/warm trans + VALU:
  #   v_exp (cold); v_exp×4; v_add; v_exp (warm); v_add; v_exp.
  k.emit(v_exp_f32_e32(v[2], v[2]))                # cold
  for _ in range(4): k.emit(v_exp_f32_e32(v[2], v[2]))
  k.emit(v_add_f32_e32(v[4], 1.0, v[4]))
  k.emit(v_exp_f32_e32(v[3], v[3]))                # warm
  k.emit(v_add_f32_e32(v[5], 1.0, v[5]))
  k.emit(v_exp_f32_e32(v[6], v[6]))

@microbench(name="mb_trans_vopd_cross", category="trans")
def _mb_trans_vopd_cross(k: "Kernel") -> None:
  # v_exp; VOPD(mul,mul); v_log — VOPD sandwiched between two trans ops.
  # VOPD runs on the VALU pipe while trans pipe is occupied — tests cross-pipe
  # concurrency.
  _seed_vopd_sources(k)
  k.emit(v_exp_f32_e32(v[2], v[2]))
  k.emit(_vopd(VOPDOp.V_DUAL_MUL_F32, VOPDOp.V_DUAL_MUL_F32))
  k.emit(v_log_f32_e32(v[3], v[3]))


__all__: list[str] = []  # registration happens as a side effect of import
