#!/usr/bin/env python3
"""Batch G — Targeted gap-fill probes.

~40 kernels for instruction families not yet characterized by Batches A-F:
  G.1 (5) — SMEM load sizes (s_load_b32 / b64 / b128 / b256 / b512)
  G.2 (6) — DS atomics and bitwise LDS ops
  G.3 (9) — VALU ops we haven't probed: v_bfe, v_sad, v_lerp, v_cubeid,
            v_min/max_u32, v_cvt, v_ashrrev (shift arith)
  G.4 (6) — SALU variants: s_and/or/xor/bfe/mul chains
  G.5 (5) — F16 non-packed (v_add/mul/fmac/min/max_f16_e32)
  G.6 (5) — Integer add/sub/shift variants (v_add_nc_u32, v_sub_nc_u32,
            v_lshrrev, v_ashrrev, v_and_b32)
  G.7 (4) — SALU compare + branch variants (s_cmp_eq/lt/lt_u, s_cbranch_execz)

Standard microbench prologue/epilogue. All kernels use v[10+] to stay clear
of prologue-used v[0..1].
"""
from __future__ import annotations
import struct

from tinygrad.renderer.amd.dsl import s, v, NULL  # noqa: F401
from tinygrad.runtime.autogen.amd.rdna3.ins import (
  v_mov_b32_e32, v_add_f32_e32,
  v_bfe_u32, v_bfe_i32, v_sad_u32, v_lerp_u8, v_cubeid_f32,
  v_min_u32_e32, v_max_u32_e32, v_min_i32_e32, v_max_i32_e32,
  v_cvt_i32_f32_e32, v_cvt_u32_f32_e32,
  v_add_f16_e32, v_mul_f16_e32, v_fmac_f16_e32, v_min_f16_e32, v_max_f16_e32,
  v_add_nc_u32_e32, v_sub_nc_u32_e32, v_lshrrev_b32_e32, v_lshlrev_b32_e32,
  v_ashrrev_i32_e32, v_and_b32_e32, v_or_b32_e32, v_xor_b32_e32,
  s_load_b32, s_load_b64, s_load_b128, s_load_b256, s_load_b512,
  s_and_b32, s_or_b32, s_xor_b32, s_bfe_u32, s_mul_i32, s_add_u32, s_mov_b32,
  s_cmp_eq_i32, s_cmp_lt_i32, s_cmp_lt_u32,
  s_cbranch_execz, s_cbranch_vccz, s_cbranch_scc1,
  s_waitcnt_lgkmcnt, s_waitcnt_vmcnt, s_nop,
  ds_add_u32, ds_min_u32, ds_max_u32, ds_and_b32, ds_or_b32, ds_xor_b32,
  ds_load_b32, ds_store_b32,
  global_load_b32, global_store_b32,
)

from extra.sqtt.rgp.microbench import microbench, Kernel  # noqa: F401


# ═════════════════════════════════════════════════════════════════════════════
# G.1 — SMEM load sizes (5 kernels)
# ═════════════════════════════════════════════════════════════════════════════
# Note: custom prologue_builder per SMEM kernel because standard prologue already
# issues s_load_b64. We want to study the size-latency relationship.

def _smem_prologue_simple(k: Kernel) -> None:
  """Prologue for SMEM probes: just the initial s_load_b64 + waitcnt + offset setup."""
  k.emit(s_load_b64(s[0:1], s[0:1], soffset=NULL))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))
  k.emit(v_lshlrev_b32_e32(v[0], 2, v[0]))
  k.emit(global_load_b32(v[1], v[0], saddr=s[0:1]))
  k.emit(s_waitcnt_vmcnt(sdst=NULL, simm16=0))

@microbench(name="mb_g1_smem_b32", category="G.1-smem-sizes", prologue=_smem_prologue_simple)
def _g1_smem_b32(k: Kernel) -> None:
  """Single s_load_b32 with waitcnt drain."""
  k.emit(s_load_b32(s[10], s[0:1], soffset=NULL))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))

@microbench(name="mb_g1_smem_b64", category="G.1-smem-sizes", prologue=_smem_prologue_simple)
def _g1_smem_b64(k: Kernel) -> None:
  k.emit(s_load_b64(s[10:11], s[0:1], soffset=NULL))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))

@microbench(name="mb_g1_smem_b128", category="G.1-smem-sizes", prologue=_smem_prologue_simple)
def _g1_smem_b128(k: Kernel) -> None:
  k.emit(s_load_b128(s[12:15], s[0:1], soffset=NULL))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))

@microbench(name="mb_g1_smem_b256", category="G.1-smem-sizes", prologue=_smem_prologue_simple)
def _g1_smem_b256(k: Kernel) -> None:
  k.emit(s_load_b256(s[16:23], s[0:1], soffset=NULL))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))

@microbench(name="mb_g1_smem_b512", category="G.1-smem-sizes", prologue=_smem_prologue_simple)
def _g1_smem_b512(k: Kernel) -> None:
  k.emit(s_load_b512(s[24:39], s[0:1], soffset=NULL))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))


# ═════════════════════════════════════════════════════════════════════════════
# G.2 — DS atomics / bitwise LDS ops (6 kernels)
# ═════════════════════════════════════════════════════════════════════════════
# These reserve LDS in the kernel. Use lds_size=256 via @microbench(lds_size=256).

@microbench(name="mb_g2_ds_add_u32_n2", category="G.2-ds-atomic", lds_size=256)
def _g2_ds_add_u32_n2(k: Kernel) -> None:
  """2 consecutive ds_add_u32 atomics to the same LDS address."""
  k.emit(v_mov_b32_e32(v[10], 0))      # LDS offset in v[10] (byte 0)
  k.emit(v_mov_b32_e32(v[11], 1))      # data
  k.emit(ds_add_u32(addr=v[10], data0=v[11], offset0=0))
  k.emit(ds_add_u32(addr=v[10], data0=v[11], offset0=0))

@microbench(name="mb_g2_ds_min_u32_n1", category="G.2-ds-atomic", lds_size=256)
def _g2_ds_min_u32_n1(k: Kernel) -> None:
  k.emit(v_mov_b32_e32(v[10], 0))
  k.emit(v_mov_b32_e32(v[11], 1))
  k.emit(ds_min_u32(addr=v[10], data0=v[11], offset0=0))

@microbench(name="mb_g2_ds_max_u32_n1", category="G.2-ds-atomic", lds_size=256)
def _g2_ds_max_u32_n1(k: Kernel) -> None:
  k.emit(v_mov_b32_e32(v[10], 0))
  k.emit(v_mov_b32_e32(v[11], 1))
  k.emit(ds_max_u32(addr=v[10], data0=v[11], offset0=0))

@microbench(name="mb_g2_ds_and_b32_n1", category="G.2-ds-atomic", lds_size=256)
def _g2_ds_and_b32_n1(k: Kernel) -> None:
  k.emit(v_mov_b32_e32(v[10], 0))
  k.emit(v_mov_b32_e32(v[11], 1))
  k.emit(ds_and_b32(addr=v[10], data0=v[11], offset0=0))

@microbench(name="mb_g2_ds_or_b32_n1", category="G.2-ds-atomic", lds_size=256)
def _g2_ds_or_b32_n1(k: Kernel) -> None:
  k.emit(v_mov_b32_e32(v[10], 0))
  k.emit(v_mov_b32_e32(v[11], 1))
  k.emit(ds_or_b32(addr=v[10], data0=v[11], offset0=0))

@microbench(name="mb_g2_ds_xor_b32_n1", category="G.2-ds-atomic", lds_size=256)
def _g2_ds_xor_b32_n1(k: Kernel) -> None:
  k.emit(v_mov_b32_e32(v[10], 0))
  k.emit(v_mov_b32_e32(v[11], 1))
  k.emit(ds_xor_b32(addr=v[10], data0=v[11], offset0=0))


# ═════════════════════════════════════════════════════════════════════════════
# G.3 — Under-probed VALU ops (9 kernels)
# ═════════════════════════════════════════════════════════════════════════════

@microbench(name="mb_g3_bfe_u32_n4", category="G.3-valu-misc")
def _g3_bfe_u32_n4(k: Kernel) -> None:
  """v_bfe_u32 3-src: (data, offset, width). RAW chain on v[10]."""
  k.emit(v_bfe_u32(v[10], v[1], 0, 8))
  for _ in range(3): k.emit(v_bfe_u32(v[10], v[10], 0, 8))

@microbench(name="mb_g3_bfe_i32_n4", category="G.3-valu-misc")
def _g3_bfe_i32_n4(k: Kernel) -> None:
  k.emit(v_bfe_i32(v[10], v[1], 0, 8))
  for _ in range(3): k.emit(v_bfe_i32(v[10], v[10], 0, 8))

@microbench(name="mb_g3_sad_u32_n4", category="G.3-valu-misc")
def _g3_sad_u32_n4(k: Kernel) -> None:
  """v_sad_u32 3-src: sum-of-absolute-differences. RAW chain."""
  k.emit(v_sad_u32(v[10], v[1], v[2], v[3]))
  for _ in range(3): k.emit(v_sad_u32(v[10], v[10], v[2], v[3]))

@microbench(name="mb_g3_min_u32_n4", category="G.3-valu-misc")
def _g3_min_u32_n4(k: Kernel) -> None:
  k.emit(v_min_u32_e32(v[10], v[1], v[2]))
  for _ in range(3): k.emit(v_min_u32_e32(v[10], v[10], v[2]))

@microbench(name="mb_g3_max_u32_n4", category="G.3-valu-misc")
def _g3_max_u32_n4(k: Kernel) -> None:
  k.emit(v_max_u32_e32(v[10], v[1], v[2]))
  for _ in range(3): k.emit(v_max_u32_e32(v[10], v[10], v[2]))

@microbench(name="mb_g3_min_i32_n4", category="G.3-valu-misc")
def _g3_min_i32_n4(k: Kernel) -> None:
  k.emit(v_min_i32_e32(v[10], v[1], v[2]))
  for _ in range(3): k.emit(v_min_i32_e32(v[10], v[10], v[2]))

@microbench(name="mb_g3_cvt_i32_f32_n4", category="G.3-valu-misc")
def _g3_cvt_i32_f32_n4(k: Kernel) -> None:
  """v_cvt_i32_f32 chain. Converts — probably 1cy pipelined."""
  for _ in range(4): k.emit(v_cvt_i32_f32_e32(v[10], v[10]))

@microbench(name="mb_g3_cvt_u32_f32_n4", category="G.3-valu-misc")
def _g3_cvt_u32_f32_n4(k: Kernel) -> None:
  for _ in range(4): k.emit(v_cvt_u32_f32_e32(v[10], v[10]))

@microbench(name="mb_g3_ashrrev_chain_n4", category="G.3-valu-misc")
def _g3_ashrrev_chain_n4(k: Kernel) -> None:
  """Arithmetic right shift chain."""
  for _ in range(4): k.emit(v_ashrrev_i32_e32(v[10], 1, v[10]))


# ═════════════════════════════════════════════════════════════════════════════
# G.4 — SALU ops (6 kernels)
# ═════════════════════════════════════════════════════════════════════════════

@microbench(name="mb_g4_s_and_b32_n4", category="G.4-salu")
def _g4_s_and_n4(k: Kernel) -> None:
  """SALU and chain — RAW on s[20]."""
  k.emit(s_mov_b32(s[20], 0xff))
  for _ in range(4): k.emit(s_and_b32(s[20], s[20], 0xff))

@microbench(name="mb_g4_s_or_b32_n4", category="G.4-salu")
def _g4_s_or_n4(k: Kernel) -> None:
  k.emit(s_mov_b32(s[20], 0))
  for _ in range(4): k.emit(s_or_b32(s[20], s[20], 0xff))

@microbench(name="mb_g4_s_xor_b32_n4", category="G.4-salu")
def _g4_s_xor_n4(k: Kernel) -> None:
  k.emit(s_mov_b32(s[20], 0))
  for _ in range(4): k.emit(s_xor_b32(s[20], s[20], 0xff))

@microbench(name="mb_g4_s_bfe_u32_n4", category="G.4-salu")
def _g4_s_bfe_n4(k: Kernel) -> None:
  k.emit(s_mov_b32(s[20], 0xff00))
  for _ in range(4): k.emit(s_bfe_u32(s[20], s[20], 0x80008))

@microbench(name="mb_g4_s_mul_i32_n4", category="G.4-salu")
def _g4_s_mul_n4(k: Kernel) -> None:
  """Scalar integer multiply chain. Note: SALU mul is typically multi-cycle."""
  k.emit(s_mov_b32(s[20], 3))
  for _ in range(4): k.emit(s_mul_i32(s[20], s[20], 2))

@microbench(name="mb_g4_s_add_u32_n8", category="G.4-salu")
def _g4_s_add_n8(k: Kernel) -> None:
  """Longer SALU add chain to probe chain-length effect."""
  k.emit(s_mov_b32(s[20], 0))
  for _ in range(8): k.emit(s_add_u32(s[20], s[20], 1))


# ═════════════════════════════════════════════════════════════════════════════
# G.5 — F16 non-packed (5 kernels)
# ═════════════════════════════════════════════════════════════════════════════

@microbench(name="mb_g5_vadd_f16_n4", category="G.5-f16")
def _g5_vadd_f16_n4(k: Kernel) -> None:
  for _ in range(4): k.emit(v_add_f16_e32(v[10], v[10], v[2]))

@microbench(name="mb_g5_vmul_f16_n4", category="G.5-f16")
def _g5_vmul_f16_n4(k: Kernel) -> None:
  for _ in range(4): k.emit(v_mul_f16_e32(v[10], v[10], v[2]))

@microbench(name="mb_g5_vfmac_f16_n4", category="G.5-f16")
def _g5_vfmac_f16_n4(k: Kernel) -> None:
  for _ in range(4): k.emit(v_fmac_f16_e32(v[10], v[2], v[3]))

@microbench(name="mb_g5_vmin_f16_n4", category="G.5-f16")
def _g5_vmin_f16_n4(k: Kernel) -> None:
  for _ in range(4): k.emit(v_min_f16_e32(v[10], v[10], v[2]))

@microbench(name="mb_g5_vmax_f16_n4", category="G.5-f16")
def _g5_vmax_f16_n4(k: Kernel) -> None:
  for _ in range(4): k.emit(v_max_f16_e32(v[10], v[10], v[2]))


# ═════════════════════════════════════════════════════════════════════════════
# G.6 — Integer add/sub/shift variants (5 kernels)
# ═════════════════════════════════════════════════════════════════════════════

@microbench(name="mb_g6_vadd_nc_u32_n4", category="G.6-int-ops")
def _g6_vadd_nc_u32_n4(k: Kernel) -> None:
  """v_add_nc_u32 (no-carry int add) chain."""
  for _ in range(4): k.emit(v_add_nc_u32_e32(v[10], 1, v[10]))

@microbench(name="mb_g6_vsub_nc_u32_n4", category="G.6-int-ops")
def _g6_vsub_nc_u32_n4(k: Kernel) -> None:
  """v_sub_nc_u32 VOP2 requires src1 to be a VGPR (vsrc). Pass v[2] as rhs."""
  for _ in range(4): k.emit(v_sub_nc_u32_e32(v[10], 1, v[10]))

@microbench(name="mb_g6_vlshr_b32_n4", category="G.6-int-ops")
def _g6_vlshr_n4(k: Kernel) -> None:
  for _ in range(4): k.emit(v_lshrrev_b32_e32(v[10], 1, v[10]))

@microbench(name="mb_g6_vand_b32_n4", category="G.6-int-ops")
def _g6_vand_n4(k: Kernel) -> None:
  for _ in range(4): k.emit(v_and_b32_e32(v[10], v[10], v[2]))

@microbench(name="mb_g6_vor_b32_n4", category="G.6-int-ops")
def _g6_vor_n4(k: Kernel) -> None:
  for _ in range(4): k.emit(v_or_b32_e32(v[10], v[10], v[2]))


# ═════════════════════════════════════════════════════════════════════════════
# G.7 — SALU compare + branch variants (4 kernels)
# ═════════════════════════════════════════════════════════════════════════════

@microbench(name="mb_g7_s_cmp_eq_branch", category="G.7-cmp-branch")
def _g7_s_cmp_eq_branch(k: Kernel) -> None:
  """s_cmp_eq_i32 → s_cbranch_scc1 pattern (taken-branch writes SCC)."""
  k.emit(s_mov_b32(s[20], 5))
  k.emit(s_cmp_eq_i32(s[20], 5))
  k.emit(s_cbranch_scc1(simm16=1))  # branch forward 1 dword (effectively nop)
  k.emit(s_nop(0))

@microbench(name="mb_g7_s_cmp_lt_branch", category="G.7-cmp-branch")
def _g7_s_cmp_lt_branch(k: Kernel) -> None:
  k.emit(s_mov_b32(s[20], 3))
  k.emit(s_cmp_lt_i32(s[20], 5))
  k.emit(s_cbranch_scc1(simm16=1))
  k.emit(s_nop(0))

@microbench(name="mb_g7_s_cmp_lt_u_chain_n3", category="G.7-cmp-branch")
def _g7_s_cmp_lt_u_chain(k: Kernel) -> None:
  """3 s_cmp_lt_u32 writing SCC — tests SCC-write-to-write serialization."""
  k.emit(s_mov_b32(s[20], 3))
  k.emit(s_cmp_lt_u32(s[20], 5))
  k.emit(s_cmp_lt_u32(s[20], 6))
  k.emit(s_cmp_lt_u32(s[20], 7))

@microbench(name="mb_g7_s_cbranch_execz", category="G.7-cmp-branch")
def _g7_s_cbranch_execz(k: Kernel) -> None:
  """s_cbranch_execz (branch when EXEC==0) — different scalar dep path."""
  k.emit(s_cbranch_execz(simm16=1))
  k.emit(s_nop(0))
