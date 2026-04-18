#!/usr/bin/env python3
"""Batch B.1 — LDS bank-conflict sweep (48 kernels).

Targets the RDNA3 cycle-accurate emulator bounty. Probes the LDS bank-arbiter
timing model across the product of:
  - op kind:  {rd, wr}              (ds_load / ds_store)
  - op width: {b32, b64, b128}      (32/64/128-bit LDS transactions)
  - address pattern: {bank1, bank2, bank4, bank8, bank16, stride4, stride8, random}

= 2 * 3 * 8 = 48 microbenches, all following the `mb_lds_{rd|wr}_{b32|b64|b128}_{pattern}`
naming convention (see MICROBENCH_TAXONOMY.md §B.1).

Address-pattern math (RDNA3 has 16 LDS banks, 4B per bank):
  v[0] after the standard prologue holds `tid*4` (byte offset).
  We compute a per-thread LDS byte offset into v[100]:
    bank1   — every thread maps to byte 0           → v[100] = 0
    bank2   — (tid & 1) * 4                          → v[100] = v[0] & 0x4
    bank4   — (tid & 3) * 4                          → v[100] = v[0] & 0xC
    bank8   — (tid & 7) * 4                          → v[100] = v[0] & 0x1C
    bank16  — (tid & 15) * 4                         → v[100] = v[0] & 0x3C   (no conflict)
    stride4 — tid * 16                               → v[100] = v[0] << 2
    stride8 — tid * 32                               → v[100] = v[0] << 3
    random  — (tid*13 + 7) & 0xFC                    → v[100] non-linear covers many banks

Every kernel allocates a 4 KB LDS slab (`extra["lds_size"]=4096`) so even the
stride-8 pattern (max byte offset = 63*32 = 2016) stays in bounds.

After emitting the address VGPR and the LDS op, we drain lgkmcnt and, for reads,
fold the loaded value into v[1] so the standard epilogue's global_store has a
real data dependency on the LDS op (keeps the compiler/scheduler from dropping it).
"""
from __future__ import annotations

from tinygrad.renderer.amd.dsl import s, v, NULL  # noqa: F401
from tinygrad.runtime.autogen.amd.rdna3.ins import (
  s_load_b64, s_waitcnt_lgkmcnt, s_waitcnt_vmcnt,
  v_lshlrev_b32_e32, v_lshrrev_b32_e32, v_and_b32_e32, v_or_b32_e32,
  v_mov_b32_e32, v_add_f32_e32, v_add_nc_u32_e32,
  v_mul_u32_u24_e32,
  global_load_b32,
  ds_store_b32, ds_store_b64, ds_store_b128,
  ds_load_b32, ds_load_b64, ds_load_b128,
)

from extra.sqtt.rgp.microbench import microbench, Kernel

# ── Address-pattern emitters ────────────────────────────────────────────────
# Each writes the per-thread LDS byte offset into v[100].
# Precondition: v[0] = tid*4 (set by the standard prologue's v_lshlrev_b32_e32).
# The VGPR index 100 is far above anything the prologue/epilogue uses (v[0..9]).

_ADDR_VREG = 100  # scratch VGPR for the computed LDS byte offset

def _emit_addr_bank1(k: "Kernel") -> None:
  # All threads hit byte 0 — full broadcast.
  k.emit(v_mov_b32_e32(v[_ADDR_VREG], 0))

def _emit_addr_bank2(k: "Kernel") -> None:
  # (tid & 1) * 4 = (tid*4) & 4 = v[0] & 0x4
  k.emit(v_and_b32_e32(v[_ADDR_VREG], 0x4, v[0]))

def _emit_addr_bank4(k: "Kernel") -> None:
  # (tid & 3) * 4 = v[0] & 0xC
  k.emit(v_and_b32_e32(v[_ADDR_VREG], 0xC, v[0]))

def _emit_addr_bank8(k: "Kernel") -> None:
  # (tid & 7) * 4 = v[0] & 0x1C
  k.emit(v_and_b32_e32(v[_ADDR_VREG], 0x1C, v[0]))

def _emit_addr_bank16(k: "Kernel") -> None:
  # (tid & 15) * 4 = v[0] & 0x3C — no-conflict baseline (16 distinct banks).
  k.emit(v_and_b32_e32(v[_ADDR_VREG], 0x3C, v[0]))

def _emit_addr_stride4(k: "Kernel") -> None:
  # tid * 16 = v[0] << 2
  k.emit(v_lshlrev_b32_e32(v[_ADDR_VREG], 2, v[0]))

def _emit_addr_stride8(k: "Kernel") -> None:
  # tid * 32 = v[0] << 3
  k.emit(v_lshlrev_b32_e32(v[_ADDR_VREG], 3, v[0]))

def _emit_addr_random(k: "Kernel") -> None:
  # (tid*13 + 7) & 0xFC — non-linear coverage that hits many banks.
  # tid = v[0] >> 2, so compute via ((v[0] >> 2) * 13 + 7) * 4 = (tid*13+7)*4
  # We actually want the byte offset = ((tid*13 + 7) mod something) * 4.
  # Simpler: output = (tid*13 + 7) & 0xFC — keep low 8 bits, align to 4.
  # Sequence:
  #   v[101] = v[0] >> 2           ; v[101] = tid
  #   v[101] = v[101] * 13         ; using v_mul_u32_u24 (24-bit inputs ok for tid<1024)
  #   v[101] = v[101] + 7          ; v_add_nc_u32_e32
  #   v[100] = v[101] & 0xFC
  k.emit(v_lshrrev_b32_e32(v[101], 2, v[0]))
  k.emit(v_mul_u32_u24_e32(v[101], 13, v[101]))
  k.emit(v_add_nc_u32_e32(v[101], 7, v[101]))
  k.emit(v_and_b32_e32(v[_ADDR_VREG], 0xFC, v[101]))

_ADDR_PATTERNS: dict[str, callable] = {
  "bank1":   _emit_addr_bank1,
  "bank2":   _emit_addr_bank2,
  "bank4":   _emit_addr_bank4,
  "bank8":   _emit_addr_bank8,
  "bank16":  _emit_addr_bank16,
  "stride4": _emit_addr_stride4,
  "stride8": _emit_addr_stride8,
  "random":  _emit_addr_random,
}

# ── Width-specific LDS op emitters ──────────────────────────────────────────
# Reads: drop into a dedicated VGPR range (v[10..13]) then fold into v[1].
# Writes: seed a payload in v[3..6] (avoiding v[0]=offset, v[1]=warm load dest)
# then ds_store. Drain lgkmcnt after so the trace timestamp is unambiguous.

def _emit_read_b32(k: "Kernel") -> None:
  k.emit(ds_load_b32(vdst=v[10], addr=v[_ADDR_VREG]))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))
  # fold into v[1] so the epilogue's store depends on the LDS value
  k.emit(v_add_f32_e32(v[1], 1.0, v[10]))

def _emit_read_b64(k: "Kernel") -> None:
  k.emit(ds_load_b64(vdst=v[10:11], addr=v[_ADDR_VREG]))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))
  k.emit(v_add_f32_e32(v[1], 1.0, v[10]))

def _emit_read_b128(k: "Kernel") -> None:
  k.emit(ds_load_b128(vdst=v[10:13], addr=v[_ADDR_VREG]))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))
  k.emit(v_add_f32_e32(v[1], 1.0, v[10]))

def _emit_write_b32(k: "Kernel") -> None:
  k.emit(ds_store_b32(addr=v[_ADDR_VREG], data0=v[1]))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))

def _emit_write_b64(k: "Kernel") -> None:
  # Seed a 2-VGPR payload.
  k.emit(v_mov_b32_e32(v[3], 1.0))
  k.emit(v_mov_b32_e32(v[4], 2.0))
  k.emit(ds_store_b64(addr=v[_ADDR_VREG], data0=v[3:4]))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))

def _emit_write_b128(k: "Kernel") -> None:
  # Seed a 4-VGPR payload.
  for i in range(4):
    k.emit(v_mov_b32_e32(v[3+i], float(i+1)))
  k.emit(ds_store_b128(addr=v[_ADDR_VREG], data0=v[3:6]))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))

_OP_EMITTERS: dict[tuple[str, str], callable] = {
  ("rd", "b32"):  _emit_read_b32,
  ("rd", "b64"):  _emit_read_b64,
  ("rd", "b128"): _emit_read_b128,
  ("wr", "b32"):  _emit_write_b32,
  ("wr", "b64"):  _emit_write_b64,
  ("wr", "b128"): _emit_write_b128,
}

# ── Body factory + registration ─────────────────────────────────────────────

def _make_body(kind: str, width: str, pattern: str):
  """Build a microbench body: emit the LDS-address VGPR, then the op."""
  addr_emit = _ADDR_PATTERNS[pattern]
  op_emit = _OP_EMITTERS[(kind, width)]
  def _body(k: "Kernel") -> None:
    addr_emit(k)
    op_emit(k)
  _body.__name__ = f"_mb_lds_{kind}_{width}_{pattern}_body"
  return _body

# Register all 48 kernels. 4 KB LDS covers stride8 (max byte = 63*32 = 2016).
for _kind in ("rd", "wr"):
  for _width in ("b32", "b64", "b128"):
    for _pattern in _ADDR_PATTERNS:
      _name = f"mb_lds_{_kind}_{_width}_{_pattern}"
      microbench(
        name=_name,
        category="lds_bank",
        lds_size=4096,
      )(_make_body(_kind, _width, _pattern))
