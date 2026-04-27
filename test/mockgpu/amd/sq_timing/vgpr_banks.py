"""Static VGPR-bank conflict analyzer for VOPD pairs.

RDNA3 has 4 VGPR banks per SIMD with `bank_of(vN) = vN % 4`. A VOPD pair
issues two ops in one cycle and reads both X-side and Y-side source operands
in parallel. When two reads land on the same bank, the issue stalls one cycle.

This module provides:
  - `bank_of(reg_num)`            : trivial bank index helper.
  - `inst_has_bank_conflict(inst)`: True if a VOPD instruction's X/Y source
                                    VGPRs collide on any bank. Only counts
                                    actual VGPR reads (offset >= 256). SGPR
                                    reads (0..107) and inline constants
                                    (240..255) and literals don't conflict.
  - `vgpr_bank_conflicts(lib, target) -> dict[pc, cycle_penalty]`:
        decodes a captured ELF and returns {pc: 1 if conflict else 0}
        for every VOPD/VOPD_LIT instruction in the .text section.
"""
from __future__ import annotations

from tinygrad.runtime.autogen.amd.rdna3 import ins as ir3
from tinygrad.runtime.autogen.amd.rdna4 import ins as ir4
from tinygrad.runtime.autogen.amd.rdna3.operands import OPERANDS as OPERANDS_RDNA3
from tinygrad.runtime.autogen.amd.rdna4.operands import OPERANDS as OPERANDS_RDNA4

VOPD_TYPES = (ir3.VOPD, ir4.VOPD)
VOPD_LIT_TYPES = (ir3.VOPD_LIT, ir4.VOPD_LIT)

NUM_BANKS = 4
VGPR_OFFSET_BASE = 256


def bank_of(reg_num: int) -> int:
  return reg_num % NUM_BANKS


def _vgpr_num(field) -> int | None:
  o = getattr(field, 'offset', -1)
  if o is None: return None
  if 256 <= o <= 511: return o - VGPR_OFFSET_BASE
  return None


def _operand_keys_for_op(op) -> set[str]:
  spec = OPERANDS_RDNA3.get(op)
  if spec is None: spec = OPERANDS_RDNA4.get(op)
  return set(spec.keys()) if spec else set()


def _vopd_side_vgprs(inst, side: str, op) -> list[int]:
  keys = _operand_keys_for_op(op)
  out: list[int] = []
  if 'srcy0' in keys and hasattr(inst, f'src{side}0'):
    n = _vgpr_num(getattr(inst, f'src{side}0'))
    if n is not None: out.append(n)
  if 'vsrcy1' in keys and hasattr(inst, f'vsrc{side}1'):
    n = _vgpr_num(getattr(inst, f'vsrc{side}1'))
    if n is not None: out.append(n)
  return out


def inst_has_bank_conflict(inst) -> bool:
  if not isinstance(inst, VOPD_TYPES): return False
  opx = getattr(inst, 'opx', None)
  opy = getattr(inst, 'opy', None)
  if opx is None or opy is None: return False
  x_vgprs = _vopd_side_vgprs(inst, 'x', opx)
  y_vgprs = _vopd_side_vgprs(inst, 'y', opy)
  if not x_vgprs or not y_vgprs: return False
  x_banks = {bank_of(n) for n in x_vgprs}
  y_banks = {bank_of(n) for n in y_vgprs}
  return bool(x_banks & y_banks)


def vgpr_bank_conflicts(lib: bytes, target: str) -> dict[int, int]:
  from tinygrad.viz.serve import amd_decode
  table = amd_decode(lib, target)
  out: dict[int, int] = {}
  for pc, inst in table.items():
    if isinstance(inst, VOPD_TYPES):
      out[pc] = 1 if inst_has_bank_conflict(inst) else 0
  return out
