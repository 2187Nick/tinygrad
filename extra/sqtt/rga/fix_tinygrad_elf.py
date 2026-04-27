#!/usr/bin/env python3
"""Reconstruct a standards-compliant AMDGPU HSACO from tinygrad's minimal ELF.

tinygrad's compile path emits a stripped Elf64 where e_ident, e_ehsize, e_shentsize, and
e_machine are zeroed — its own elf_loader trusts the offsets and skips magic. Standard
parsers (RGA, llvm-objdump) reject it. This script fills in only the fields needed to
make the binary parse cleanly without altering section content/offsets.

Usage:
  python extra/sqtt/rga/fix_tinygrad_elf.py <input.pkl|input.elf> <output.elf>
"""
import sys, struct, pickle, pathlib

ELF_MAGIC = b"\x7fELF"
EM_AMDGPU = 224  # ELF e_machine value for AMDGPU
ELFCLASS64 = 2
ELFDATA2LSB = 1
EV_CURRENT = 1
ELFOSABI_AMDGPU_HSA = 64
ELFABIVERSION_AMDGPU_HSA_V4 = 2  # ABI v4 = e_ident[8] = 2
ET_DYN = 3

# EF_AMDGPU_MACH constants from AMDGPU ABI (LLVM include/llvm/BinaryFormat/ELF.h)
EF_AMDGPU_MACH = {
  "gfx900":  0x02c, "gfx906":  0x02f, "gfx908":  0x030, "gfx90a":  0x03f, "gfx940":  0x046,
  "gfx1010": 0x033, "gfx1011": 0x034, "gfx1012": 0x035, "gfx1030": 0x036, "gfx1031": 0x037,
  "gfx1032": 0x038, "gfx1033": 0x039, "gfx1034": 0x03e, "gfx1035": 0x03d, "gfx1036": 0x045,
  "gfx1100": 0x041, "gfx1101": 0x042, "gfx1102": 0x043, "gfx1103": 0x044,
  "gfx1150": 0x047, "gfx1151": 0x048, "gfx1152": 0x049,
  "gfx1200": 0x04c, "gfx1201": 0x04d,
}

def fix_elf(blob: bytes, target: str = "gfx1100") -> bytes:
  buf = bytearray(blob)
  # e_ident (16 bytes): magic + class + data + version + osabi + osabi-version + padding
  buf[0:16] = (ELF_MAGIC + bytes([ELFCLASS64, ELFDATA2LSB, EV_CURRENT,
                                  ELFOSABI_AMDGPU_HSA, ELFABIVERSION_AMDGPU_HSA_V4]) + b"\x00" * 7)
  struct.pack_into("<H", buf, 16, ET_DYN)             # e_type
  struct.pack_into("<H", buf, 18, EM_AMDGPU)          # e_machine
  struct.pack_into("<I", buf, 20, EV_CURRENT)         # e_version
  if target not in EF_AMDGPU_MACH: raise ValueError(f"unknown AMDGPU target {target!r}")
  struct.pack_into("<I", buf, 48, EF_AMDGPU_MACH[target])  # e_flags = AMDGPU mach id
  struct.pack_into("<H", buf, 52, 64)                 # e_ehsize
  struct.pack_into("<H", buf, 54, 0)                  # e_phentsize (no phs)
  struct.pack_into("<H", buf, 56, 0)                  # e_phnum
  struct.pack_into("<H", buf, 58, 64)                 # e_shentsize (Elf64_Shdr = 64 bytes)
  return bytes(buf)

def load_lib(path: str) -> tuple[bytes, str]:
  if path.endswith('.pkl'):
    d = pickle.load(open(path,'rb'))
    return d['lib'], d.get('target', 'gfx1100')
  return open(path,'rb').read(), 'gfx1100'

if __name__ == '__main__':
  if len(sys.argv) != 3:
    print(__doc__); sys.exit(1)
  src, dst = sys.argv[1], sys.argv[2]
  lib, target = load_lib(src)
  out = fix_elf(lib, target)
  pathlib.Path(dst).write_bytes(out)
  print(f"wrote {dst}  size={len(out)}  target={target}")
