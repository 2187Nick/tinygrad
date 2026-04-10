#!/usr/bin/env python3
"""Diagnostic: decode hardware SQTT pkl files and print all REG packets to verify subop values for COMPUTE_PGM_LO/HI."""
import pickle, sys
from pathlib import Path
from tinygrad.renderer.amd.sqtt import decode, REG

def main():
  examples = Path(__file__).parent / "examples" / "gfx1100"
  for pkl_path in sorted(examples.glob("*.pkl")):
    with pkl_path.open("rb") as f: profile = pickle.load(f)
    for i, ev in enumerate(profile):
      blob = getattr(ev, "blob", None)
      if not isinstance(blob, bytes) or len(blob) == 0: continue
      pkts = list(decode(blob))
      reg_pkts = [p for p in pkts if isinstance(p, REG)]
      if not reg_pkts: continue
      print(f"\n{pkl_path.name} event {i}: {len(reg_pkts)} REG packets")
      for rp in reg_pkts:
        if rp.subop in (0xC, 0xD) and rp.is_config:
          name = "COMPUTE_PGM_LO" if rp.subop == 0xC else "COMPUTE_PGM_HI"
          print(f"  *** {name}: subop=0x{rp.subop:04X} hi_byte=0x{rp.hi_byte:02X} val32=0x{rp.val32:08X}")
      break  # first blob per pkl

if __name__ == "__main__":
  main()
