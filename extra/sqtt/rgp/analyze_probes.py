#!/usr/bin/env python3
"""Analyze the parametric probe captures to extract cycle functions.

Run after: sudo PROBE=1 ... rigorous_hw_test.py --capture

For each probe family this prints the per-token (idx, dt, inst) table for each
wave, plus the HW vs EMU deltas. That lets us read off the cycle function
directly (e.g. for nop_chain_nN: does the last nop always cost 20?).
"""
from __future__ import annotations
import os, sys, pickle, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
CAPTURE_DIR = ROOT / "extra" / "sqtt" / "captures" / "rigorous"

PROBE_FAMILIES = {
  "cold_start":  ["probe_cold_start_n2", "probe_cold_start_n4", "probe_cold_start_n8"],
  "nop_chain":   ["probe_nop_chain_n1", "probe_nop_chain_n3", "probe_nop_chain_n5"],
  "store":       ["probe_store_cold", "probe_store_warm"],
  "trans_pair":  ["probe_trans_pair_tight", "probe_trans_pair_spaced"],
  "scalar_beat": ["probe_scalar_beat_p0", "probe_scalar_beat_p1", "probe_scalar_beat_p2", "probe_scalar_beat_p3"],
  "vopd":        ["probe_vopd_chain", "probe_vopd_split", "probe_vopd_nodep"],
}

def load(name):
  p = CAPTURE_DIR / f"{name}.pkl"
  if not p.exists(): return None
  with open(p, "rb") as f: return pickle.load(f)

def fmt_dt(d):
  return f"{d:3d}" if d < 1000 else f"{d//100:3d}×"

def print_trace(name, traces):
  print(f"\n── {name} ──")
  if traces is None: print("  (no capture)"); return
  for wid in sorted(traces.keys()):
    print(f"  wave {wid}:")
    prev_t = None
    for idx, (pc, t, tt, inst) in enumerate(traces[wid]):
      dt = (t - prev_t) if prev_t is not None else 0
      prev_t = t
      if tt in ("INST", "VALUINST", "IMMEDIATE"):
        print(f"    [{idx:2d}] dt={dt:4d} {tt:10s} {inst[:72]}")

def family_report(fam, names):
  print(f"\n╔══ {fam} family ═════════════════════════════════════════════════")
  for name in names:
    t = load(name)
    if t is None:
      print(f"  {name}: NO CAPTURE")
      continue
    print_trace(name, t)
  print("╚" + "═" * 70)

def compare_against_baseline():
  """For each probe that was captured, run emulator and report delta."""
  os.environ["MOCKGPU"] = "1"; os.environ["DEV"] = "AMD"
  os.environ["PYTHON_REMU"] = "1"; os.environ["PROFILE"] = "1"; os.environ["SQTT"] = "1"
  os.environ["PROBE"] = "1"
  sys.path.insert(0, str(ROOT))
  from extra.sqtt.rigorous_hw_test import PROBE_KERNELS, extract_traces
  from tinygrad import Device, Tensor
  # Warmup
  (Tensor([1.]) + Tensor([1.])).realize()
  Device[Device.DEFAULT].synchronize()
  rows = []
  for name, (run_fn, _) in PROBE_KERNELS.items():
    hw = load(name)
    if hw is None: continue
    try:
      run_fn(); Device[Device.DEFAULT].synchronize()
      Device[Device.DEFAULT]._at_profile_finalize()
      emu, _ = extract_traces()
    except Exception as e:
      print(f"  {name}: EMU error: {e}"); continue
    # Compare token-by-token (assumes same wave structure; probe designs enforce this)
    for wid in sorted(set(hw.keys()) & set(emu.keys())):
      hw_w, emu_w = hw[wid], emu[wid]
      prev_hw = prev_emu = None
      for idx in range(min(len(hw_w), len(emu_w))):
        ht = hw_w[idx][1]; et = emu_w[idx][1]
        hd = (ht - prev_hw) if prev_hw is not None else 0
        ed = (et - prev_emu) if prev_emu is not None else 0
        prev_hw, prev_emu = ht, et
        inst = hw_w[idx][3]
        if idx > 0 and abs(hd - ed) > 0:
          rows.append((name, wid, idx, hd, ed, hd - ed, inst[:64]))
  print("\n╔══ Probe HW vs EMU deltas (|diff| > 0) ══════════════════════════════")
  print(f"  {'probe':<28} w idx HW  EMU diff inst")
  for r in rows:
    print(f"  {r[0]:<28} {r[1]} {r[2]:3d} {r[3]:3d} {r[4]:3d} {r[5]:+4d} {r[6]}")
  print(f"  Total non-matching tokens: {len(rows)}")
  print("╚" + "═" * 70)

if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--compare", action="store_true", help="also run emu and diff")
  args = ap.parse_args()
  for fam, names in PROBE_FAMILIES.items(): family_report(fam, names)
  if args.compare: compare_against_baseline()
