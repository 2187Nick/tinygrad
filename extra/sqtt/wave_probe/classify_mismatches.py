#!/usr/bin/env python3
"""Classify emu↔HW mismatches using 50×100 HW distribution + single EMU run.

Problem: `rigorous_hw_test.py --compare` shows EMU vs HW-reference-capture but
gives no signal on HW variance across runs. A token that HW produces at dt=1
vs dt=5 across 100 runs is a wave-variance (stochastic-arbitration) problem;
a token that HW produces at dt=5 100/100 runs but EMU says 1 is a deterministic
rule bug. These demand very different fixes.

This script:
  1) Loads captures from extra/sqtt/wave_probe/captures/targeted/<kernel>/*.pkl
  2) For each (wave, token) computes HW dt {distribution, mode, min, max, stdev}
  3) Runs EMU once (via rigorous_hw_test machinery) for the same kernels
  4) Classifies each missed token into one of:
        • BUG_DET         HW stable, EMU differs by >0   (fix EMU deterministic rule)
        • WAVE_VARIANCE   HW has ≥2 modes with ≥20% each (emu can pick mode but not match both)
        • NOISY           HW has stdev > 3               (emu can only match ±3 at best)
        • NEAR_MISS       EMU within HW range            (already optimal for this token)

Output: JSON + console table, saved to
  extra/sqtt/wave_probe/captures/targeted/mismatch_classification.json

Usage (no GPU needed):
  MICROBENCH=1 PYTHONPATH=. .venv/bin/python \\
    extra/sqtt/wave_probe/classify_mismatches.py
"""
import os, sys, pickle, pathlib, json
import statistics
from collections import defaultdict, Counter

os.environ.setdefault("DEV", "AMD")
os.environ.setdefault("MOCKGPU", "1")
os.environ.setdefault("PYTHON_REMU", "1")
os.environ.setdefault("PROFILE", "1")
os.environ.setdefault("SQTT", "1")
os.environ.setdefault("MICROBENCH", "1")

CAPTURE_DIR = pathlib.Path("extra/sqtt/wave_probe/captures/targeted")


def _load_captures(kname: str) -> list[dict]:
  d = CAPTURE_DIR / kname
  if not d.exists(): return []
  caps = []
  for pkl in sorted(d.glob("run_*.pkl")):
    try:
      with open(pkl, "rb") as f: caps.append(pickle.load(f))
    except Exception:
      pass
  return caps


def _compute_hw_dt_distribution(caps: list[dict]) -> dict:
  """Returns {wave_idx: {token_idx: {dt: count, ...}}}"""
  out: dict = defaultdict(lambda: defaultdict(Counter))
  for cap in caps:
    for wid, pkts in cap.items():
      if wid == '_meta': continue
      prev_t = None
      tok = 0
      for (pc, t, ptype, inst) in pkts:
        if ptype in ("INST", "VALUINST", "IMMEDIATE"):
          dt = t - prev_t if prev_t is not None else 0
          # Convert wave-HW-slot → wave-launch-idx if possible (use sorted order).
          out[wid][tok][dt] += 1
          tok += 1
        prev_t = t
  return out


def _normalize_waves_by_start(caps: list[dict]) -> list[list[int]]:
  """Map each capture's HW wave-slots to an ordered launch index list.
  Returns a list (len == n_captures) of lists where index i is the HW slot of launch-wave i.
  Uses _meta.wavestarts when available, else sorts by slot id."""
  order_lists = []
  for cap in caps:
    meta = cap.get('_meta', {}) if isinstance(cap, dict) else {}
    wss = meta.get('wavestarts', {}) if isinstance(meta, dict) else {}
    wave_slots = [w for w in cap.keys() if w != '_meta']
    if wss and len(wss) >= len(wave_slots):
      order = sorted(wave_slots, key=lambda w: wss.get(w, 10**18))
    else:
      order = sorted(wave_slots)
    order_lists.append(order)
  return order_lists


def _compute_hw_dist_by_launch_index(caps: list[dict]) -> dict:
  """Same as _compute_hw_dt_distribution but indexed by launch_idx (wave_idx by start time)."""
  order_lists = _normalize_waves_by_start(caps)
  out: dict = defaultdict(lambda: defaultdict(Counter))
  inst_at: dict[int, dict[int, str]] = defaultdict(dict)
  for cap, order in zip(caps, order_lists):
    for launch_idx, slot in enumerate(order):
      pkts = cap[slot]
      prev_t = None
      tok = 0
      for (pc, t, ptype, inst) in pkts:
        if ptype in ("INST", "VALUINST", "IMMEDIATE"):
          dt = t - prev_t if prev_t is not None else 0
          out[launch_idx][tok][dt] += 1
          inst_at[launch_idx].setdefault(tok, str(inst)[:60])
          tok += 1
        prev_t = t
  return {'dist': out, 'inst': inst_at}


def _run_emu(kname: str):
  """Run EMU via rigorous_hw_test's compare path for a single kernel, extract per-wave per-token dts."""
  try:
    from extra.sqtt.rigorous_hw_test import KERNELS, _clear
    from tinygrad import Device
    from tinygrad.device import Compiled
    from tinygrad.renderer.amd.sqtt import map_insts
  except Exception as e:
    raise RuntimeError(f"EMU import failed: {e}")
  if kname not in KERNELS: return None
  run_fn, _ = KERNELS[kname]
  # Run once under emu
  _clear()
  try:
    run_fn()
    Device[Device.DEFAULT].synchronize()
    Device[Device.DEFAULT]._at_profile_finalize()
  except Exception as e:
    return None
  sqtt_events = [e for e in Compiled.profile_events if type(e).__name__ == 'ProfileSQTTEvent']
  program_events = {e.tag: e for e in Compiled.profile_events if type(e).__name__ == 'ProfileProgramEvent'}
  per_wave: dict = defaultdict(list)
  for ev in sqtt_events:
    if not ev.itrace: continue
    if ev.kern not in program_events: continue
    kev = program_events[ev.kern]
    for pkt, info in map_insts(ev.blob, kev.lib, "gfx1100"):
      if info is None: continue
      per_wave[info.wave].append((pkt._time, type(pkt).__name__))
  # Compute dts per wave (index by sort-order, same convention as HW)
  ordered = sorted(per_wave.keys())
  out = {}
  for launch_idx, w in enumerate(ordered):
    pkts = per_wave[w]
    prev = None
    dts = []
    for (t, ptype) in pkts:
      if ptype in ("INST", "VALUINST", "IMMEDIATE"):
        dts.append(t - prev if prev is not None else 0)
      prev = t
    out[launch_idx] = dts
  return out


def _classify_token(hw_counter: Counter, emu_dt: int) -> tuple[str, dict]:
  total = sum(hw_counter.values())
  modes = hw_counter.most_common(3)
  top_val, top_n = modes[0]
  top_frac = top_n / total
  vals = list(hw_counter.elements())
  mn, mx = min(vals), max(vals)
  stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
  info = {
    'total_runs': total,
    'mode': top_val, 'mode_frac': round(top_frac, 3),
    'min': mn, 'max': mx, 'stdev': round(stdev, 2),
    'top3': [(v, n) for v, n in modes],
    'emu': emu_dt,
  }
  if emu_dt is None:
    return 'EMU_MISSING', info
  if emu_dt == top_val:
    return 'HIT', info
  if mn <= emu_dt <= mx:
    return 'NEAR_MISS', info
  # Wave-variance vs noisy vs bug
  if top_frac < 0.8 and len(modes) >= 2 and modes[1][1] / total >= 0.2:
    return 'WAVE_VARIANCE', info
  if stdev > 3:
    return 'NOISY', info
  return 'BUG_DET', info


def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--kernel", type=str, default=None, help="Classify single kernel only")
  parser.add_argument("--top", type=int, default=30, help="Max kernels to analyze")
  args = parser.parse_args()

  # Find kernels with captures
  kernels_with_caps = sorted([
    d.name for d in CAPTURE_DIR.iterdir()
    if d.is_dir() and list(d.glob("run_*.pkl"))
  ])
  if args.kernel:
    kernels_with_caps = [args.kernel] if args.kernel in kernels_with_caps else []
  if not kernels_with_caps:
    print("No captures found. Run capture_50x100.py or capture_expansion.py first.")
    return

  print(f"Classifying {min(args.top, len(kernels_with_caps))} of {len(kernels_with_caps)} kernels...\n")

  out = {}
  totals = Counter()
  for kname in kernels_with_caps[:args.top]:
    caps = _load_captures(kname)
    if not caps: continue
    hw = _compute_hw_dist_by_launch_index(caps)
    hw_dist, hw_inst = hw['dist'], hw['inst']
    emu = _run_emu(kname)
    if emu is None:
      print(f"  SKIP {kname}: EMU unavailable")
      continue

    k_class = Counter()
    mismatches = []
    for w_idx in sorted(hw_dist.keys()):
      for tok in sorted(hw_dist[w_idx].keys()):
        emu_dt = emu.get(w_idx, [None]*(tok+1))[tok] if tok < len(emu.get(w_idx, [])) else None
        cls, info = _classify_token(hw_dist[w_idx][tok], emu_dt)
        k_class[cls] += 1
        if cls not in ('HIT', 'NEAR_MISS'):
          mismatches.append({
            'wave': w_idx, 'tok': tok, 'class': cls,
            'inst': hw_inst[w_idx].get(tok, '?'),
            **info,
          })
    totals.update(k_class)
    out[kname] = {
      'runs': len(caps),
      'summary': dict(k_class),
      'mismatches': mismatches[:20],  # top 20 per kernel
    }
    print(f"  {kname:<45s} runs={len(caps):3d}  "
          f"HIT={k_class['HIT']:3d}  NEAR={k_class['NEAR_MISS']:3d}  "
          f"BUG_DET={k_class['BUG_DET']:3d}  WAVE_VAR={k_class['WAVE_VARIANCE']:3d}  "
          f"NOISY={k_class['NOISY']:3d}")

  print(f"\n{'='*80}")
  print("TOTALS:")
  for cls, n in sorted(totals.items(), key=lambda x: -x[1]):
    print(f"  {cls:<20s} {n:5d}")

  # Write JSON
  out_path = CAPTURE_DIR / "mismatch_classification.json"
  with open(out_path, "w") as f:
    json.dump({'totals': dict(totals), 'per_kernel': out}, f, indent=2)
  print(f"\n  Full details: {out_path}")


if __name__ == "__main__":
  main()
