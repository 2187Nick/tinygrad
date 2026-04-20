#!/usr/bin/env python3
"""Decode raw SQTT blobs without filtering by simd==0.

Complement to capture_raw_sqtt.py. The stock decoder in
tinygrad/renderer/amd/sqtt.py:652-658 hard-filters for (traced_cu, simd==0).
That's intentional — the SQTT viewer only supports one unit — but it hides
the data we need to answer: "which SIMDs did the workgroup's waves actually
land on?"

This script re-decodes the raw blobs captured by capture_raw_sqtt.py, records
every WAVESTART/WAVEEND packet regardless of SIMD, and emits a JSON summary
per kernel: per-(cu, simd) wave counts, per-wave (cu, simd) assignment,
lifetime (WAVESTART→WAVEEND delta).

Usage:
  .venv/bin/python extra/sqtt/wave_probe/decode_all_simds.py \\
      extra/sqtt/wave_probe/captures/raw_sqtt_<ts>
"""
import os, sys, json, pickle, pathlib
from collections import defaultdict, deque

_repo_root = pathlib.Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path: sys.path.insert(0, str(_repo_root))

from tinygrad.renderer.amd.sqtt import decode, WAVESTART, WAVESTART_RDNA4, WAVEEND


def decode_all_simds(blob: bytes) -> dict:
  """Return {(cu, simd): [(wave_slot, start_time, end_time), ...]} from a raw SQTT blob."""
  open_waves: dict[tuple[int, int, int], deque[int]] = defaultdict(deque)  # (cu, simd, wave) → deque of start_times
  result: dict[tuple[int, int], list] = defaultdict(list)
  for p in decode(blob):
    if isinstance(p, (WAVESTART, WAVESTART_RDNA4)):
      key = (p.cu, p.simd, p.wave)
      open_waves[key].append(p._time)
    elif isinstance(p, WAVEEND):
      key = (p.cu, p.simd, p.wave)
      if not open_waves[key]: continue
      start_t = open_waves[key].popleft()
      result[(p.cu, p.simd)].append((p.wave, start_t, p._time))
  # Include still-open waves (kernel may have truncated trace)
  for (cu, simd, wave), pending in open_waves.items():
    for start_t in pending:
      result[(cu, simd)].append((wave, start_t, None))
  return dict(result)


def summarize(decoded: dict) -> dict:
  cu_simd_counts = {f"cu{cu}_simd{simd}": len(waves) for (cu, simd), waves in decoded.items()}
  per_wave = defaultdict(list)
  for (cu, simd), waves in decoded.items():
    for wave, start_t, end_t in waves:
      per_wave[wave].append({"cu": cu, "simd": simd, "start": start_t,
                             "end": end_t, "lifetime": (end_t - start_t) if end_t else None})
  return {
    "total_waves_observed": sum(len(v) for v in decoded.values()),
    "unique_units": len(decoded),
    "per_cu_simd_wave_count": cu_simd_counts,
    "per_wave_placements": {str(k): v for k, v in per_wave.items()},
    "simd_balance": _simd_balance(decoded),
  }


def _simd_balance(decoded: dict) -> dict:
  """Per-CU how many waves on each of the 4 SIMDs. Shows whether allocator spreads."""
  cu_simd: dict[int, dict[int, int]] = defaultdict(lambda: {0: 0, 1: 0, 2: 0, 3: 0})
  for (cu, simd), waves in decoded.items():
    if simd < 4: cu_simd[cu][simd] = len(waves)
  return {f"cu{cu}": dict(v) for cu, v in cu_simd.items()}


def main():
  if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <capture_dir>")
    sys.exit(1)
  capture_dir = pathlib.Path(sys.argv[1])
  assert capture_dir.is_dir(), f"not a dir: {capture_dir}"

  out_path = capture_dir / "all_simds_summary.json"
  all_results = {}

  for pkl_path in sorted(capture_dir.glob("*.pkl")):
    if pkl_path.name == "all_simds_summary.json": continue
    with open(pkl_path, "rb") as f:
      rec = pickle.load(f)
    blob = rec["blob"]
    try:
      decoded = decode_all_simds(blob)
      summary = summarize(decoded)
      all_results[pkl_path.stem] = summary
      print(f"  {pkl_path.stem:45s} waves={summary['total_waves_observed']:4d} "
            f"units={summary['unique_units']:2d} balance={summary['simd_balance']}")
    except Exception as e:
      print(f"  {pkl_path.stem}: DECODE ERROR {e}")
      all_results[pkl_path.stem] = {"error": str(e)}

  with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
  print(f"\nSummary saved: {out_path}")

  # Compare against the EMU assumption as a rough diagnostic. SQTT's `wave` is a
  # hardware wave slot, not a launch wave index, so this can still mismatch even
  # when scheduling is behaving as expected.
  print("\n=== EMU assumption check: wave_idx % 4 == observed simd_id? ===")
  matches, mismatches = 0, 0
  for kernel, summary in all_results.items():
    if "error" in summary: continue
    for wave_str, placements in summary.get("per_wave_placements", {}).items():
      wave_idx = int(wave_str)
      expected_simd = wave_idx % 4
      for p in placements:
        if p["simd"] == expected_simd: matches += 1
        else: mismatches += 1
  total = matches + mismatches
  if total:
    print(f"  matches: {matches}/{total} ({100.0*matches/total:.1f}%)")
    print(f"  mismatches: {mismatches}/{total} — emu.py:3536 and simd_arbiter.py assume wave_idx % 4")


if __name__ == "__main__":
  main()
