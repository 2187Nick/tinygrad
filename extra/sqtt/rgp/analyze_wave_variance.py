#!/usr/bin/env python3
"""Classify per-wave dt variance across HW captures.

For each kernel in captures/rigorous/, loads the HW .pkl and for each
token index produces the multiset of (dt) values across all waves.
Classifies each token as:

  INVARIANT       — all waves same dt
  LINEAR(slope)   — dt roughly linear in wave_idx (slope ≥ 0.3cy/wave)
  BIMODAL(a,b)    — two distinct dt values, each ≥25% of waves
  CLUSTERED(N)    — exactly N distinct dt values
  CHAOTIC         — >4 distinct dts, no linear fit

Skips DRAM-like dts (>50cy) — those are orthogonal to the SQ arbitration
model we want to fix.

Run:
  .venv/bin/python extra/sqtt/rgp/analyze_wave_variance.py

Outputs STOCHASTIC_FINGERPRINTS.md with the per-kernel classification
and a summary of where wave variance is tractable vs chaotic.
"""
from __future__ import annotations
import pickle, pathlib
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[3]
CAPTURES = ROOT / "extra/sqtt/captures/rigorous"
OUT = ROOT / "extra/sqtt/rgp/STOCHASTIC_FINGERPRINTS.md"


def classify(dts: list[int]) -> tuple[str, str]:
  """Given per-wave dts for one token position, return (class, detail)."""
  n = len(dts)
  if n < 2:
    return ("N/A", "single wave")
  uniq = Counter(dts)
  if len(uniq) == 1:
    return ("INVARIANT", str(dts[0]))
  if len(uniq) == 2:
    (a, ac), (b, bc) = uniq.most_common(2)
    if ac >= n * 0.25 and bc >= n * 0.25:
      return ("BIMODAL", f"{a}x{ac},{b}x{bc}")
  if len(uniq) <= 4:
    bits = ",".join(f"{v}x{c}" for v, c in uniq.most_common(len(uniq)))
    return (f"CLUSTERED{len(uniq)}", bits)
  # linear check: sort by wave_idx, fit slope
  # dts is already in wave_idx order
  slope = sum((i - (n-1)/2) * (d - sum(dts)/n) for i, d in enumerate(dts))
  slope /= sum((i - (n-1)/2) ** 2 for i in range(n))
  mean = sum(dts) / n
  fit = [mean + slope * (i - (n-1)/2) for i in range(n)]
  residual = sum((d - f) ** 2 for d, f in zip(dts, fit))
  total_var = sum((d - mean) ** 2 for d in dts)
  r2 = 1 - residual / total_var if total_var > 0 else 0
  if abs(slope) >= 0.3 and r2 >= 0.5:
    return ("LINEAR", f"slope={slope:+.2f} r2={r2:.2f}")
  return ("CHAOTIC", f"{len(uniq)} distinct")


def analyze_kernel(name: str, traces: dict) -> list[tuple[int, str, str, list[int]]]:
  """Returns [(token_idx, class, detail, dts)] for every token position
  with variance. Skips INVARIANT tokens."""
  waves = sorted(traces.keys())
  if len(waves) < 2:
    return []
  # Compute per-wave per-token dts
  min_len = min(len(traces[w]) for w in waves)
  out = []
  for j in range(1, min_len):
    dts = []
    for w in waves:
      tr = traces[w]
      dt = tr[j][1] - tr[j-1][1]
      if dt > 50:
        dts = None
        break
      dts.append(dt)
    if dts is None:
      continue
    cls, detail = classify(dts)
    if cls == "INVARIANT":
      continue
    out.append((j, cls, detail, dts))
  return out


def main():
  pkl_files = sorted(CAPTURES.glob("*.pkl"))
  total_variant = 0
  per_class: Counter = Counter()
  per_class_kernels: dict[str, list[str]] = {}
  kernel_reports = []

  for pkl in pkl_files:
    with open(pkl, "rb") as f:
      try:
        traces = pickle.load(f)
      except Exception:
        continue
    if not isinstance(traces, dict):
      continue
    report = analyze_kernel(pkl.stem, traces)
    if not report:
      continue
    kernel_reports.append((pkl.stem, report))
    class_bins: Counter = Counter()
    for _, cls, _, _ in report:
      # Normalize class name (drop CLUSTERED number for top-level roll-up)
      base = cls.split("(")[0].split()[0]
      if base.startswith("CLUSTERED"):
        base = "CLUSTERED"
      class_bins[base] += 1
      per_class[base] += 1
      per_class_kernels.setdefault(base, []).append(pkl.stem)
    total_variant += len(report)

  # Emit markdown report
  lines = [
    "# HW wave-variance fingerprints",
    "",
    f"Scanned {len(pkl_files)} captures, found {total_variant} tokens with per-wave variance.",
    "",
    "## Summary by class",
    "",
    "| Class | Token count | % of variant |",
    "|---|---|---|",
  ]
  total = max(total_variant, 1)
  for cls, cnt in per_class.most_common():
    lines.append(f"| {cls} | {cnt} | {100*cnt/total:.1f}% |")
  lines += [
    "",
    "## Class semantics",
    "",
    "- **BIMODAL**: two distinct dts, 25%+ in each — suggests 2-way arbitration split",
    "  (wave-parity, odd/even SIMD, or fast/slow lane).",
    "- **CLUSTERED(N)**: N ≤ 4 distinct dts — N-slot round-robin or tiered latency.",
    "- **LINEAR**: dt grows/shrinks monotonically with wave_idx — per-wave launch stagger or",
    "  accumulated contention.",
    "- **CHAOTIC**: >4 distinct dts without linear fit — needs full HW arbitration model",
    "  (likely unreachable with simple rules).",
    "",
    "## Per-kernel breakdown",
    "",
  ]
  for name, report in sorted(kernel_reports):
    if len(report) == 0:
      continue
    cls_count: Counter = Counter()
    for _, cls, _, _ in report:
      base = cls.split("(")[0].split()[0]
      if base.startswith("CLUSTERED"):
        base = "CLUSTERED"
      cls_count[base] += 1
    cls_str = ", ".join(f"{c}={n}" for c, n in cls_count.most_common())
    lines.append(f"### {name} — {len(report)} variant tokens ({cls_str})")
    lines.append("")
    # Show first 12 variant tokens with details
    for j, cls, detail, dts in report[:12]:
      lines.append(f"- `[{j}]` **{cls}** {detail}  →  dts: `{dts}`")
    if len(report) > 12:
      lines.append(f"- ... {len(report)-12} more")
    lines.append("")

  OUT.write_text("\n".join(lines))
  print(f"Wrote {OUT}")
  print(f"\nTotal variant tokens: {total_variant}")
  print(f"Class roll-up:")
  for cls, cnt in per_class.most_common():
    print(f"  {cls:12s}: {cnt:5d} ({100*cnt/total:.1f}%)")


if __name__ == "__main__":
  main()
