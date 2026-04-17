#!/usr/bin/env python3
"""Visualize SQTT timing mismatches between HW captures and emulator.

Usage:
  DEV=AMD MOCKGPU=1 PYTHON_REMU=1 PROFILE=1 SQTT=1 PYTHONPATH=. .venv/bin/python extra/sqtt/visualize_timing.py [--kernel NAME] [--output FILE]

Loads HW capture .pkl files from extra/sqtt/captures/rigorous/, runs the emulator,
and produces detailed color-coded timing comparison maps.
"""
import os, sys, pickle, pathlib, argparse, functools
from collections import defaultdict
os.environ.setdefault("DEV", "AMD")
os.environ.setdefault("PROFILE", "1")
os.environ.setdefault("SQTT", "1")
os.environ.setdefault("VIZ", "-2")

from tinygrad import Device, Tensor, dtypes
from tinygrad.device import Compiled, ProfileProgramEvent, ProfileDeviceEvent

CAPTURE_DIR = pathlib.Path("extra/sqtt/captures/rigorous")

# ═══════════════════════════════════════════════════════════════════════════════
# Reuse kernel definitions and helpers from rigorous_hw_test
# ═══════════════════════════════════════════════════════════════════════════════

from extra.sqtt.rigorous_hw_test import KERNELS, extract_traces, _clear

# ═══════════════════════════════════════════════════════════════════════════════
# Instruction categorization
# ═══════════════════════════════════════════════════════════════════════════════

def categorize_inst(pkt_type: str, inst_str: str) -> str:
  """Categorize an instruction into a pipeline bucket."""
  mn = inst_str.split("(")[0].strip().lower()
  if mn.startswith("s_load") or mn.startswith("s_buffer"): return "smem"
  if mn.startswith("s_wait") or mn.startswith("s_clause"): return "wait"
  if mn.startswith("s_barrier") or mn == "s_sleep": return "sync"
  if mn.startswith("s_branch") or mn.startswith("s_cbranch") or mn.startswith("s_call"): return "branch"
  if mn.startswith("s_endpgm"): return "ctrl"
  if mn.startswith("s_"): return "salu"
  if mn.startswith("v_cmp") or mn.startswith("v_cndmask"): return "vopc"
  if mn.startswith("v_interp"): return "vinterp"
  if any(mn.startswith(p) for p in ("v_exp_", "v_log_", "v_rcp_", "v_rsq_", "v_sqrt_", "v_sin_", "v_cos_")): return "valut"
  if mn.startswith("v_"): return "valu"
  if mn.startswith("ds_"): return "lds"
  if mn.startswith("global_") or mn.startswith("flat_") or mn.startswith("buffer_"): return "vmem"
  if pkt_type == "VALUINST": return "valu"
  return "other"

def mnemonic(inst_str: str) -> str:
  """Extract instruction mnemonic from full instruction string."""
  return inst_str.split("(")[0].strip()

# ═══════════════════════════════════════════════════════════════════════════════
# ANSI color helpers
# ═══════════════════════════════════════════════════════════════════════════════

USE_COLOR = True

def _green(s): return f"\033[32m{s}\033[0m" if USE_COLOR else s
def _yellow(s): return f"\033[33m{s}\033[0m" if USE_COLOR else s
def _red(s): return f"\033[31m{s}\033[0m" if USE_COLOR else s
def _bold(s): return f"\033[1m{s}\033[0m" if USE_COLOR else s
def _dim(s): return f"\033[2m{s}\033[0m" if USE_COLOR else s
def _cyan(s): return f"\033[36m{s}\033[0m" if USE_COLOR else s

def status_str(diff: int) -> str:
  if diff == 0: return _green("  ✓  ")
  if abs(diff) <= 1: return _yellow(" ±1  ")
  if abs(diff) <= 2: return _yellow(" ±2  ")
  return _red(f" ✗{diff:+d} ")

def color_diff(diff: int) -> str:
  s = f"{diff:+d}"
  if diff == 0: return _green(s)
  if abs(diff) <= 2: return _yellow(s)
  return _red(s)

# ═══════════════════════════════════════════════════════════════════════════════
# Core comparison logic
# ═══════════════════════════════════════════════════════════════════════════════

DRAM_THRESHOLD = 50  # skip deltas > this (DRAM wait noise)

def compare_wave(hw_trace, emu_trace):
  """Compare a single wave's HW vs EMU traces. Returns list of row dicts."""
  min_len = min(len(hw_trace), len(emu_trace))
  # check PC alignment
  for j in range(min_len):
    if hw_trace[j][0] != emu_trace[j][0]:
      return None, f"PC mismatch at idx {j}: HW=0x{hw_trace[j][0]:x} EMU=0x{emu_trace[j][0]:x}"
  rows = []
  for j in range(min_len):
    hd = 0 if j == 0 else hw_trace[j][1] - hw_trace[j-1][1]
    ed = 0 if j == 0 else emu_trace[j][1] - emu_trace[j-1][1]
    pc = hw_trace[j][0]
    pkt_type = hw_trace[j][2]
    inst_str = hw_trace[j][3] if len(hw_trace[j]) > 3 else ""
    cat = categorize_inst(pkt_type, inst_str)
    mn = mnemonic(inst_str)
    diff = ed - hd
    skip = hd > DRAM_THRESHOLD or ed > DRAM_THRESHOLD
    rows.append(dict(idx=j, pc=pc, mnemonic=mn, category=cat, pkt_type=pkt_type,
                     hw_delta=hd, emu_delta=ed, diff=diff, inst_str=inst_str, skip=skip))
  return rows, None

# ═══════════════════════════════════════════════════════════════════════════════
# Pattern detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_patterns(all_rows):
  """Find recurring mismatch patterns across all rows (from all waves/kernels).

  Returns list of (description, count, examples) tuples.
  """
  # Group mismatches by (mnemonic, diff)
  by_mn_diff = defaultdict(list)
  # Group by (category, diff)
  by_cat_diff = defaultdict(list)

  for kernel_name, wave_idx, rows in all_rows:
    for r in rows:
      if r["skip"] or r["diff"] == 0: continue
      key_mn = (r["mnemonic"], r["diff"])
      key_cat = (r["category"], r["diff"])
      loc = f"{kernel_name}/w{wave_idx}[{r['idx']}]"
      by_mn_diff[key_mn].append(loc)
      by_cat_diff[key_cat].append(loc)

  patterns = []
  # Mnemonic-level patterns (≥2 occurrences with same mnemonic+diff)
  for (mn, diff), locs in sorted(by_mn_diff.items(), key=lambda x: -len(x[1])):
    if len(locs) >= 2:
      patterns.append((f"all {mn} mismatches are {diff:+d}", len(locs), locs[:5]))
  # Category-level patterns (≥3 occurrences)
  for (cat, diff), locs in sorted(by_cat_diff.items(), key=lambda x: -len(x[1])):
    if len(locs) >= 3:
      desc = f"{cat} instructions tend to be {diff:+d}"
      # avoid duplicating mnemonic-level patterns
      if not any(desc.startswith(p[0][:len(cat)]) for p in patterns):
        patterns.append((desc, len(locs), locs[:5]))

  return patterns

# ═══════════════════════════════════════════════════════════════════════════════
# Rendering
# ═══════════════════════════════════════════════════════════════════════════════

def render_wave_table(rows, wave_label: str, out):
  """Print detailed per-instruction table for one wave."""
  # Header
  out(f"\n  {_bold(wave_label)}")
  hdr = f"  {'IDX':>4s}  {'PC':>8s}  {'MNEMONIC':<28s} {'CAT':<8s} {'HW':>4s} {'EMU':>4s} {'DIFF':>5s}  STATUS"
  out(f"  {_dim('─' * 82)}")
  out(hdr)
  out(f"  {_dim('─' * 82)}")

  for r in rows:
    if r["skip"]:
      out(_dim(f"  {r['idx']:4d}  0x{r['pc']:06x}  {r['mnemonic']:<28s} {r['category']:<8s} "
          f"{r['hw_delta']:4d} {r['emu_delta']:4d}        [DRAM skip]"))
      continue
    diff_s = color_diff(r["diff"])
    stat = status_str(r["diff"])
    line = f"  {r['idx']:4d}  0x{r['pc']:06x}  {r['mnemonic']:<28s} {r['category']:<8s} {r['hw_delta']:4d} {r['emu_delta']:4d}  {diff_s:>5s}  {stat}"
    out(line)

  # Wave-level stats
  counted = [r for r in rows if not r["skip"]]
  if not counted:
    out(f"  (no comparable instructions)")
    return 0, 0, 0
  exact = sum(1 for r in counted if r["diff"] == 0)
  within2 = sum(1 for r in counted if abs(r["diff"]) <= 2)
  total = len(counted)
  pct_exact = 100 * exact / total
  pct_w2 = 100 * within2 / total
  out(f"  {_dim('─' * 82)}")
  out(f"  Stats: {exact}/{total} exact ({pct_exact:.1f}%), {within2}/{total} ±2 ({pct_w2:.1f}%)")
  return exact, within2, total

def render_mismatch_context(rows, out):
  """Show 3-instruction context around each mismatch (non-skip, diff != 0)."""
  mismatch_idxs = [i for i, r in enumerate(rows) if not r["skip"] and r["diff"] != 0]
  if not mismatch_idxs:
    return
  out(f"\n  {_bold('Mismatch context')} (3 instructions before/after):")
  shown = set()
  for mi in mismatch_idxs:
    start = max(0, mi - 3)
    end = min(len(rows), mi + 4)
    rng = range(start, end)
    if all(j in shown for j in rng): continue  # already shown
    out(f"  {_dim('···')}")
    for j in rng:
      r = rows[j]
      marker = ">>>" if j == mi else "   "
      diff_s = color_diff(r["diff"]) if not r["skip"] else _dim("skip")
      color_fn = _red if j == mi else (lambda s: s)
      out(color_fn(f"  {marker} [{r['idx']:2d}] 0x{r['pc']:06x} {r['mnemonic']:<24s} {r['category']:<6s} "
                   f"HW={r['hw_delta']:3d} EMU={r['emu_delta']:3d} {diff_s}"))
      shown.add(j)

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
  parser = argparse.ArgumentParser(description="Visualize SQTT timing mismatches between HW and emulator")
  parser.add_argument("--kernel", type=str, default=None, help="Only analyze this kernel (by name)")
  parser.add_argument("--output", type=str, default=None, help="Write output to file (plain text, no ANSI)")
  parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
  parser.add_argument("--no-context", action="store_true", help="Skip mismatch context display")
  args = parser.parse_args()

  global USE_COLOR
  if args.output or args.no_color:
    USE_COLOR = False

  outfile = None
  if args.output:
    outfile = open(args.output, "w")

  lines = []
  def out(s=""):
    lines.append(s)

  assert os.environ.get("MOCKGPU") == "1", "Must run with MOCKGPU=1"

  # Warmup emulator
  (Tensor([1.]) + Tensor([1.])).realize()
  Device[Device.DEFAULT].synchronize()

  # Determine which kernels to process
  kernel_names = [args.kernel] if args.kernel else list(KERNELS.keys())

  # Per-kernel summary data
  summary = {}  # name -> (exact, within2, total, n_waves)
  all_rows_for_patterns = []  # [(kernel_name, wave_idx, rows)]

  for name in kernel_names:
    if name not in KERNELS:
      out(f"WARNING: unknown kernel '{name}', skipping")
      continue

    hw_pkl = CAPTURE_DIR / f"{name}.pkl"
    if not hw_pkl.exists():
      out(f"SKIP {name}: no HW capture at {hw_pkl}")
      continue

    with open(hw_pkl, "rb") as f:
      hw_traces = pickle.load(f)

    run_fn = KERNELS[name][0]

    # Run emulator
    _clear()
    try:
      run_fn()
      Device[Device.DEFAULT].synchronize()
      Device[Device.DEFAULT]._at_profile_finalize()
    except Exception as e:
      out(f"ERROR {name}: emulator run failed: {e}")
      continue

    emu_traces, _ = extract_traces()

    hw_waves = sorted(hw_traces.keys())
    emu_waves = sorted(emu_traces.keys())

    # Skip if wave count mismatch (HW > EMU means different workgroup size)
    if len(hw_waves) > len(emu_waves):
      out(f"SKIP {name}: wave count mismatch HW={len(hw_waves)} > EMU={len(emu_waves)}")
      continue

    n_common = min(len(hw_waves), len(emu_waves))

    out(f"\n{'═' * 90}")
    out(f"  {_bold(name.upper())}  │  HW: {len(hw_waves)} wave(s)  EMU: {len(emu_waves)} wave(s)  comparing: {n_common}")
    out(f"{'═' * 90}")

    k_exact = k_within2 = k_total = 0

    for i in range(n_common):
      hw_wid, emu_wid = hw_waves[i], emu_waves[i]
      rows, err = compare_wave(hw_traces[hw_wid], emu_traces[emu_wid])
      if rows is None:
        out(f"  Wave {i} (HW={hw_wid} EMU={emu_wid}): {err}")
        continue

      wave_label = f"Wave {i}  (HW id={hw_wid}, EMU id={emu_wid}, {len(rows)} instructions)"
      ex, w2, tot = render_wave_table(rows, wave_label, out)
      k_exact += ex; k_within2 += w2; k_total += tot

      all_rows_for_patterns.append((name, i, rows))

      # Mismatch context
      if not args.no_context:
        render_mismatch_context(rows, out)

    if k_total > 0:
      summary[name] = (k_exact, k_within2, k_total, n_common)

  # ─── Summary table ────────────────────────────────────────────────────────
  out(f"\n{'═' * 90}")
  out(f"  {_bold('SUMMARY')}")
  out(f"{'═' * 90}")
  out(f"  {'KERNEL':<22s} {'WAVES':>5s}  {'EXACT':>12s}  {'±2':>12s}  {'MISMATCHES':>10s}")
  out(f"  {'─' * 70}")

  grand_exact = grand_w2 = grand_total = 0
  for name, (ex, w2, tot, nw) in summary.items():
    pct_ex = 100 * ex / tot if tot else 0
    pct_w2 = 100 * w2 / tot if tot else 0
    miss = tot - w2
    ex_s = f"{ex}/{tot} ({pct_ex:5.1f}%)"
    w2_s = f"{w2}/{tot} ({pct_w2:5.1f}%)"
    miss_color = _red if miss > 0 else _green
    out(f"  {name:<22s} {nw:5d}  {ex_s:>12s}  {w2_s:>12s}  {miss_color(str(miss)):>10s}")
    grand_exact += ex; grand_w2 += w2; grand_total += tot

  if grand_total > 0:
    out(f"  {'─' * 70}")
    pct_ex = 100 * grand_exact / grand_total
    pct_w2 = 100 * grand_w2 / grand_total
    grand_miss = grand_total - grand_w2
    out(f"  {'TOTAL':<22s} {'':>5s}  {grand_exact}/{grand_total} ({pct_ex:5.1f}%)  "
        f"{grand_w2}/{grand_total} ({pct_w2:5.1f}%)  {grand_miss}")

  # ─── Pattern detection ────────────────────────────────────────────────────
  patterns = detect_patterns(all_rows_for_patterns)
  if patterns:
    out(f"\n{'═' * 90}")
    out(f"  {_bold('RECURRING MISMATCH PATTERNS')}")
    out(f"{'═' * 90}")
    for desc, count, examples in patterns:
      out(f"  • {_cyan(desc)}  ({count} occurrences)")
      for loc in examples:
        out(f"      {_dim(loc)}")

  # ─── Output ───────────────────────────────────────────────────────────────
  output_text = "\n".join(lines) + "\n"
  print(output_text)

  if outfile:
    # Strip ANSI codes for file output
    import re
    clean = re.sub(r'\033\[[0-9;]*m', '', output_text)
    outfile.write(clean)
    outfile.close()
    print(f"Output written to {args.output}")

if __name__ == "__main__":
  main()
