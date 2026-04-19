#!/usr/bin/env python3
"""Per-mismatch token-diff analyzer.

For each kernel in the reference suite, loads the HW capture and runs EMU,
collects every token where HW dt != EMU dt, extracts 5 tokens of context
before and 2 after, and bins by (prev_inst_class, curr_inst_class) pair.

Outputs a ranked list of patterns: rule = (prev→curr) with all observed
(HW_dt, EMU_dt, delta) triples. Patterns appearing ≥3× with a consistent
delta are actionable; ≤2× are probably wave-noise.

Run with:
  MOCKGPU=1 DEV=AMD PYTHON_REMU=1 PROFILE=1 SQTT=1 PYTHONPATH=. \\
      python3 extra/sqtt/rgp/analyze_mismatches.py
"""
from __future__ import annotations
import os, sys, pickle, re
from pathlib import Path
from collections import defaultdict, Counter
from statistics import median

os.environ.setdefault("MOCKGPU", "1")
os.environ.setdefault("DEV", "AMD")
os.environ.setdefault("PROFILE", "1")
os.environ.setdefault("SQTT", "1")
os.environ.setdefault("VIZ", "-2")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tinygrad import Device, Tensor
from tinygrad.device import Compiled
from extra.sqtt.rigorous_hw_test import extract_traces, KERNELS, CAPTURE_DIR, _clear

# ─── Inst classification ──────────────────────────────────────────────────────

def classify(inst: str) -> str:
  """Map raw instruction string to an opcode-family class used for pattern bins.

  The classifier is deliberately coarse — we want to collapse v_add_f32,
  v_add_u32, v_mul_f32 into VALU; v_cmp_* into VCMP; etc. This is what makes
  patterns emerge across kernels."""
  s = inst.strip().lower()
  if not s: return "?"
  if s.startswith("s_load"): return "S_LOAD"
  if s.startswith("global_load"): return "GLOAD"
  if s.startswith("global_store"): return "GSTORE"
  if s.startswith("ds_load") or s.startswith("ds_read"): return "DS_RD"
  if s.startswith("ds_store") or s.startswith("ds_write"): return "DS_WR"
  if s.startswith("s_waitcnt_vmcnt"): return "WAIT_VM"
  if s.startswith("s_waitcnt_lgkmcnt"): return "WAIT_LGKM"
  if s.startswith("s_waitcnt_depctr"): return "WAIT_DEP"
  if s.startswith("s_waitcnt_expcnt"): return "WAIT_EXP"
  if s.startswith("s_waitcnt"): return "WAIT"
  if s.startswith("s_nop"): return "S_NOP"
  if s.startswith("s_delay_alu"): return "S_DELAY"
  if s.startswith("s_cbranch"): return "S_CBR"
  if s.startswith("s_branch"): return "S_BR"
  if s.startswith("s_mov"): return "S_MOV"
  if s.startswith("s_cmp"): return "S_CMP"
  if s.startswith("s_and"): return "S_AND"
  if s.startswith("s_or"): return "S_OR"
  if s.startswith("s_add") or s.startswith("s_sub"): return "S_ADDSUB"
  if s.startswith("s_mul"): return "S_MUL"
  if s.startswith("s_lshl") or s.startswith("s_lshr") or s.startswith("s_ashr"): return "S_SHIFT"
  if s.startswith("s_bitcmp") or s.startswith("s_bitset"): return "S_BITOP"
  if s.startswith("s_endpgm"): return "S_ENDPGM"
  if s.startswith("s_barrier"): return "S_BARRIER"
  if s.startswith("s_setpc") or s.startswith("s_swappc"): return "S_JUMP"
  if s.startswith("v_cmp") or s.startswith("v_cmpx"):
    # LIT (constant src) vs reg: look for 0x, decimal, or float literals
    if re.search(r"\b(lit|0x|\d+\.\d|\d+,\s)", s): return "V_CMP_LIT"
    return "V_CMP"
  if s.startswith("v_cndmask"): return "V_CNDMASK"
  if s.startswith("v_dual"): return "VOPD"
  if s.startswith("v_exp") or s.startswith("v_log") or s.startswith("v_rcp") \
     or s.startswith("v_sqrt") or s.startswith("v_rsq") or s.startswith("v_sin") \
     or s.startswith("v_cos"):
    return "V_TRANS"
  if s.startswith("v_fmac") or s.startswith("v_fma") or s.startswith("v_mad"): return "V_FMA"
  if s.startswith("v_mov") and "lit" in s: return "V_MOV_LIT"
  if s.startswith("v_mov"): return "V_MOV"
  if s.startswith("v_lshl") or s.startswith("v_lshr") or s.startswith("v_ashr"): return "V_SHIFT"
  if s.startswith("v_add") or s.startswith("v_sub"): return "V_ADDSUB"
  if s.startswith("v_mul"): return "V_MUL"
  if s.startswith("v_max") or s.startswith("v_min"): return "V_MINMAX"
  if s.startswith("v_and") or s.startswith("v_or") or s.startswith("v_xor"): return "V_LOGIC"
  if s.startswith("v_cvt"): return "V_CVT"
  if s.startswith("v_"): return "V_OTHER"
  return "OTHER"

# ─── Mismatch collection ──────────────────────────────────────────────────────

def collect_mismatches(verbose: bool = False):
  """Returns list of dicts, one per (kernel, wave, token_index) where HW_dt != EMU_dt.

  Each dict: {kernel, wave, idx, hw_dt, emu_dt, delta,
              ctx_before: [5 (type, class, inst)], ctx_at: (type, class, inst),
              ctx_after: [2 (type, class, inst)]}"""
  (Tensor([1.]) + Tensor([1.])).realize()
  Device[Device.DEFAULT].synchronize()

  mismatches = []
  kernels_processed = 0
  for name, (run_fn, _) in KERNELS.items():
    hw_pkl = CAPTURE_DIR / f"{name}.pkl"
    if not hw_pkl.exists():
      continue
    with open(hw_pkl, "rb") as f:
      hw_traces = pickle.load(f)

    _clear()
    try:
      run_fn()
      Device[Device.DEFAULT].synchronize()
      Device[Device.DEFAULT]._at_profile_finalize()
      emu_traces, _ = extract_traces()
    except Exception as e:
      if verbose: print(f"  {name}: EMU error {e}")
      continue

    hw_waves = sorted(hw_traces.keys())
    emu_waves = sorted(emu_traces.keys())
    if len(hw_waves) > len(emu_waves): continue
    n_common = min(len(hw_waves), len(emu_waves))
    kernels_processed += 1

    for i in range(n_common):
      hw = hw_traces[hw_waves[i]]
      emu = emu_traces[emu_waves[i]]
      min_len = min(len(hw), len(emu))
      if min_len < 2: continue

      # PC alignment check
      hw_pc0 = hw[0][0] if hw else 0
      emu_pc0 = emu[0][0] if emu else 0
      if any(hw[j][0] - hw_pc0 != emu[j][0] - emu_pc0 for j in range(min_len)):
        continue  # stream divergence

      for j in range(1, min_len):
        hd = hw[j][1] - hw[j-1][1]
        ed = emu[j][1] - emu[j-1][1]
        if hd > 50 or ed > 50: continue  # DRAM wait
        if hd == ed: continue  # exact match

        ctx_before = []
        for k in range(max(0, j-5), j):
          tt = hw[k][2]; inst = hw[k][3] if len(hw[k]) > 3 else ""
          ctx_before.append((tt, classify(inst), inst))
        tt_at = hw[j][2]; inst_at = hw[j][3] if len(hw[j]) > 3 else ""
        ctx_at = (tt_at, classify(inst_at), inst_at)
        ctx_after = []
        for k in range(j+1, min(j+3, min_len)):
          tt = hw[k][2]; inst = hw[k][3] if len(hw[k]) > 3 else ""
          ctx_after.append((tt, classify(inst), inst))

        mismatches.append({
          "kernel": name, "wave": i, "idx": j,
          "hw_dt": hd, "emu_dt": ed, "delta": ed - hd,
          "ctx_before": ctx_before, "ctx_at": ctx_at, "ctx_after": ctx_after,
        })

  return mismatches, kernels_processed

# ─── Pattern binning ──────────────────────────────────────────────────────────

def pair_key(m, back: int = 1) -> tuple:
  """Return the (prev_class[-back], curr_class) tuple for binning."""
  prev = m["ctx_before"][-back][1] if len(m["ctx_before"]) >= back else "START"
  return (prev, m["ctx_at"][1])

def triple_key(m) -> tuple:
  """Return (prev-1, prev, curr) classes — 3-inst window."""
  cb = m["ctx_before"]
  pp = cb[-2][1] if len(cb) >= 2 else "START"
  p = cb[-1][1] if len(cb) >= 1 else "START"
  return (pp, p, m["ctx_at"][1])

def report_pair_patterns(mismatches, min_count=1):
  bins = defaultdict(list)
  for m in mismatches:
    bins[pair_key(m, 1)].append(m)

  print(f"\n{'='*78}")
  print("PAIR PATTERNS  (prev → curr)")
  print(f"{'='*78}")
  print(f"{'prev':>12s} → {'curr':<12s} n= {'hw_dts':<22} {'emu_dts':<22} delta_mode")
  print("─" * 92)
  rows = []
  for (prev, curr), ms in bins.items():
    if len(ms) < min_count: continue
    hws = Counter(m["hw_dt"] for m in ms)
    emus = Counter(m["emu_dt"] for m in ms)
    deltas = Counter(m["delta"] for m in ms)
    hw_str = ",".join(f"{v}x{c}" for v, c in hws.most_common(3))
    emu_str = ",".join(f"{v}x{c}" for v, c in emus.most_common(3))
    d_top, d_cnt = deltas.most_common(1)[0]
    rows.append((len(ms), prev, curr, hw_str, emu_str, d_top, d_cnt))
  rows.sort(key=lambda r: (-r[0], r[1], r[2]))
  for n, prev, curr, hw, emu, dtop, dcnt in rows:
    print(f"{prev:>12s} → {curr:<12s} n={n:<3d} HW[{hw:<20s}] EMU[{emu:<20s}] Δ{dtop:+d}×{dcnt}")

def report_triple_patterns(mismatches, min_count=2):
  bins = defaultdict(list)
  for m in mismatches:
    bins[triple_key(m)].append(m)

  print(f"\n{'='*78}")
  print(f"TRIPLE PATTERNS  (prev-1 → prev → curr), min_count={min_count}")
  print(f"{'='*78}")
  print(f"{'window':>30s}  n= {'HW':<18s} {'EMU':<18s} Δ")
  print("─" * 92)
  rows = []
  for (pp, p, c), ms in bins.items():
    if len(ms) < min_count: continue
    hws = Counter(m["hw_dt"] for m in ms)
    emus = Counter(m["emu_dt"] for m in ms)
    deltas = Counter(m["delta"] for m in ms)
    d_top, d_cnt = deltas.most_common(1)[0]
    hw_str = ",".join(f"{v}x{c}" for v, c in hws.most_common(2))
    emu_str = ",".join(f"{v}x{c}" for v, c in emus.most_common(2))
    rows.append((len(ms), pp, p, c, hw_str, emu_str, d_top, d_cnt))
  rows.sort(key=lambda r: -r[0])
  for n, pp, p, c, hw, emu, dtop, dcnt in rows:
    label = f"{pp}→{p}→{c}"
    print(f"{label:>30s}  n={n:<3d} HW[{hw:<16s}] EMU[{emu:<16s}] Δ{dtop:+d}×{dcnt}")

def report_by_kernel(mismatches):
  per = defaultdict(list)
  for m in mismatches: per[m["kernel"]].append(m)
  print(f"\n{'='*78}")
  print("PER-KERNEL MISMATCH COUNTS")
  print(f"{'='*78}")
  for k, ms in sorted(per.items(), key=lambda kv: -len(kv[1])):
    n_exact_miss = sum(1 for m in ms if abs(m["delta"]) > 2)
    print(f"  {k:24s} {len(ms):3d} mismatches  ({n_exact_miss} >±2)")

def dump_examples(mismatches, key_filter: tuple | None = None, n: int = 5):
  """Print example contexts for mismatches matching a given (prev, curr) pair key."""
  shown = 0
  for m in mismatches:
    if key_filter and pair_key(m, 1) != key_filter: continue
    cb = " → ".join(f"{c[1]}" for c in m["ctx_before"])
    ca = m["ctx_at"][1]
    caf = " → ".join(f"{c[1]}" for c in m["ctx_after"])
    print(f"\n  [{m['kernel']} W{m['wave']} #{m['idx']}] HW={m['hw_dt']} EMU={m['emu_dt']} Δ{m['delta']:+d}")
    print(f"    ctx  : {cb}  ║  [{ca}]  ║  {caf}")
    print(f"    inst : ...{m['ctx_before'][-1][2][:40] if m['ctx_before'] else '<start>'}  ║  {m['ctx_at'][2][:50]}")
    shown += 1
    if shown >= n: break

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
  import argparse
  ap = argparse.ArgumentParser()
  ap.add_argument("--min-pair", type=int, default=1, help="min count to show pair pattern")
  ap.add_argument("--min-triple", type=int, default=2, help="min count to show triple pattern")
  ap.add_argument("--examples", type=str, default="", help="prev,curr class pair to dump examples for")
  ap.add_argument("--n-examples", type=int, default=8)
  ap.add_argument("--no-run", action="store_true", help="don't run EMU; just reload from cached /tmp/mismatches.pkl")
  args = ap.parse_args()

  cache = Path("/tmp/mismatches.pkl")
  if args.no_run and cache.exists():
    with open(cache, "rb") as f: mismatches = pickle.load(f)
    print(f"Loaded {len(mismatches)} mismatches from cache")
  else:
    mismatches, k = collect_mismatches()
    with open(cache, "wb") as f: pickle.dump(mismatches, f)
    print(f"\nCollected {len(mismatches)} mismatches across {k} kernels")
    print(f"(cached → {cache})")

  report_by_kernel(mismatches)
  report_pair_patterns(mismatches, min_count=args.min_pair)
  report_triple_patterns(mismatches, min_count=args.min_triple)

  if args.examples:
    parts = args.examples.split(",")
    if len(parts) == 2:
      print(f"\n{'='*78}")
      print(f"EXAMPLES for pair {parts[0]}→{parts[1]}")
      print(f"{'='*78}")
      dump_examples(mismatches, key_filter=(parts[0].strip(), parts[1].strip()), n=args.n_examples)
