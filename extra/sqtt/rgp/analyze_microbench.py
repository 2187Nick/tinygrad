#!/usr/bin/env python3
"""Analyze Batch A microbench HW captures and validate against hypotheses.

For each microbench, loads the per-wave SQTT .pkl, computes per-token dt, and
prints the median dt per instruction. Then compares against EMU predictions
and against the hypothesized HW constants from PROBE_FINDINGS.md and the
MICROBENCH_TAXONOMY.md expectations column.
"""
from __future__ import annotations
import os, sys, pickle
from pathlib import Path
from statistics import median
from collections import Counter

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
CAPTURE_DIR = ROOT / "extra" / "sqtt" / "captures" / "rigorous"

def load(name):
  p = CAPTURE_DIR / f"{name}.pkl"
  if not p.exists(): return None
  with open(p, "rb") as f: return pickle.load(f)

def per_token_dts(traces):
  """Returns list of lists — outer is per-token-index, inner is dts across all waves."""
  if not traces: return []
  max_idx = max(len(t) for t in traces.values())
  by_idx: list[list[int]] = [[] for _ in range(max_idx)]
  for wid, toks in traces.items():
    prev = None
    for i, (pc, t, tt, inst) in enumerate(toks):
      if prev is not None and tt in ("INST", "VALUINST", "IMMEDIATE"):
        by_idx[i].append(t - prev)
      prev = t
  return by_idx

def fmt_hw(dts_by_idx, traces):
  """Print table: idx | dt median | dt mode(count) | inst"""
  # pick wave 0 as representative for inst names
  w0 = traces[sorted(traces.keys())[0]]
  lines = []
  for idx in range(min(len(dts_by_idx), len(w0))):
    dts = dts_by_idx[idx]
    inst = w0[idx][3] if idx < len(w0) else "?"
    tt = w0[idx][2]
    if tt not in ("INST", "VALUINST", "IMMEDIATE") or not dts:
      continue
    c = Counter(dts)
    top, top_count = c.most_common(1)[0]
    m = int(median(dts))
    same = "=" if top == m else "~"
    lines.append(f"    [{idx:2d}] {tt:10s} dt median={m:3d} mode={top:3d}×{top_count:<2d} {same}  {inst[:56]}")
  return lines

# ─── Hypothesis checks ────────────────────────────────────────────────────────

HYP = [
  # (microbench, expected_dt_at_idx, description)
  # s_nop formula
  ("mb_snop_0_after_valu",             [1],      "s_nop(0) after VALU: 1cy"),
  ("mb_snop_5_after_valu",             [6],      "s_nop(5) after VALU: 6cy"),
  ("mb_snop_10_after_valu",            [11],     "s_nop(10) after VALU: 11cy"),
  ("mb_snop_15_after_valu",            [16],     "s_nop(15) after VALU: 16cy"),
  ("mb_snop_15_after_waitcnt_vmcnt",   [20],     "s_nop(15) after vmcnt: 20cy (PROBE_FINDINGS §B)"),
  ("mb_snop_15_after_waitcnt_lgkmcnt", [16,20],  "s_nop(15) after lgkmcnt: 16 or 20?"),
  ("mb_snop_15_after_waitcnt_empty",   [16,20],  "s_nop(15) after empty waitcnt: ?"),
  ("mb_snop_15_after_depctr",          [16,20],  "s_nop(15) after depctr: ?"),
  # nop chain: last-in-chain +4?
  ("mb_snop_15_chain_n2",              [16,20],  "2-chain: [16,20]? (last +4)"),
  ("mb_snop_15_chain_n3",              [16,16,20], "3-chain: [16,16,20]?"),
  ("mb_snop_15_chain_n5",              [16,16,16,16,20], "5-chain: middle 16, last 20"),
  ("mb_snop_15_chain_n8",              [16]*7+[20], "8-chain: middle 16, last 20"),
  ("mb_snop_15_chain_after_vmcnt_n3",  [20,20,20], "vmcnt→3-chain: all 20"),
  # scalar beat (E)
  ("mb_salu_scmp_tight",               [8],      "tight s_mov→s_cmp→s_cbranch: 8cy"),
  ("mb_salu_scmp_spaced_nop0",         [13],     "s_mov→s_nop→s_cmp→s_cbranch: 13cy"),
  ("mb_salu_scmp_spaced_nop0x2",       [13],     "same w/ 2 nops"),
  ("mb_salu_scmp_spaced_nop0x3",       [13],     "same w/ 3 nops"),
  ("mb_salu_smov_followed_by_nop0",    [3],      "s_mov→s_nop(0): 3cy (scalar warmup)"),
  # VALU burst
  ("mb_valu_add_n4",                   [1,1,1,1], "VALU burst dt=1"),
  ("mb_valu_fmac_n8",                  [1]*8,    "v_fmac burst"),
  # Trans pipe
  ("mb_trans_exp_n4",                  [1,4,4,4], "trans pipe 4cy occupancy"),
  ("mb_trans_mixed_exp_log",           [1,4],    "exp→log back-to-back"),
  # VOPD independent
  ("mb_vopd_fmac_mul_n2",              [1,1],    "VOPD no-dep n2"),
  ("mb_vopd_fmac_mul_n4",              [1,1,1,1], "VOPD no-dep n4"),
  ("mb_vopd_mixed_n4",                 [1,1,1,1], "VOPD mixed types"),
]

def check_hypotheses(verbose: bool = False):
  print("\n" + "="*72)
  print("Hypothesis checks — does HW match predictions?")
  print("="*72)
  for name, expected, desc in HYP:
    traces = load(name)
    if traces is None:
      print(f"  SKIP {name}: no capture")
      continue
    dts = per_token_dts(traces)
    # find body tokens (skip prologue: s_load, waitcnt_lgkmcnt, lshlrev, load, waitcnt_vmcnt = 5 tokens)
    w0 = traces[sorted(traces.keys())[0]]
    # body starts after the second s_waitcnt in the prologue; find it.
    body_start = 5  # default: after prologue 5 tokens
    for i, (_, _, tt, inst) in enumerate(w0[:10]):
      if tt == "IMMEDIATE" and "vmcnt" in inst.lower():
        body_start = i + 1; break
    # pick the body tokens of interest, one per expected
    body = dts[body_start:]
    # find the first non-trivial INST/VALUINST/IMMEDIATE indices in the body
    probe_idxs = []
    for i, toks in enumerate(body):
      if not toks: continue
      if body_start + i < len(w0):
        tt = w0[body_start + i][2]
        if tt in ("INST", "VALUINST", "IMMEDIATE"):
          probe_idxs.append(i)
      if len(probe_idxs) >= len(expected): break
    got = []
    for i in probe_idxs[:len(expected)]:
      if body[i]:
        got.append(Counter(body[i]).most_common(1)[0][0])
      else:
        got.append(-1)
    # classify
    exp_list = [e if isinstance(e, list) else [e] for e in expected]
    ok = all(g in e for g, e in zip(got, exp_list))
    icon = "✓" if ok else ("?" if all(g >= 0 for g in got) else "!")
    print(f"  [{icon}] {name:38s} expected {expected}  got {got}  — {desc}")
    if verbose and not ok:
      print(f"       wave-0 body (first 10):")
      for j in range(min(10, len(body))):
        if j < len(probe_idxs):
          idx = body_start + probe_idxs[j] if probe_idxs[j] < len(probe_idxs) else -1
        inst = w0[body_start + j][3][:50] if body_start + j < len(w0) else "?"
        dts_here = body[j] if j < len(body) else []
        print(f"         [{j}] dts={dts_here[:4]} inst={inst}")

def dump_all():
  print("\n" + "="*72); print("All mb_* captures, wave 0 body only"); print("="*72)
  files = sorted(CAPTURE_DIR.glob("mb_*.pkl"))
  for f in files:
    name = f.stem
    traces = load(name)
    if not traces: continue
    print(f"\n── {name} ── ({len(traces)} waves)")
    dts = per_token_dts(traces)
    for line in fmt_hw(dts, traces)[:20]: print(line)

if __name__ == "__main__":
  if "--dump" in sys.argv: dump_all()
  else: check_hypotheses(verbose="-v" in sys.argv)
