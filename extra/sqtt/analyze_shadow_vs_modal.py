#!/usr/bin/env python3
"""Correlate SIMD-arbiter shadow events with strict/MODAL mismatches.

Purpose (SIMD-arbiter refactor Step 4 per simd_arbiter.py:87-106):
  The "naive 1cy/SIMD apply" dead-end proved that applying a blanket port-
  availability stall regresses 24K tokens because SQTT stamps at DISPATCH,
  which absorbs back-to-back same-SIMD issues that aren't clustered. We need
  a NARROWER gate. The hypothesis: genuine queue pressure only materializes
  when multiple peer waves on the same SIMD are ready within a few cycles of
  each other — that's the "peer-wave ready clustering" signal.

Method:
  1. Run the full rigorous_hw_test compare loop with SIMD_ARB_SHADOW=2 so emu.py
     emits one shadow-event dict per VALU issue (wave_id, stamp_cycle, simd,
     issue_cycle, port_avail_before, would_stall_cy, peer_cluster_2cy).
  2. For every VALU token in emu_traces, look up its shadow event by
     (wave_id, stamp_cycle). Compute hd/ed like the compare loop does.
  3. Bucket tokens by peer_cluster_2cy and by would_stall_cy>0, report how
     strict-exact and MODAL-exact rates vary across buckets.

If the hypothesis holds, strict-exact rate should drop sharply in the
high-cluster bucket (that's where HW stalls and emu doesn't); applying a
stall only in that bucket should reclaim those tokens.

Usage:
  DEV=AMD MOCKGPU=1 PYTHON_REMU=1 PROFILE=1 SQTT=1 MICROBENCH=1 \\
    PYTHONPATH=. .venv/bin/python extra/sqtt/analyze_shadow_vs_modal.py
"""
import os, pickle, pathlib, sys

os.environ['SIMD_ARB_SHADOW'] = '2'
os.environ.setdefault('DEV', 'AMD')
os.environ.setdefault('MOCKGPU', '1')
os.environ.setdefault('PYTHON_REMU', '1')
os.environ.setdefault('PROFILE', '1')
os.environ.setdefault('SQTT', '1')
os.environ.setdefault('MICROBENCH', '1')

# Import AFTER env setup — KERNELS is built at import time and reads MICROBENCH.
from tinygrad import Device, Tensor
from tinygrad.device import Compiled, ProfileEvent, ProfileProgramEvent, ProfileDeviceEvent
from extra.sqtt import rigorous_hw_test as rht
from test.mockgpu.amd import emu as emu_mod


def _kernel_shadow_slice(prev_end: int) -> tuple[list[dict], int]:
  """Return shadow events produced since prev_end, and the new end offset."""
  new_end = len(emu_mod._simd_arb_shadow_events)
  return emu_mod._simd_arb_shadow_events[prev_end:new_end], new_end


def main():
  print("=== Shadow-event ↔ mismatch correlation ===")

  # Warmup
  (Tensor([1.]) + Tensor([1.])).realize()
  Device[Device.DEFAULT].synchronize()

  buckets_strict = {c: [0, 0] for c in range(8)}  # cluster_n → [exact, total]
  buckets_modal = {c: [0, 0] for c in range(8)}
  would_stall_buckets = {"none": [0, 0, 0], "any": [0, 0, 0]}  # [strict_exact, modal_exact, total]

  # Kernel-level MODAL miss density per cluster bucket (for narrow-gate design)
  kernel_modal_miss = {}  # name -> {cluster_n: [miss, total]}
  # Sample mismatches to inspect (high-cluster MODAL misses only — actionable signal)
  high_cluster_modal_miss_samples = []

  kernel_tallies = []
  prev_events_end = 0

  for name, (run_fn, _) in rht.KERNELS.items():
    hw_pkl = rht.CAPTURE_DIR / f"{name}.pkl"
    if not hw_pkl.exists(): continue

    with open(hw_pkl, "rb") as f:
      hw_traces = pickle.load(f)

    rht._clear()
    try:
      run_fn()
    except Exception as e:
      print(f"  {name}: SKIP ({e})")
      continue
    Device[Device.DEFAULT].synchronize()
    Device[Device.DEFAULT]._at_profile_finalize()
    emu_traces, _ = rht.extract_traces()

    k_shadow, prev_events_end = _kernel_shadow_slice(prev_events_end)

    hw_waves = sorted(hw_traces.keys())
    emu_waves = sorted(emu_traces.keys())
    if len(hw_waves) > len(emu_waves): continue
    n_common = min(len(hw_waves), len(emu_waves))

    # Build shadow index: (wave_id, stamp_cycle) → event dict
    shadow_idx: dict[tuple[int, int], dict] = {}
    for e in k_shadow:
      shadow_idx[(e['wave_id'], e['stamp_cycle'])] = e

    # Build HW MODAL set per token index
    k_min_len = 10**9
    for wi in range(n_common):
      hw = hw_traces[hw_waves[wi]]
      k_min_len = min(k_min_len, len(hw))

    hw_dt_at: dict[int, set[int]] = {}
    for wi in range(n_common):
      hw = hw_traces[hw_waves[wi]]
      for jj in range(1, min(k_min_len, len(hw))):
        dt = hw[jj][1] - hw[jj-1][1]
        if dt > 50: continue
        hw_dt_at.setdefault(jj, set()).add(dt)

    k_tagged = 0
    k_untagged = 0
    k_cluster_tokens = 0
    k_cluster_mismatches_strict = 0

    for wi in range(n_common):
      hw_wid, emu_wid = hw_waves[wi], emu_waves[wi]
      hw = hw_traces[hw_wid]
      emu = emu_traces[emu_wid]
      min_len = min(len(hw), len(emu))

      # PC-offset consistency (same as compare loop)
      hw_pc0 = hw[0][0] if hw else 0
      emu_pc0 = emu[0][0] if emu else 0
      pc_match = all((hw[j][0] - hw_pc0) == (emu[j][0] - emu_pc0) for j in range(min_len))
      if not pc_match: continue

      for j in range(min_len):
        if j == 0: continue
        hd = hw[j][1] - hw[j-1][1]
        ed = emu[j][1] - emu[j-1][1]
        if hd > 50 or ed > 50: continue

        emu_stamp = emu[j][1]
        ev = shadow_idx.get((emu_wid, emu_stamp))
        if ev is None:
          k_untagged += 1
          continue
        k_tagged += 1

        # Strict exact?
        strict_ex = (hd == ed)
        # MODAL exact?
        allowed = hw_dt_at.get(j, {hd})
        modal_ex = ed in allowed

        # Bucket by peer_cluster_2cy (no clamp — 32-wave kernels can have 7 peers on a SIMD)
        c2 = min(ev['peer_cluster_2cy'], 7)
        buckets_strict[c2][0] += 1 if strict_ex else 0
        buckets_strict[c2][1] += 1
        buckets_modal[c2][0] += 1 if modal_ex else 0
        buckets_modal[c2][1] += 1

        # Bucket by would_stall presence
        key = "any" if ev['would_stall_cy'] > 0 else "none"
        would_stall_buckets[key][0] += 1 if strict_ex else 0
        would_stall_buckets[key][1] += 1 if modal_ex else 0
        would_stall_buckets[key][2] += 1

        # Per-kernel MODAL miss tracking
        kmm = kernel_modal_miss.setdefault(name, {c: [0, 0] for c in range(8)})
        kmm[c2][0] += 0 if modal_ex else 1
        kmm[c2][1] += 1

        # Collect samples of high-cluster MODAL misses (these are the actionable mismatches)
        if c2 >= 4 and not modal_ex and len(high_cluster_modal_miss_samples) < 200:
          inst = emu[j][3] if len(emu[j]) > 3 else ""
          # wave_idx in simulation: simulation index from shadow event (not wave_id)
          high_cluster_modal_miss_samples.append(
            (name, ev['wave_idx'], emu_wid, j, hd, ed, sorted(allowed), c2, ev['would_stall_cy'], inst[:50])
          )

        if c2 >= 1:
          k_cluster_tokens += 1
          if not strict_ex: k_cluster_mismatches_strict += 1

    kernel_tallies.append((name, k_tagged, k_cluster_tokens, k_cluster_mismatches_strict))

  # Report
  print()
  print("  Bucket by peer_cluster_2cy (N peer waves on same SIMD ready within ±2cy):")
  print(f"  {'cluster_n':>10}  {'strict_exact':>14}  {'modal_exact':>14}  {'total':>10}")
  for c in sorted(buckets_strict.keys()):
    se, st = buckets_strict[c]
    me, mt = buckets_modal[c]
    if st == 0: continue
    print(f"  {c:>10}  {se:>6}/{st:<6} ({100*se/st:4.1f}%)  {me:>6}/{mt:<6} ({100*me/mt:4.1f}%)  {st:>10}")

  print()
  print("  Bucket by would_stall_cy>0 (naive-apply would force a stall):")
  print(f"  {'bucket':>10}  {'strict_exact':>14}  {'modal_exact':>14}  {'total':>10}")
  for k, (se, me, tot) in would_stall_buckets.items():
    if tot == 0: continue
    print(f"  {k:>10}  {se:>6}/{tot:<6} ({100*se/tot:4.1f}%)  {me:>6}/{tot:<6} ({100*me/tot:4.1f}%)  {tot:>10}")

  total_tagged = sum(t[1] for t in kernel_tallies)
  total_cluster = sum(t[2] for t in kernel_tallies)
  total_cluster_miss = sum(t[3] for t in kernel_tallies)
  print()
  print(f"  Tagged VALU tokens: {total_tagged}")
  print(f"  Cluster≥1 tokens: {total_cluster}  ({100*total_cluster/max(1,total_tagged):.1f}% of tagged)")
  if total_cluster > 0:
    print(f"  Of which strict-mismatches: {total_cluster_miss}  "
          f"({100*total_cluster_miss/total_cluster:.1f}% strict-miss rate in cluster bucket)")

  # Top 20 kernels by cluster-region mismatch density
  print()
  print("  Top kernels by cluster-region strict-mismatch density:")
  kernel_tallies.sort(key=lambda t: (-t[3], -t[2]))
  for name, tag, cl, miss in kernel_tallies[:20]:
    if cl == 0: continue
    print(f"    {name:45s}  cluster={cl:4d}  miss={miss:4d}  ({100*miss/cl:4.1f}%)")

  # Top 20 kernels by HIGH-cluster (≥4) MODAL miss density — the actionable signal
  print()
  print("  Top kernels by HIGH-cluster (≥4 peers) MODAL miss (actionable — emu lands outside any HW wave):")
  print(f"  {'kernel':45s}  {'modal_miss':>12}  {'high_cluster_total':>20}  {'miss%':>6}")
  kernel_high = []
  for name, per_c in kernel_modal_miss.items():
    hi_miss = sum(per_c[c][0] for c in range(4, 8))
    hi_tot = sum(per_c[c][1] for c in range(4, 8))
    if hi_tot == 0: continue
    kernel_high.append((name, hi_miss, hi_tot))
  kernel_high.sort(key=lambda t: -t[1])
  for name, hi_miss, hi_tot in kernel_high[:20]:
    if hi_miss == 0: continue
    print(f"    {name:45s}  {hi_miss:>12}  {hi_tot:>20}  {100*hi_miss/hi_tot:5.1f}%")

  # Sample mismatches at high-cluster MODAL miss sites
  print()
  print("  Sample high-cluster MODAL-miss tokens (cluster≥4, ed not in HW set):")
  # Group by kernel+j, show how many waves (sim idx) miss at each position
  from collections import defaultdict
  grouped = defaultdict(list)
  for name, sim_idx, wid, j, hd, ed, allowed, c2, ws, inst in high_cluster_modal_miss_samples:
    grouped[(name, j)].append((sim_idx, hd, ed, allowed, c2, inst))
  for (name, j), entries in sorted(grouped.items(), key=lambda kv: -len(kv[1]))[:15]:
    sim_idxs = sorted(e[0] for e in entries)
    allowed_samples = set()
    for _, hd, ed, allowed, c2, _ in entries:
      allowed_samples.add(tuple(allowed))
    sim_range = f"[{min(sim_idxs)}..{max(sim_idxs)}]" if len(sim_idxs) > 3 else str(sim_idxs)
    first = entries[0]
    print(f"    {name:40s} j={j:3d} n_waves={len(entries):2d} sim={sim_range}  HW_set={first[3]} EMU={first[2]} | {first[5]}")


if __name__ == "__main__":
  main()
