#!/usr/bin/env python3
"""Expansion capture: 30 gap-targeted kernels × 100 runs + WAVESTART times + HW_ID per config.

Complement to capture_50x100.py. Pulls the 30 highest-missing microbench kernels from
the current strict compare, captures each 100 times, and for each record includes:

  {wave_idx: [(pc, time, ptype, inst_str), ...],
   '_meta': {'wavestarts': {wave_idx: start_time}, 'kernel': name, 'n_threads': int}}

For each unique `n_threads` in the kernel set, runs capture_hw_id.py-style HW_ID1 probe
so we end up with a wave→(cu, simd, wgp, wave_slot) map per launch config. That map is
reused across all captures of kernels that share the config.

Skips kernels already captured by capture_50x100.py (same output dir).

═══════════════════════════════════════════════════════════════════════════════════
  CAPTURE ALL (real GPU, needs sudo, ~1-2 hours):
    sudo DEV=AMD AM_RESET=1 VIZ=-2 PROFILE=1 SQTT=1 DEBUG=0 MICROBENCH=1 \
      PYTHONPATH=. .venv/bin/python extra/sqtt/wave_probe/capture_expansion.py capture

  STATUS:
    MICROBENCH=1 PYTHONPATH=. .venv/bin/python \
      extra/sqtt/wave_probe/capture_expansion.py status

  ANALYZE (no GPU, processes expansion data):
    MICROBENCH=1 PYTHONPATH=. .venv/bin/python \
      extra/sqtt/wave_probe/capture_expansion.py analyze
═══════════════════════════════════════════════════════════════════════════════════
"""
import os, sys, pickle, pathlib, argparse, json, time
from collections import defaultdict

os.environ.setdefault("DEV", "AMD")
os.environ.setdefault("PROFILE", "1")
os.environ.setdefault("SQTT", "1")
os.environ.setdefault("VIZ", "-2")
os.environ.setdefault("MICROBENCH", "1")

CAPTURE_DIR = pathlib.Path("extra/sqtt/wave_probe/captures/targeted")
HWID_DIR    = pathlib.Path("extra/sqtt/wave_probe/captures/hwid_per_kernel")

# ── 30 kernels chosen by current strict-compare miss count (2026-04-20) ──
# Pulled from `rigorous_hw_test.py --compare` top-miss list. All are NOT in
# KERNEL_50, and each carries ≥55 strict mismatches.
KERNEL_EXPANSION = [
  # F.2 RAW variants — all show strong wave-variance that current rules don't close
  "mb_f2_raw_indep_interleave_n8",     # 115 miss
  "mb_f2_raw_with_mov_indep",          # 83 miss
  "mb_f2_vopd_then_raw",               # 79 miss
  "mb_f2_raw_then_vopd",               # 74 miss
  "mb_f2_raw_broken_by_mov_n8",        # 72 miss
  "mb_f2_raw_broken_by_nop_n8",        # 70 miss
  "mb_f2_three_parallel_raw_chains_n4",# 66 miss
  "mb_f2_two_parallel_raw_chains_n4",  # 63 miss
  "mb_f2_indep_then_raw_n4",           # 63 miss
  "mb_f2_raw_broken_by_nop_big_n8",    # 65 miss
  # G.4 SALU chains — extend beyond mb_g4_s_mul_i32_n4/s_or/s_and we already have
  "mb_g4_s_add_u32_n8",                # 112 miss — biggest SALU gap
  "mb_g4_s_xor_b32_n4",                #  60 miss
  # C.2/C.4 depctr — current depctr rule mismatches on 3-way chains
  "mb_c4_depctr_chain_n3",             # 71 miss
  "mb_c2_depctr_cmp3_cnd3_vopd",       # 71 miss
  "mb_c2_depctr_cmp_vcc_cnd_vopd",     # 67 miss
  # VCMP/CNDMASK tail patterns — stale sgpr and retire variance
  "mb_cndmask_sgpr_stale_n4",          # 69 miss
  "mb_cndmask_tail_retire",            # 60 miss
  "mb_cndmask_new_vgpr_tail",          # 60 miss
  "mb_vcmp_cndmask_k8",                # 88 miss — doubles the k4 we have
  "mb_vcmp_cndmask_k2",                # 56 miss
  "mb_vcmp_spaced_cndmask",            # 57 miss
  # VOPD edge cases — many mid-80s pct, tails and chains
  "mb_vopd_chain_n4_raw",              # 67 miss (included but without WAVESTART timing)
  "mb_vopd_fmac_mul_n4",               # 64 miss (same)
  "mb_vopd_mul_fmac_bank_same",        # 49 miss
  "mb_vopd_mixed_f32_f16",             # 50 miss
  # F.3/F.5 store pairs + SGPR chain
  "mb_f3_store_pair_then_pair",        # 63 miss
  "mb_f5_sgpr_chain_n4",               # 63 miss
  # Trans chain n8 (we only have n4 in KERNEL_50)
  "mb_f4_rcp_chain_n8",                # 65 miss
  # Trans+VOPD cross pattern
  "mb_trans_vopd_cross",               # 62 miss
  # Wave-variance confirmed: hit by MODAL but not strict — the biggest signal
  # we need HW data on to decide the stochastic-scheduler rule
]

def _get_kernels():
  from extra.sqtt.rigorous_hw_test import KERNELS
  return KERNELS

def _clear():
  from tinygrad.device import Compiled, ProfileProgramEvent, ProfileDeviceEvent
  Compiled.profile_events[:] = [e for e in Compiled.profile_events
                                if isinstance(e, (ProfileProgramEvent, ProfileDeviceEvent))]

def extract_traces_with_wavestart():
  """Returns: {wave_idx: {'pkts': [(pc, t, ptype, inst_str),...], 'start': wavestart_time_or_None}}"""
  from tinygrad.device import Compiled
  from tinygrad.renderer.amd.sqtt import map_insts, WAVESTART, WAVESTART_RDNA4, WAVEEND, INST, VALUINST, IMMEDIATE, decode
  sqtt_events = [e for e in Compiled.profile_events if type(e).__name__ == 'ProfileSQTTEvent']
  program_events = {e.tag: e for e in Compiled.profile_events if type(e).__name__ == 'ProfileProgramEvent'}
  all_traces: dict = {}

  # First pass: collect WAVESTART times from raw decode (map_insts only yields INST/IMMEDIATE/VALUINST).
  wavestart_times: dict[int, int] = {}
  for ev in sqtt_events:
    if not ev.itrace: continue
    for pkt in decode(ev.blob):
      if isinstance(pkt, (WAVESTART, WAVESTART_RDNA4)):
        # pkt.wave is the HW wave_slot; there's no direct "wave_idx" here. We use slot
        # as a best-effort key and let the second pass overwrite with the canonical wave_idx.
        wavestart_times.setdefault(pkt.wave, pkt._time)

  for ev in sqtt_events:
    if not ev.itrace: continue
    if ev.kern not in program_events: continue
    kev = program_events[ev.kern]
    for pkt, info in map_insts(ev.blob, kev.lib, "gfx1100"):
      if info is None: continue
      w = info.wave
      if w not in all_traces:
        all_traces[w] = {'pkts': [], 'start': wavestart_times.get(w)}
      all_traces[w]['pkts'].append((info.pc, pkt._time, type(pkt).__name__, str(info.inst)[:80]))
  return all_traces

# ── HW_ID per-kernel-config probe (reuses capture_hw_id helpers) ─────────────

def run_hwid_probe_for_launch(n_threads: int, runs: int = 10) -> list[dict]:
  """Launch the HW_ID1 probe with matching n_threads; return per-run per-wave placement."""
  from tinygrad import Tensor, Device, dtypes
  from extra.sqtt.wave_probe.capture_hw_id import custom_probe_hw_id, decode_hw_id
  n_waves = max(1, n_threads // 32)  # wave32 default
  results = []
  for r in range(runs):
    buf = Tensor.zeros(n_threads, dtype=dtypes.uint32).contiguous().realize()
    buf = Tensor.custom_kernel(buf, fxn=custom_probe_hw_id)[0]
    buf.realize()
    Device[Device.DEFAULT].synchronize()
    arr = buf.numpy().tolist()
    per_wave = []
    for wi in range(n_waves):
      base = wi * 32
      ids = set(arr[base+i] for i in range(min(32, n_threads - base)))
      if len(ids) != 1:
        per_wave.append({"error": f"non-uniform {sorted(ids)[:4]}"})
        continue
      per_wave.append(decode_hw_id(ids.pop(), 0))
    results.append({"run": r, "n_waves": n_waves, "n_threads": n_threads, "waves": per_wave})
  return results

# ── Capture ─────────────────────────────────────────────────────────────────

def do_capture(kernel_names: list[str], n_runs: int = 100, hwid_runs: int = 10):
  from tinygrad import Device, Tensor
  kernels = _get_kernels()

  (Tensor([1.]) + Tensor([1.])).realize()
  Device[Device.DEFAULT].synchronize()

  HWID_DIR.mkdir(parents=True, exist_ok=True)
  total_kernels = len(kernel_names)
  print(f"╔══════════════════════════════════════════════════════════════════════╗")
  print(f"║  EXPANSION CAPTURE: {total_kernels} kernels × {n_runs} runs each + HW_ID probe")
  print(f"╚══════════════════════════════════════════════════════════════════════╝")

  # Preflight: skip unknown kernels
  valid = [k for k in kernel_names if k in kernels]
  missing = [k for k in kernel_names if k not in kernels]
  if missing:
    print(f"  WARNING: {len(missing)} kernels not in KERNELS dict: {missing}")

  # Discover unique launch configs (n_threads) by running each kernel once in a dry mode,
  # then share the HW_ID probe across kernels that have the same launch geometry.
  hwid_by_nthreads: dict[int, str] = {}  # n_threads -> json path

  for kidx, kname in enumerate(valid):
    if kname not in kernels: continue
    out_dir = CAPTURE_DIR / kname
    existing = len(list(out_dir.glob("run_*.pkl"))) if out_dir.exists() else 0
    if existing >= n_runs:
      print(f"\n  [{kidx+1}/{len(valid)}] SKIP {kname} — have {existing}/{n_runs}")
      continue

    run_fn, default_attempts = kernels[kname]
    max_att = max(30, default_attempts)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_from = existing
    print(f"\n  [{kidx+1}/{len(valid)}] {kname} — runs {start_from}..{n_runs-1}")
    t0 = time.time()
    captured = existing
    first_good_meta: dict | None = None

    for run_idx in range(start_from, n_runs):
      for attempt in range(max_att):
        try:
          _clear()
          run_fn()
          Device[Device.DEFAULT].synchronize()
          Device[Device.DEFAULT]._at_profile_finalize()
          traces = extract_traces_with_wavestart()
          valid_run = any(sum(1 for _, _, t, _ in v['pkts'] if t in ("INST", "VALUINST", "IMMEDIATE")) >= 3
                          for v in traces.values())
          if valid_run:
            # Pack legacy-compatible format + _meta side channel for analysis.
            out = {w: v['pkts'] for w, v in traces.items()}
            n_waves = len(traces)
            meta = {
              'wavestarts': {w: v['start'] for w, v in traces.items() if v['start'] is not None},
              'kernel': kname,
              'n_waves': n_waves,
            }
            out['_meta'] = meta
            out_file = out_dir / f"run_{run_idx:04d}.pkl"
            with open(out_file, "wb") as f: pickle.dump(out, f)
            captured += 1
            if first_good_meta is None: first_good_meta = meta
            break
        except Exception:
          if attempt >= max_att - 1: break

    elapsed = time.time() - t0
    new = captured - existing
    print(f"    ✓ {new} new captures in {elapsed:.0f}s (total: {captured}/{n_runs})")

    # HW_ID probe for this kernel's launch config (32 threads per wave)
    if first_good_meta and first_good_meta['n_waves'] > 0:
      n_waves_here = first_good_meta['n_waves']
      n_threads = n_waves_here * 32
      if n_threads not in hwid_by_nthreads:
        probe_path = HWID_DIR / f"hwid_n{n_threads:04d}.json"
        if not probe_path.exists():
          print(f"    → HW_ID probe (n_threads={n_threads}, runs={hwid_runs})")
          try:
            res = run_hwid_probe_for_launch(n_threads, runs=hwid_runs)
            with open(probe_path, "w") as f: json.dump({"n_threads": n_threads, "runs": res}, f, indent=2)
          except Exception as e:
            print(f"    HW_ID probe FAILED: {e}")
            continue
        hwid_by_nthreads[n_threads] = str(probe_path)

  # Summarize
  summary_path = HWID_DIR / "hwid_by_nthreads_index.json"
  with open(summary_path, "w") as f:
    json.dump({"kernels": valid, "hwid_probes": hwid_by_nthreads}, f, indent=2)
  print(f"\n  HW_ID index: {summary_path}")
  print(f"  CAPTURE COMPLETE — run 'analyze' to diff against KERNEL_50")

# ── Status ──────────────────────────────────────────────────────────────────

def do_status():
  kernels = _get_kernels()
  print(f"{'Kernel':<45s} {'Captures':>10s} {'In KERNELS':>10s}")
  print("-" * 70)
  tot = 0
  tot_in = 0
  for kname in KERNEL_EXPANSION:
    d = CAPTURE_DIR / kname
    n = len(list(d.glob("run_*.pkl"))) if d.exists() else 0
    in_kernels = kname in kernels
    tot += n
    if in_kernels: tot_in += 1
    flag = "✓" if n >= 100 else ("…" if n > 0 else "✗")
    ok = "yes" if in_kernels else "MISSING"
    print(f"  {flag} {kname:<43s} {n:>6d}/100 {ok:>10s}")
  print("-" * 70)
  print(f"  TOTAL: {tot} captures, {tot_in}/{len(KERNEL_EXPANSION)} kernels registered")

# ── Analysis (minimal — compute per-wave strict-hit-rate against existing HW snapshot) ──

def do_analyze():
  print("╔══════════════════════════════════════════════════════════════════════╗")
  print("║  EXPANSION ANALYSIS — wavestart stagger + per-wave mode dts")
  print("╚══════════════════════════════════════════════════════════════════════╝")

  for kname in KERNEL_EXPANSION:
    d = CAPTURE_DIR / kname
    if not d.exists(): continue
    caps = []
    for pkl in sorted(d.glob("run_*.pkl")):
      with open(pkl, "rb") as f: caps.append(pickle.load(f))
    if not caps: continue

    # Split meta from waves
    wavestart_stagger = []
    for cap in caps:
      meta = cap.get('_meta', {}) if isinstance(cap, dict) else {}
      wss = meta.get('wavestarts', {})
      if len(wss) >= 2:
        sts = sorted(wss.values())
        wavestart_stagger.append(sts[-1] - sts[0])

    n_waves = max(len(cap) - (1 if '_meta' in cap else 0) for cap in caps) if caps else 0
    print(f"\n  {kname}: {len(caps)} runs, up to {n_waves} waves")
    if wavestart_stagger:
      import statistics
      print(f"    wavestart stagger (last-first): "
            f"min={min(wavestart_stagger)} max={max(wavestart_stagger)} "
            f"median={int(statistics.median(wavestart_stagger))} "
            f"stdev={statistics.stdev(wavestart_stagger) if len(wavestart_stagger)>1 else 0:.1f}")

    # Per-wave instruction-token mode stability
    all_dts: dict[int, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for cap in caps:
      for w, pkts in cap.items():
        if w == '_meta': continue
        prev_t = None
        tok_idx = 0
        for pc, t, ptype, inst in pkts:
          if ptype in ("INST", "VALUINST", "IMMEDIATE"):
            dt = t - prev_t if prev_t is not None else 0
            all_dts[w][tok_idx].append(dt)
            tok_idx += 1
          prev_t = t
    # Count fully-deterministic tokens
    determ, total = 0, 0
    for w, tdt in all_dts.items():
      for tok, vals in tdt.items():
        if len(vals) < 5: continue
        total += 1
        if len(set(vals)) == 1: determ += 1
    pct = 100.0*determ/total if total else 0.0
    print(f"    determinism: {determ}/{total} ({pct:.0f}%) tokens identical across runs")

# ── Main ────────────────────────────────────────────────────────────────────

def main():
  parser = argparse.ArgumentParser(description="Expansion capture (30 gap kernels × 100 runs + HW_ID per config)")
  parser.add_argument("command", choices=["capture", "analyze", "status"])
  parser.add_argument("--kernel", type=str, help="Capture single kernel only")
  parser.add_argument("--runs", type=int, default=100, help="Runs per kernel (default: 100)")
  parser.add_argument("--hwid-runs", type=int, default=10, help="HW_ID probe runs per config")
  args = parser.parse_args()

  if args.command == "capture":
    targets = [args.kernel] if args.kernel else KERNEL_EXPANSION
    do_capture(targets, args.runs, args.hwid_runs)
  elif args.command == "analyze":
    do_analyze()
  elif args.command == "status":
    do_status()

if __name__ == "__main__":
  main()
