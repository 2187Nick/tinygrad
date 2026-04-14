#!/usr/bin/env python3
"""Validate emulator SQTT timing against real hardware captures.
Run on the same machine after hw_capture.py:
  DEV=AMD MOCKGPU=1 PYTHON_REMU=1 PROFILE=1 SQTT=1 python extra/sqtt/hw_validate.py extra/sqtt/captures/<timestamp>

Or just analyze existing captures without emulator (works anywhere):
  python extra/sqtt/hw_validate.py extra/sqtt/captures/<timestamp> --analyze-only
"""
import sys, json, pickle, pathlib, argparse
from collections import defaultdict

TARGET = "gfx1100"

def load_pkl(path):
  with open(path, "rb") as f:
    return pickle.load(f)

def extract_traces(data, target):
  from tinygrad.renderer.amd.sqtt import map_insts
  sqtt_events = [e for e in data if type(e).__name__ == "ProfileSQTTEvent"]
  kern_events = {e.tag: e for e in data if type(e).__name__ == "ProfileProgramEvent"}
  all_traces = {}
  for ev in sqtt_events:
    if not ev.itrace or ev.kern not in kern_events: continue
    prg = kern_events[ev.kern]
    traces = {}
    for pkt, info in map_insts(ev.blob, prg.lib, target):
      if info is None: continue
      w = info.wave
      if w not in traces: traces[w] = []
      traces[w].append({"pc": info.pc, "time": pkt._time, "type": type(pkt).__name__, "inst": str(info.inst) if info.inst else ""})
    if traces:
      all_traces[ev.kern] = {"traces": traces, "lib": prg.lib}
  return all_traces

def compute_deltas(trace_list):
  return [0] + [trace_list[i]["time"] - trace_list[i-1]["time"] for i in range(1, len(trace_list))]

def classify_inst(inst_str):
  """Classify instruction as DRAM, LDS, SALU, VALU, SMEM, or BARRIER."""
  inst = inst_str.lower()
  if any(x in inst for x in ["global_load", "global_store", "buffer_load", "buffer_store"]): return "DRAM"
  if any(x in inst for x in ["ds_store", "ds_load", "ds_read", "ds_write"]): return "LDS"
  if "s_barrier" in inst: return "BARRIER"
  if any(x in inst for x in ["s_waitcnt", "s_wait"]): return "WAIT"
  if inst.startswith("s_"): return "SALU"
  if inst.startswith("v_"): return "VALU"
  return "OTHER"

def find_non_dram_windows(trace_list):
  """Find contiguous non-DRAM instruction windows (no DRAM waits)."""
  windows, current = [], []
  in_dram_wait = False
  for i, entry in enumerate(trace_list):
    cat = classify_inst(entry["inst"])
    if cat == "DRAM":
      if current and len(current) >= 2:
        windows.append(current[:])
      current = []
      in_dram_wait = True
    elif cat == "WAIT" and in_dram_wait:
      current = []  # wait after DRAM — still in DRAM zone
    else:
      in_dram_wait = False
      current.append(i)
  if current and len(current) >= 2:
    windows.append(current)
  return windows

def analyze_captures(capture_dir):
  """Analyze all captures for determinism and non-DRAM windows."""
  manifest_path = capture_dir / "manifest.json"
  if not manifest_path.exists():
    print(f"ERROR: No manifest.json in {capture_dir}")
    return None

  with open(manifest_path) as f:
    manifest = json.load(f)

  print(f"Arch: {manifest['arch']}")
  print(f"Captured: {manifest['timestamp']}")
  print()

  report = {"arch": manifest["arch"], "kernels": {}}

  for kernel_name, info in manifest["kernels"].items():
    print(f"\n{'='*70}")
    print(f"KERNEL: {kernel_name}")
    print(f"{'='*70}")

    if info["status"] != "ok":
      print(f"  SKIPPED — capture status: {info['status']}")
      report["kernels"][kernel_name] = {"status": info["status"]}
      continue

    # load all successful runs
    runs_data = []
    for run_info in info["runs"]:
      if run_info["status"] != "ok": continue
      pkl_path = pathlib.Path(run_info["file"])
      if not pkl_path.exists():
        print(f"  WARNING: {pkl_path} not found, skipping")
        continue
      data = load_pkl(pkl_path)
      traces = extract_traces(data, TARGET)
      runs_data.append({"run": run_info["run"], "traces": traces})

    if not runs_data:
      print("  No valid runs loaded")
      report["kernels"][kernel_name] = {"status": "no_data"}
      continue

    # analyze first run in detail
    first_run = runs_data[0]
    kernel_report = {"status": "ok", "waves": {}, "deterministic_deltas": {}}

    for kern_tag, kern_data in first_run["traces"].items():
      traces = kern_data["traces"]
      print(f"\n  Kern tag={kern_tag}: {len(traces)} waves")

      for wid in sorted(traces.keys()):
        wave = traces[wid]
        deltas = compute_deltas(wave)
        types = [w["type"] for w in wave]
        cats = [classify_inst(w["inst"]) for w in wave]

        print(f"\n  Wave {wid}: {len(wave)} instructions")
        print(f"    Types: {types}")
        print(f"    Categories: {cats}")
        print(f"    Deltas: {deltas}")

        # find non-DRAM windows
        windows = find_non_dram_windows(wave)
        if windows:
          print(f"    Non-DRAM windows: {len(windows)}")
          for wi, window in enumerate(windows):
            win_deltas = [deltas[i] for i in window]
            win_insts = [wave[i]["inst"][:40] for i in window]
            print(f"      Window {wi} (idx {window[0]}-{window[-1]}): deltas={win_deltas}")
            for idx, (d, inst) in zip(window, zip(win_deltas, win_insts)):
              print(f"        [{idx:3d}] delta={d:4d}  {inst}")

        # check determinism across runs
        deterministic = True
        for other_run in runs_data[1:]:
          if kern_tag not in other_run["traces"]: continue
          other_traces = other_run["traces"][kern_tag]["traces"]
          if wid not in other_traces: continue
          other_deltas = compute_deltas(other_traces[wid])
          if deltas != other_deltas:
            deterministic = False
            diffs = [(i, deltas[i], other_deltas[i]) for i in range(min(len(deltas), len(other_deltas))) if deltas[i] != other_deltas[i]]
            print(f"    NONDETERMINISTIC vs run_{other_run['run']}: {len(diffs)} diffs at {[d[0] for d in diffs]}")
            for idx, d1, d2 in diffs[:10]:
              print(f"      idx {idx}: {d1} vs {d2} (diff={abs(d1-d2)})")

        if deterministic:
          print(f"    DETERMINISTIC across all {len(runs_data)} runs ✓")

        kernel_report["waves"][str(wid)] = {
          "n_insts": len(wave), "deltas": deltas, "types": types, "categories": cats,
          "deterministic": deterministic, "non_dram_windows": [[deltas[i] for i in w] for w in windows]
        }

    report["kernels"][kernel_name] = kernel_report

  # save report
  report_path = capture_dir / "analysis_report.json"
  with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
  print(f"\n\nFull report saved to: {report_path}")

  # summary
  print(f"\n{'='*70}")
  print("SUMMARY — Deterministic Non-DRAM Deltas Found")
  print(f"{'='*70}")
  total_det = 0
  for kname, kinfo in report["kernels"].items():
    if kinfo.get("status") != "ok": continue
    for wid_str, winfo in kinfo.get("waves", {}).items():
      if winfo["deterministic"]:
        for win in winfo["non_dram_windows"]:
          total_det += len(win)
          print(f"  {kname} wave {wid_str}: {len(win)} deltas {win}")
  print(f"\nTotal deterministic non-DRAM deltas: {total_det}")
  print(f"\nCopy the captures/ folder and share analysis_report.json for emulator validation.")
  return report

def run_emulator_comparison(capture_dir):
  """Run emulator on same kernels and compare timing (requires MOCKGPU environment)."""
  import functools
  from tinygrad import Device, Tensor, dtypes
  from tinygrad.device import Compiled, ProfileEvent, ProfileProgramEvent, ProfileDeviceEvent
  from tinygrad.renderer.amd.sqtt import map_insts
  from test.amd.test_custom_kernel import custom_lds_sync, custom_data_deps, custom_wave_sync
  from test.amd.helpers import TARGET_TO_ARCH

  arch = TARGET_TO_ARCH[Device["AMD"].arch]
  manifest_path = capture_dir / "manifest.json"
  with open(manifest_path) as f:
    manifest = json.load(f)

  print(f"\n{'='*70}")
  print("EMULATOR vs HARDWARE COMPARISON")
  print(f"{'='*70}")

  # map kernel names to functions that produce SQTT traces
  kernel_runners = {
    "sync": lambda: _run_custom(custom_lds_sync, arch, Tensor.empty(128, dtype=dtypes.int32).contiguous().realize()),
    "data_deps": lambda: _run_custom(custom_data_deps, arch, Tensor.full((32,), 5.0).realize()),
    "wave_sync": lambda: _run_custom(custom_wave_sync, arch, Tensor.empty(128, dtype=dtypes.int32).contiguous().realize()),
  }

  def _run_custom(fn, arch, inp):
    Device[Device.DEFAULT].synchronize()
    Compiled.profile_events[:] = [e for e in Compiled.profile_events if isinstance(e, (ProfileProgramEvent, ProfileDeviceEvent))]
    Tensor.custom_kernel(inp, fxn=functools.partial(fn, arch=arch))[0].realize()
    Device[Device.DEFAULT].synchronize()
    Device[Device.DEFAULT]._at_profile_finalize()

  results = {}
  for kernel_name in ["sync", "data_deps", "wave_sync"]:
    if kernel_name not in manifest["kernels"] or manifest["kernels"][kernel_name]["status"] != "ok":
      print(f"\n  {kernel_name}: skipped (no HW capture)")
      continue

    print(f"\n  Running emulator for: {kernel_name}")
    if kernel_name not in kernel_runners:
      print(f"    No runner defined, skipping")
      continue

    # run through emulator
    kernel_runners[kernel_name]()
    sqtt_events = [e for e in Compiled.profile_events if type(e).__name__ == "ProfileSQTTEvent"]
    kern_events = {e.tag: e for e in Compiled.profile_events if type(e).__name__ == "ProfileProgramEvent"}

    emu_traces = {}
    for ev in sqtt_events:
      if not ev.itrace or ev.kern not in kern_events: continue
      prg = kern_events[ev.kern]
      for pkt, info in map_insts(ev.blob, prg.lib, TARGET):
        if info is None: continue
        w = info.wave
        if w not in emu_traces: emu_traces[w] = []
        emu_traces[w].append({"pc": info.pc, "time": pkt._time, "type": type(pkt).__name__})

    # load HW capture
    hw_pkl = pathlib.Path(manifest["kernels"][kernel_name]["runs"][0]["file"])
    hw_data = load_pkl(hw_pkl)
    hw_all = extract_traces(hw_data, TARGET)
    if not hw_all:
      print(f"    No HW traces extracted")
      continue

    hw_traces = list(hw_all.values())[0]["traces"]

    # compare wave by wave
    emu_wids = sorted(emu_traces.keys())
    hw_wids = sorted(hw_traces.keys())
    n_waves = min(len(emu_wids), len(hw_wids))

    kernel_results = {"waves": {}, "total_deltas": 0, "matched": 0, "mismatched": 0}
    for wi in range(n_waves):
      emu_wave = emu_traces[emu_wids[wi]]
      hw_wave = hw_traces[hw_wids[wi]]

      emu_deltas = [0] + [emu_wave[i]["time"] - emu_wave[i-1]["time"] for i in range(1, len(emu_wave))]
      hw_deltas = compute_deltas(hw_wave)

      # find non-DRAM windows in HW trace
      windows = find_non_dram_windows(hw_wave)
      print(f"\n    Wave {wi}: {len(emu_wave)} emu insts, {len(hw_wave)} hw insts, {len(windows)} non-DRAM windows")

      wave_results = {"matches": [], "mismatches": []}
      for window in windows:
        for idx in window:
          if idx >= len(emu_deltas) or idx >= len(hw_deltas): continue
          hw_d, emu_d = hw_deltas[idx], emu_deltas[idx]
          diff = abs(hw_d - emu_d)
          kernel_results["total_deltas"] += 1
          if diff == 0:
            kernel_results["matched"] += 1
            wave_results["matches"].append(idx)
          else:
            kernel_results["mismatched"] += 1
            wave_results["mismatches"].append({"idx": idx, "hw": hw_d, "emu": emu_d, "diff": diff})
            print(f"      MISMATCH idx {idx}: HW={hw_d} EMU={emu_d} diff={diff}")

      if not wave_results["mismatches"]:
        print(f"      ALL non-DRAM deltas match ✓")
      kernel_results["waves"][str(wi)] = wave_results

    results[kernel_name] = kernel_results
    t = kernel_results["total_deltas"]
    m = kernel_results["matched"]
    print(f"\n    {kernel_name}: {m}/{t} deltas exact match ({m/t*100:.1f}%)" if t else f"\n    {kernel_name}: no comparable deltas")

  # final summary
  print(f"\n{'='*70}")
  print("FINAL RESULTS")
  print(f"{'='*70}")
  grand_total, grand_match = 0, 0
  for kname, kres in results.items():
    t, m = kres["total_deltas"], kres["matched"]
    grand_total += t
    grand_match += m
    pct = f"{m/t*100:.1f}%" if t else "N/A"
    status = "EXACT MATCH ✓" if m == t and t > 0 else f"{t-m} mismatches"
    print(f"  {kname:15s}: {m:3d}/{t:3d} ({pct}) — {status}")
  pct = f"{grand_match/grand_total*100:.1f}%" if grand_total else "N/A"
  print(f"\n  TOTAL: {grand_match}/{grand_total} ({pct})")

  report_path = capture_dir / "emulator_comparison.json"
  with open(report_path, "w") as f:
    json.dump(results, f, indent=2)
  print(f"\nDetailed comparison saved to: {report_path}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Validate emulator SQTT timing against real hardware")
  parser.add_argument("capture_dir", type=pathlib.Path, help="Path to captures directory from hw_capture.py")
  parser.add_argument("--analyze-only", action="store_true", help="Only analyze HW captures (no emulator comparison)")
  args = parser.parse_args()

  if not args.capture_dir.exists():
    print(f"ERROR: {args.capture_dir} does not exist")
    sys.exit(1)

  report = analyze_captures(args.capture_dir)

  if not args.analyze_only:
    try:
      run_emulator_comparison(args.capture_dir)
    except ImportError as e:
      print(f"\nCannot run emulator comparison (missing deps: {e})")
      print("Run with MOCKGPU environment or use --analyze-only to just analyze HW captures.")
