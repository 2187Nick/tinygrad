#!/usr/bin/env python3
"""Hardware validation suite: capture SQTT traces from real AMD 7900 XTX and compare against emulator.

Capture mode (real HW, requires sudo + AM_RESET):
  echo '<pw>' | sudo -S bash -c 'echo profile_standard > /sys/class/drm/card1/device/power_dpm_force_performance_level'
  sudo DEV=AMD AM_RESET=1 VIZ=-2 PROFILE=1 SQTT=1 DEBUG=0 PYTHONPATH=. .venv/bin/python extra/sqtt/hw_validation_suite.py --capture

Compare mode (emulator):
  DEV=AMD MOCKGPU=1 PYTHON_REMU=1 PROFILE=1 SQTT=1 PYTHONPATH=. .venv/bin/python extra/sqtt/hw_validation_suite.py --compare

Compare against existing captures:
  DEV=AMD MOCKGPU=1 PYTHON_REMU=1 PROFILE=1 SQTT=1 PYTHONPATH=. .venv/bin/python extra/sqtt/hw_validation_suite.py --compare --captures extra/sqtt/captures/<dir>
"""
import os, sys, pickle, argparse, functools
from pathlib import Path
from collections import Counter
from dataclasses import dataclass

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path: sys.path.insert(0, str(_repo_root))

TARGET = "gfx1100"
MAX_RETRIES = 30
CAPTURE_DIR = Path(__file__).resolve().parent / "captures" / "hw_validation"

# ═══════════════════════════════════════════════════════════════════════════════
# Kernel definitions
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class KernelSpec:
  name: str
  description: str
  run_fn: str  # method name on this module to call

KERNEL_SPECS = [
  KernelSpec("plus", "Tensor add: simple VALU + global store", "run_plus"),
  KernelSpec("custom_lds_sync", "LDS writes, barrier, LDS reads, VALU", "run_custom_lds_sync"),
  KernelSpec("custom_data_deps", "Global load, VALU add, global store", "run_custom_data_deps"),
  KernelSpec("reduce_sum", "Reduce sum with GROUP reduction", "run_reduce_sum"),
  KernelSpec("matmul_small", "16x16 matmul with complex scheduling", "run_matmul_small"),
]

def _reset_profile_events():
  """Clear profile events, keeping only device/program metadata."""
  from tinygrad.device import Compiled, ProfileDeviceEvent, ProfileProgramEvent
  Compiled.profile_events[:] = [e for e in Compiled.profile_events if isinstance(e, (ProfileProgramEvent, ProfileDeviceEvent))]

def _finalize_and_get_events():
  """Synchronize device, finalize profiling, and return all profile events."""
  from tinygrad import Device
  from tinygrad.device import Compiled
  dev = Device[Device.DEFAULT]
  dev.synchronize()
  dev._at_profile_finalize()
  return list(Compiled.profile_events)

def run_plus():
  from tinygrad import Tensor
  _reset_profile_events()
  (Tensor([1, 2, 3, 4]) + Tensor([5, 6, 7, 8])).realize()
  return _finalize_and_get_events()

def run_custom_lds_sync():
  from tinygrad import Tensor, Device, dtypes
  from tinygrad.device import Compiled
  from test.amd.test_custom_kernel import custom_lds_sync
  from test.amd.helpers import TARGET_TO_ARCH
  arch = TARGET_TO_ARCH[Device["AMD"].arch]
  _reset_profile_events()
  a = Tensor.empty(128, dtype=dtypes.int32).contiguous().realize()
  Tensor.custom_kernel(a, fxn=functools.partial(custom_lds_sync, arch=arch))[0].realize()
  return _finalize_and_get_events()

def run_custom_data_deps():
  import numpy as np
  from tinygrad import Tensor, Device
  from test.amd.test_custom_kernel import custom_data_deps
  from test.amd.helpers import TARGET_TO_ARCH
  arch = TARGET_TO_ARCH[Device["AMD"].arch]
  _reset_profile_events()
  a = Tensor(np.full(32, 5.0, dtype=np.float32)).realize()
  Tensor.custom_kernel(a, fxn=functools.partial(custom_data_deps, arch=arch))[0].realize()
  return _finalize_and_get_events()

def run_reduce_sum():
  from tinygrad import Tensor
  _reset_profile_events()
  Tensor.rand(256).sum().realize()
  return _finalize_and_get_events()

def run_matmul_small():
  from tinygrad import Tensor
  _reset_profile_events()
  (Tensor.rand(16, 16) @ Tensor.rand(16, 16)).realize()
  return _finalize_and_get_events()

# ═══════════════════════════════════════════════════════════════════════════════
# SQTT trace extraction and analysis
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TraceEntry:
  pc: int
  time: int
  pkt_type: str
  inst: str
  wave: int

def extract_sqtt_traces(events, target=TARGET):
  """Extract per-wave instruction traces from profile events. Returns dict of {kern_tag: {wave_id: [TraceEntry]}}."""
  from tinygrad.renderer.amd.sqtt import map_insts
  sqtt_events = [e for e in events if type(e).__name__ == "ProfileSQTTEvent"]
  prg_events = {e.tag: e for e in events if type(e).__name__ == "ProfileProgramEvent" and e.tag is not None}
  all_traces = {}
  for ev in sqtt_events:
    if not ev.itrace or ev.kern not in prg_events: continue
    prg = prg_events[ev.kern]
    if prg.lib is None: continue
    traces = {}
    for pkt, info in map_insts(ev.blob, prg.lib, target):
      if info is None: continue
      w = info.wave
      if w not in traces: traces[w] = []
      traces[w].append(TraceEntry(pc=info.pc, time=pkt._time, pkt_type=type(pkt).__name__, inst=str(info.inst) if info.inst else "", wave=w))
    if traces:
      tag = ev.kern
      # merge into existing if same kern (multiple SEs may have data)
      if tag in all_traces:
        for w, tlist in traces.items():
          if w not in all_traces[tag]: all_traces[tag][w] = tlist
      else:
        all_traces[tag] = traces
  return all_traces

def has_inst_packets(events):
  """Check if any SQTT event has instruction-level packets."""
  from tinygrad.renderer.amd.sqtt import decode, INST, VALUINST, IMMEDIATE
  for ev in events:
    if type(ev).__name__ != "ProfileSQTTEvent" or not ev.itrace: continue
    for pkt in decode(ev.blob):
      if isinstance(pkt, (INST, VALUINST, IMMEDIATE)): return True
  return False

def compute_deltas(trace_list):
  """Compute inter-instruction time deltas."""
  return [0] + [trace_list[i].time - trace_list[i - 1].time for i in range(1, len(trace_list))]

def classify_inst(inst_str):
  """Classify instruction as DRAM, LDS, SALU, VALU, SMEM, BARRIER, WAIT, or OTHER."""
  inst = inst_str.lower()
  if any(x in inst for x in ["global_load", "global_store", "buffer_load", "buffer_store"]): return "DRAM"
  if any(x in inst for x in ["s_load_", "s_buffer_load"]): return "SMEM"
  if any(x in inst for x in ["ds_store", "ds_load", "ds_read", "ds_write"]): return "LDS"
  if "s_barrier" in inst: return "BARRIER"
  if any(x in inst for x in ["s_waitcnt", "s_wait"]): return "WAIT"
  if inst.startswith("s_"): return "SALU"
  if inst.startswith("v_") or inst.startswith("vopd("): return "VALU"
  return "OTHER"

def find_non_dram_windows(trace_list):
  """Find contiguous non-DRAM instruction windows (indices into trace_list).
  Excludes DRAM loads, SMEM loads, and their associated WAITs — all have non-deterministic latency."""
  windows, current = [], []
  pending_mem = False  # True if we've seen DRAM/SMEM and are waiting for the stall to clear
  for i, entry in enumerate(trace_list):
    cat = classify_inst(entry.inst)
    if cat in ("DRAM", "SMEM"):
      if current and len(current) >= 2: windows.append(current[:])
      current = []
      pending_mem = True
    elif cat == "WAIT" and pending_mem:
      # This WAIT is draining a memory op — skip it and the next instruction (stall recovery)
      current = []
    else:
      pending_mem = False
      current.append(i)
  if current and len(current) >= 2: windows.append(current)
  return windows

# ═══════════════════════════════════════════════════════════════════════════════
# Capture mode — run on real hardware with retry for INST packets
# ═══════════════════════════════════════════════════════════════════════════════

def do_capture(output_dir):
  """Capture SQTT traces from real hardware for all kernels."""
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  results = {}

  for spec in KERNEL_SPECS:
    print(f"\n{'=' * 70}")
    print(f"CAPTURE: {spec.name} — {spec.description}")
    print(f"{'=' * 70}")

    runner = globals()[spec.run_fn]
    captured = False

    for attempt in range(1, MAX_RETRIES + 1):
      print(f"  attempt {attempt}/{MAX_RETRIES} ...", end=" ", flush=True)
      try:
        events = runner()
      except Exception as e:
        print(f"ERROR: {e}")
        continue

      if has_inst_packets(events):
        traces = extract_sqtt_traces(events)
        total_insts = sum(len(t) for wave_traces in traces.values() for t in wave_traces.values())
        print(f"GOT INST packets! ({total_insts} instructions across {sum(len(v) for v in traces.values())} waves)")

        # save the raw events
        pkl_path = output_dir / f"{spec.name}.pkl"
        with open(pkl_path, "wb") as f:
          pickle.dump(events, f)

        # also save extracted traces for quick analysis
        trace_data = {}
        for kern_tag, wave_dict in traces.items():
          trace_data[kern_tag] = {}
          for wid, tlist in wave_dict.items():
            deltas = compute_deltas(tlist)
            trace_data[kern_tag][wid] = {
              "entries": [{"pc": e.pc, "time": e.time, "pkt_type": e.pkt_type, "inst": e.inst, "wave": e.wave} for e in tlist],
              "deltas": deltas,
              "categories": [classify_inst(e.inst) for e in tlist],
            }

        traces_path = output_dir / f"{spec.name}_traces.pkl"
        with open(traces_path, "wb") as f:
          pickle.dump(trace_data, f)

        results[spec.name] = {"status": "ok", "attempts": attempt, "pkl": str(pkl_path), "traces_pkl": str(traces_path),
                              "n_instructions": total_insts, "n_waves": sum(len(v) for v in traces.values())}
        captured = True
        break
      else:
        print("no INST packets (wave not on traced CU)")

    if not captured:
      print(f"  FAILED after {MAX_RETRIES} attempts — no INST packets captured")
      results[spec.name] = {"status": "no_inst", "attempts": MAX_RETRIES}

  # summary
  print(f"\n{'=' * 70}")
  print("CAPTURE SUMMARY")
  print(f"{'=' * 70}")
  for name, info in results.items():
    status = "✓" if info["status"] == "ok" else "✗"
    extra = f" ({info.get('n_instructions', 0)} insts, {info.get('attempts', '?')} attempts)" if info["status"] == "ok" else ""
    print(f"  {status} {name:20s} {info['status']}{extra}")

  manifest_path = output_dir / "manifest.pkl"
  with open(manifest_path, "wb") as f:
    pickle.dump(results, f)
  print(f"\nCaptures saved to: {output_dir}")
  print(f"Manifest: {manifest_path}")
  return results

# ═══════════════════════════════════════════════════════════════════════════════
# Compare mode — run kernels through emulator and compare against HW captures
# ═══════════════════════════════════════════════════════════════════════════════

def do_compare(capture_dir=None):
  """Run kernels through emulator and compare non-DRAM deltas against HW captures (if available)."""
  # run all kernels through emulator to get SQTT traces
  emu_results = {}
  for spec in KERNEL_SPECS:
    print(f"\n{'=' * 70}")
    print(f"EMULATOR: {spec.name} — {spec.description}")
    print(f"{'=' * 70}")

    runner = globals()[spec.run_fn]
    try:
      events = runner()
    except Exception as e:
      print(f"  ERROR running kernel: {e}")
      emu_results[spec.name] = {"status": "error", "error": str(e)}
      continue

    traces = extract_sqtt_traces(events)
    if not traces:
      print(f"  WARNING: no SQTT traces extracted")
      emu_results[spec.name] = {"status": "no_traces"}
      continue

    total_insts = sum(len(t) for wave_traces in traces.values() for t in wave_traces.values())
    n_waves = sum(len(v) for v in traces.values())
    print(f"  Emulator: {total_insts} instructions across {n_waves} waves")

    # display per-wave details
    for kern_tag, wave_dict in traces.items():
      for wid in sorted(wave_dict.keys()):
        tlist = wave_dict[wid]
        deltas = compute_deltas(tlist)
        cats = [classify_inst(e.inst) for e in tlist]
        windows = find_non_dram_windows(tlist)
        print(f"    Wave {wid}: {len(tlist)} insts, categories={Counter(cats).most_common()}")
        print(f"      Deltas: {deltas}")
        if windows:
          for wi, window in enumerate(windows):
            win_deltas = [deltas[i] for i in window]
            print(f"      Non-DRAM window {wi} (idx {window[0]}-{window[-1]}): deltas={win_deltas}")
            for idx in window:
              print(f"        [{idx:3d}] delta={deltas[idx]:4d}  {cats[idx]:8s}  {tlist[idx].inst[:50]}")

    emu_results[spec.name] = {"status": "ok", "traces": traces}

  # if HW captures provided, compare
  if capture_dir is not None:
    capture_dir = Path(capture_dir)
    print(f"\n{'=' * 70}")
    print(f"COMPARISON: Emulator vs Hardware ({capture_dir})")
    print(f"{'=' * 70}")
    _compare_hw_emu(capture_dir, emu_results)
  else:
    print(f"\n{'=' * 70}")
    print("EMULATOR-ONLY SUMMARY (no HW captures to compare)")
    print(f"{'=' * 70}")
    for name, info in emu_results.items():
      if info["status"] != "ok":
        print(f"  ✗ {name:20s} {info['status']}")
        continue
      traces = info["traces"]
      total_insts = sum(len(t) for wt in traces.values() for t in wt.values())
      n_waves = sum(len(v) for v in traces.values())
      # count non-DRAM deltas
      n_non_dram = 0
      for wt in traces.values():
        for tlist in wt.values():
          windows = find_non_dram_windows(tlist)
          n_non_dram += sum(len(w) for w in windows)
      print(f"  ✓ {name:20s} {total_insts:4d} insts, {n_waves} waves, {n_non_dram} non-DRAM deltas")

  return emu_results

def _compare_hw_emu(capture_dir, emu_results):
  """Compare HW and emulator traces, reporting delta matches."""
  grand_total, grand_match, grand_close = 0, 0, 0
  kernel_summaries = []

  for spec in KERNEL_SPECS:
    name = spec.name
    traces_pkl = capture_dir / f"{name}_traces.pkl"
    raw_pkl = capture_dir / f"{name}.pkl"

    if name not in emu_results or emu_results[name]["status"] != "ok":
      print(f"\n  {name}: skipped (no emulator trace)")
      kernel_summaries.append((name, "no_emu", 0, 0, 0))
      continue

    # load HW traces
    hw_trace_data = None
    if traces_pkl.exists():
      with open(traces_pkl, "rb") as f:
        hw_trace_data = pickle.load(f)
    elif raw_pkl.exists():
      with open(raw_pkl, "rb") as f:
        hw_events = pickle.load(f)
      hw_raw_traces = extract_sqtt_traces(hw_events)
      hw_trace_data = {}
      for kern_tag, wave_dict in hw_raw_traces.items():
        hw_trace_data[kern_tag] = {}
        for wid, tlist in wave_dict.items():
          hw_trace_data[kern_tag][wid] = {
            "entries": [{"pc": e.pc, "time": e.time, "pkt_type": e.pkt_type, "inst": e.inst, "wave": e.wave} for e in tlist],
            "deltas": compute_deltas(tlist),
            "categories": [classify_inst(e.inst) for e in tlist],
          }
    else:
      print(f"\n  {name}: skipped (no HW capture at {capture_dir})")
      kernel_summaries.append((name, "no_hw", 0, 0, 0))
      continue

    emu_traces = emu_results[name]["traces"]
    print(f"\n  {name}:")

    # get first wave from each
    hw_waves = {}
    for kern_tag, wave_dict in hw_trace_data.items():
      for wid, wdata in wave_dict.items():
        hw_waves[wid] = wdata

    emu_waves = {}
    for kern_tag, wave_dict in emu_traces.items():
      for wid, tlist in wave_dict.items():
        emu_waves[wid] = tlist

    if not hw_waves:
      print(f"    No HW waves")
      kernel_summaries.append((name, "no_hw_waves", 0, 0, 0))
      continue

    # compare first available wave from each
    hw_wid = sorted(hw_waves.keys())[0]
    emu_wid = sorted(emu_waves.keys())[0]
    hw_wave = hw_waves[hw_wid]
    emu_tlist = emu_waves[emu_wid]
    emu_deltas = compute_deltas(emu_tlist)
    hw_deltas = hw_wave["deltas"]
    hw_cats = hw_wave["categories"]

    # reconstruct TraceEntry-like objects for non-DRAM window detection from HW data
    @dataclass
    class _FakeEntry:
      inst: str
    hw_entries = [_FakeEntry(inst=e["inst"]) for e in hw_wave["entries"]]
    hw_windows = find_non_dram_windows(hw_entries)

    total, matched, close = 0, 0, 0
    for window in hw_windows:
      for idx in window:
        if idx >= len(emu_deltas) or idx >= len(hw_deltas): continue
        hw_d, emu_d = hw_deltas[idx], emu_deltas[idx]
        diff = abs(hw_d - emu_d)
        total += 1
        if diff == 0:
          matched += 1
        elif diff <= 2:
          close += 1
          print(f"    CLOSE  [{idx:3d}] HW={hw_d:4d} EMU={emu_d:4d} diff={diff} {hw_cats[idx] if idx < len(hw_cats) else '?'}")
        else:
          print(f"    MISMATCH [{idx:3d}] HW={hw_d:4d} EMU={emu_d:4d} diff={diff} {hw_cats[idx] if idx < len(hw_cats) else '?'}")

    grand_total += total
    grand_match += matched
    grand_close += close

    if total > 0:
      pct = matched / total * 100
      pct_close = (matched + close) / total * 100
      status = "EXACT MATCH ✓" if matched == total else f"{total - matched} mismatches"
      print(f"    → {matched}/{total} exact ({pct:.1f}%), {matched + close}/{total} within ±2 ({pct_close:.1f}%) — {status}")
    else:
      print(f"    → no comparable non-DRAM deltas")
    kernel_summaries.append((name, "ok", total, matched, close))

  # final summary
  print(f"\n{'=' * 70}")
  print("FINAL COMPARISON RESULTS")
  print(f"{'=' * 70}")
  for name, status, total, matched, close in kernel_summaries:
    if status not in ("ok",):
      print(f"  ✗ {name:20s} {status}")
    elif total == 0:
      print(f"  - {name:20s} no comparable deltas")
    else:
      pct = matched / total * 100
      sym = "✓" if matched == total else "≈" if (matched + close) / total > 0.8 else "✗"
      print(f"  {sym} {name:20s} {matched:3d}/{total:3d} exact ({pct:.1f}%)")

  if grand_total > 0:
    print(f"\n  TOTAL: {grand_match}/{grand_total} exact ({grand_match / grand_total * 100:.1f}%), "
          f"{grand_match + grand_close}/{grand_total} within ±2 ({(grand_match + grand_close) / grand_total * 100:.1f}%)")
  else:
    print(f"\n  TOTAL: no comparable deltas")

# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="SQTT hardware validation suite for AMD 7900 XTX")
  mode = parser.add_mutually_exclusive_group(required=True)
  mode.add_argument("--capture", action="store_true", help="Capture SQTT traces from real hardware (requires sudo + AM_RESET)")
  mode.add_argument("--compare", action="store_true", help="Run emulator and compare against HW captures")
  parser.add_argument("--captures", type=str, default=None, help="Path to HW captures directory (for --compare mode)")
  parser.add_argument("--output", type=str, default=None, help="Output directory for captures (for --capture mode)")
  args = parser.parse_args()

  if args.capture:
    output = Path(args.output) if args.output else CAPTURE_DIR
    do_capture(output)
  elif args.compare:
    do_compare(capture_dir=args.captures)
