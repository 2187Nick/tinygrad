#!/usr/bin/env python3
"""SQTT Debug Trace Analyzer — dumps emulator internal state for mismatching instructions.

Usage:
  DEV=AMD MOCKGPU=1 PYTHON_REMU=1 PROFILE=1 SQTT=1 SQTT_DEBUG=1 PYTHONPATH=. \
    COMGR_PATH=/opt/rocm-6.4.1/lib/libamd_comgr.so \
    .venv/bin/python extra/sqtt/debug_trace.py [kernel_name]
"""
import os, sys, pickle, pathlib
os.environ.setdefault("DEV", "AMD")
os.environ.setdefault("PROFILE", "1")
os.environ.setdefault("SQTT", "1")
os.environ.setdefault("VIZ", "-2")
os.environ.setdefault("SQTT_DEBUG", "1")

from tinygrad import Device, Tensor
from tinygrad.device import Compiled
from extra.sqtt.rigorous_hw_test import KERNELS, _clear, extract_traces

CAPTURE_DIR = pathlib.Path("extra/sqtt/captures/rigorous")

def analyze_kernel(name: str):
  hw_pkl = CAPTURE_DIR / f"{name}.pkl"
  if not hw_pkl.exists():
    print(f"No HW capture for {name}")
    return

  with open(hw_pkl, "rb") as f:
    hw_traces = pickle.load(f)

  run_fn = KERNELS[name][0]
  _clear()
  run_fn()
  Device[Device.DEFAULT].synchronize()
  Device[Device.DEFAULT]._at_profile_finalize()
  emu_traces, _ = extract_traces()

  # Get debug log
  from test.mockgpu.amd.emu import _sqtt_debug_log
  debug_log = list(_sqtt_debug_log)

  hw_waves = sorted(hw_traces.keys())
  emu_waves = sorted(emu_traces.keys())
  n_common = min(len(hw_waves), len(emu_waves))

  print(f"\n{'='*80}")
  print(f"  DEBUG TRACE: {name}")
  print(f"  HW waves: {len(hw_waves)}, EMU waves: {len(emu_waves)}, debug entries: {len(debug_log)}")
  print(f"{'='*80}")

  for wi in range(n_common):
    hw = hw_traces[hw_waves[wi]]
    emu = emu_traces[emu_waves[wi]]
    min_len = min(len(hw), len(emu))

    pc_match = all(hw[j][0] == emu[j][0] for j in range(min_len))
    if not pc_match:
      print(f"\n  Wave {wi}: PC MISMATCH — skipping")
      continue

    print(f"\n  Wave {wi}:")
    print(f"  {'Idx':>3} {'PC':>8} {'HW_Δ':>5} {'EMU_Δ':>5} {'Diff':>5} {'Cat':>10} {'Instruction':<50} {'Debug Info'}")
    print(f"  {'─'*3} {'─'*8} {'─'*5} {'─'*5} {'─'*5} {'─'*10} {'─'*50} {'─'*40}")

    for j in range(min_len):
      hd = 0 if j == 0 else hw[j][1] - hw[j-1][1]
      ed = 0 if j == 0 else emu[j][1] - emu[j-1][1]
      if hd > 50 or ed > 50: continue
      diff = ed - hd
      cat = hw[j][2] if len(hw[j]) > 2 else ""
      inst = hw[j][3][:50] if len(hw[j]) > 3 else ""

      # Find matching debug entry
      dbg_info = ""
      for d in debug_log:
        if d["wave"] == wi and d["pc_idx"] == j:
          parts = []
          if "vmem_wr_deadline" in d:
            parts.append(f"deadline={d['vmem_wr_deadline']} set@{d['vmem_wr_set_time']} bypass={d['bypass_active']}")
            for k, v in d.items():
              if k.startswith("w") and k.endswith("_done"): parts.append(f"{k}={v}")
              if k.startswith("w") and k.endswith("_ready"): parts.append(f"{k}={v}")
              if k.startswith("w") and k.endswith("_cat"): parts.append(f"{k}={v}")
              if k.startswith("w") and k.endswith("_vmem_drain"): parts.append(f"{k}={v}")
          elif "vmem_drain_deadline" in d and cat != "vmem_wr":
            parts.append(f"drain_dl={d['vmem_drain_deadline']}")
          parts.append(f"issue@{d['issue_cycle']} ready@{d['ready']}")
          dbg_info = " | ".join(parts)
          break

      marker = "  " if diff == 0 else "← " if abs(diff) <= 2 else "◆ "
      if diff != 0:
        print(f"  {j:3d} {hw[j][0]:>#8x} {hd:5d} {ed:5d} {diff:+5d} {cat:>10} {inst:<50} {marker}{dbg_info}")

  # Summary of all vmem_wr debug entries
  vmem_entries = [d for d in debug_log if d["cat"] == "vmem_wr"]
  if vmem_entries:
    print(f"\n  VMEM_WR Bypass Analysis:")
    for d in vmem_entries:
      w = d["wave"]
      bypass = d.get("bypass_active", False)
      deadline = d.get("vmem_wr_deadline", 0)
      issue = d["issue_cycle"]
      delta = issue - (d["ready"])
      other_info = []
      for k, v in d.items():
        if k.startswith("w") and not k.startswith("wave"):
          other_info.append(f"{k}={v}")
      print(f"    Wave {w} idx={d['pc_idx']}: bypass={bypass} deadline={deadline} issue@{issue} "
            f"delta_from_ready={delta} {' '.join(other_info)}")

if __name__ == "__main__":
  # Warmup
  (Tensor([1.]) + Tensor([1.])).realize()
  Device[Device.DEFAULT].synchronize()

  target = sys.argv[1] if len(sys.argv) > 1 else None
  kernels = [target] if target else ["data_deps", "probe_branch_cost", "probe_cmp_chain", "probe_sgpr_cmps", "probe_vmem_chain"]
  for name in kernels:
    if name in KERNELS:
      analyze_kernel(name)
