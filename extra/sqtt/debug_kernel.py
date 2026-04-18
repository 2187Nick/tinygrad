#!/usr/bin/env python3
"""Show per-instruction HW vs EMU timings for a specific kernel, with annotations.

Usage: DEV=AMD MOCKGPU=1 PYTHON_REMU=1 PROFILE=1 SQTT=1 PYTHONPATH=. \
       .venv/bin/python extra/sqtt/debug_kernel.py <kernel_name> [wave_idx]
"""
import os, sys, pickle, pathlib
os.environ.setdefault("DEV", "AMD")
os.environ.setdefault("PROFILE", "1")
os.environ.setdefault("SQTT", "1")
os.environ.setdefault("VIZ", "-2")

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from rigorous_hw_test import KERNELS, CAPTURE_DIR, extract_traces, _clear
from tinygrad import Device, Tensor

name = sys.argv[1] if len(sys.argv) > 1 else "exp_chain"
wave_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0

assert os.environ.get("MOCKGPU") == "1", "run with MOCKGPU=1"
# Warmup
(Tensor([1.]) + Tensor([1.])).realize()
Device[Device.DEFAULT].synchronize()

hw = pickle.load(open(CAPTURE_DIR / f"{name}.pkl", "rb"))
run_fn, _ = KERNELS[name]
_clear()
run_fn()
Device[Device.DEFAULT].synchronize()
Device[Device.DEFAULT]._at_profile_finalize()
emu, _ = extract_traces()

hw_waves = sorted(hw.keys())
emu_waves = sorted(emu.keys())
hw_wid = hw_waves[wave_idx]
emu_wid = emu_waves[min(wave_idx, len(emu_waves)-1)]

hw_trace = hw[hw_wid]
emu_trace = emu[emu_wid]

hw_pc0, emu_pc0 = hw_trace[0][0], emu_trace[0][0]

print(f"Kernel: {name}, wave {wave_idx}")
print(f"{'idx':>3} {'PC':>6} {'type':>12} {'HWΔ':>4} {'EMUΔ':>4} {'diff':>5}  {'HW_t':>6} {'EMU_t':>6}  inst")
print(f"{'-'*3} {'-'*6} {'-'*12} {'-'*4} {'-'*4} {'-'*5}  {'-'*6} {'-'*6}  {'-'*40}")

minlen = min(len(hw_trace), len(emu_trace))
for j in range(minlen):
  hpc, ht, htyp, hinst = hw_trace[j][:4]
  epc, et, etyp, einst = emu_trace[j][:4]
  hd = 0 if j == 0 else ht - hw_trace[j-1][1]
  ed = 0 if j == 0 else et - emu_trace[j-1][1]
  diff = ed - hd
  marker = " " if hd == ed else ("*" if abs(diff) > 2 else "~")
  skip = "S" if (hd > 50 or ed > 50) else " "
  print(f"{j:>3} {hpc-hw_pc0:>06x} {htyp:>12} {hd:>4} {ed:>4} {diff:>+4}{marker}{skip} {ht:>6} {et:>6}  {hinst[:60]}")
