#!/usr/bin/env python3
"""Brute-force parameter search for SQ timing emulator constants.

Usage: DEV=AMD MOCKGPU=1 PYTHON_REMU=1 PROFILE=1 SQTT=1 PYTHONPATH=. \
       .venv/bin/python extra/sqtt/tune_harness.py [kernel_names...]

Patches emu.py module-level constants and re-runs kernels to measure exact-match
counts. Searches the parameter space one constant at a time (coordinate-descent)
holding others fixed, so each full sweep is O(parameters × values × kernels).
"""
import os, sys, pickle, pathlib, time, itertools, argparse

os.environ.setdefault("DEV", "AMD")
os.environ.setdefault("MOCKGPU", "1")
os.environ.setdefault("PYTHON_REMU", "1")
os.environ.setdefault("PROFILE", "1")
os.environ.setdefault("SQTT", "1")
os.environ.setdefault("VIZ", "-2")

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from tinygrad import Device, Tensor
from tinygrad.device import Compiled, ProfileProgramEvent, ProfileDeviceEvent

from rigorous_hw_test import KERNELS, CAPTURE_DIR, extract_traces, _clear
import test.mockgpu.amd.emu as emu

# Tunable constants and the search space for each.
# Centered on current values; widths tuned to be realistic given RDNA3 microarch.
PARAMS = {
  "_SGPR_LATENCY":            [2, 3, 4, 5, 6, 7],
  "_CNDMASK_SGPR_LATENCY":    [3, 4, 5, 6, 7, 8],
  "_TRANS_PIPELINE_LATENCY":  [24, 25, 26, 27, 28, 29, 30],
  "_TRANS_PIPELINE_LATENCY_SQRT": [28, 29, 30, 31, 32, 33, 34],
  "_TRANS_PIPE_CYCLES":       [3, 4, 5, 6],
  "_VOPD_PIPE_CYCLES":        [1, 2, 3, 4, 5],
  "_VALU_DS_WR_FORWARD":      [20, 22, 24, 26, 28],
  "_VALU_DS_RD_FORWARD":      [18, 20, 22, 24, 26],
  "_VALU_VMEM_WR_FORWARD":    [18, 19, 20, 21, 22, 23, 24],
  "_VALU_VMEM_WR_BYPASS":     [2, 3, 4, 5, 6, 7, 8],
  "_VALU_VMEM_ADDR_FORWARD":  [24, 25, 26, 27, 28, 29, 30],
  "_VALU_VMEM_RD_FORWARD":    [18, 20, 22, 24, 26],
  "_VMEM_DRAIN_CYCLES":       [12, 13, 14, 15, 16, 17, 18],
  "_VMEM_EXEC_MIN":           [4, 6, 8, 10, 12],
  "_EXEC_WRITE_LATENCY":      [20, 22, 24, 26, 28],
  "_LDS_SERVICE_COST":        [3, 4, 5, 6, 7, 8],
  "_BARRIER_FROM_LAST":       [4, 5, 6, 7, 8],
  "_LDS_RD_LATENCY":          [29, 30, 31, 32, 33],
  "_LDS_WR_LATENCY":          [31, 32, 33, 34, 35],
  "_FIRST_INST_GAP":          [0, 1, 2, 3, 4],
  "_WAVESTART_GAP":           [0, 1, 2, 3],
}

# Warmup so the emulator is ready
def warmup():
  (Tensor([1.]) + Tensor([1.])).realize()
  Device[Device.DEFAULT].synchronize()

def measure(kernel_names):
  """Run each kernel through the emulator and return (total_exact, total_compared, per_kernel).
  Also tracks total_score = 10*exact + within2 count (weighting to prefer both exact and close).
  Returned as per_kernel['_score'] summary tuple."""
  total_exact = 0
  total_compared = 0
  total_score = 0
  per_kernel = {}
  for name in kernel_names:
    if name not in KERNELS:
      continue
    run_fn, _ = KERNELS[name]
    pkl_path = CAPTURE_DIR / f"{name}.pkl"
    if not pkl_path.exists():
      continue
    with open(pkl_path, "rb") as f:
      hw_traces = pickle.load(f)
    _clear()
    try:
      run_fn()
      Device[Device.DEFAULT].synchronize()
      Device[Device.DEFAULT]._at_profile_finalize()
    except Exception as e:
      print(f"  {name}: error {e}")
      continue
    emu_traces, _ = extract_traces()

    hw_waves = sorted(hw_traces.keys())
    emu_waves = sorted(emu_traces.keys())
    if len(hw_waves) > len(emu_waves):
      continue
    n_common = min(len(hw_waves), len(emu_waves))

    k_exact = 0
    k_compared = 0
    k_score = 0
    for i in range(n_common):
      hw = hw_traces[hw_waves[i]]
      em = emu_traces[emu_waves[i]]
      min_len = min(len(hw), len(em))
      hw_pc0 = hw[0][0] if hw else 0
      emu_pc0 = em[0][0] if em else 0
      pc_match = all(hw[j][0] - hw_pc0 == em[j][0] - emu_pc0 for j in range(min_len))
      if not pc_match:
        continue
      for j in range(min_len):
        hd = 0 if j == 0 else hw[j][1] - hw[j-1][1]
        ed = 0 if j == 0 else em[j][1] - em[j-1][1]
        if hd > 50 or ed > 50:
          continue
        k_compared += 1
        if hd == ed:
          k_exact += 1
          k_score += 10   # exact: weight 10
        elif abs(hd - ed) <= 2:
          k_score += 1    # close: weight 1
        # Miss: weight 0 (no credit)
    total_exact += k_exact
    total_compared += k_compared
    total_score += k_score
    per_kernel[name] = (k_exact, k_compared, k_score)
  per_kernel["_score"] = total_score
  return total_exact, total_compared, per_kernel

def set_param(name, val):
  setattr(emu, name, val)

def get_param(name):
  return getattr(emu, name)

def coordinate_descent(kernel_names, max_passes=3):
  """Hold all parameters fixed except one; sweep its values; pick best. Repeat until no change."""
  baseline = {k: get_param(k) for k in PARAMS.keys()}
  print(f"Baseline: {baseline}")

  # Measure baseline
  t0 = time.time()
  best_exact, best_compared, per_k = measure(kernel_names)
  best_score = per_k["_score"]
  print(f"Baseline: {best_exact}/{best_compared} exact ({100*best_exact/max(1,best_compared):.1f}%), score={best_score} in {time.time()-t0:.1f}s")
  for n, v in per_k.items():
    if n == "_score": continue
    ex, cm, sc = v
    print(f"  {n}: {ex}/{cm} score={sc}")

  current = dict(baseline)
  for p in range(max_passes):
    any_change = False
    print(f"\n=== Pass {p+1} ===")
    for pname, values in PARAMS.items():
      orig_val = current[pname]
      best_val = orig_val
      best_at_param = best_score
      best_ex_at_param = best_exact
      for v in values:
        if v == orig_val:
          continue
        set_param(pname, v)
        ex, cm, pk = measure(kernel_names)
        sc = pk["_score"]
        flag = ""
        if sc > best_at_param:
          best_at_param = sc
          best_ex_at_param = ex
          best_val = v
          flag = " <-- best"
        print(f"  {pname} = {v}: {ex}/{cm} score={sc}{flag}")
      set_param(pname, best_val)
      current[pname] = best_val
      if best_val != orig_val:
        best_exact = best_ex_at_param
        best_score = best_at_param
        any_change = True
        print(f"  ===> {pname}: {orig_val} -> {best_val}, exact={best_exact} score={best_score}")
    if not any_change:
      print(f"\nConverged after {p+1} passes. Final: {best_exact}")
      break

  print(f"\n=== Final config ===")
  for k in PARAMS.keys():
    v = get_param(k)
    orig = baseline[k]
    marker = " (CHANGED)" if v != orig else ""
    print(f"  {k} = {v}{marker}")

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("kernels", nargs="*", default=[])
  parser.add_argument("--param", help="Tune single parameter")
  parser.add_argument("--passes", type=int, default=2)
  args = parser.parse_args()

  kernels = args.kernels or [
    "data_deps", "exp_chain", "elementwise", "plus", "cast",
    "probe_sgpr_cmps", "probe_cmp_chain", "probe_branch_cost", "probe_vmem_chain",
  ]
  print(f"Tuning against kernels: {kernels}")

  warmup()

  if args.param:
    global PARAMS
    PARAMS = {args.param: PARAMS[args.param]}

  coordinate_descent(kernels, max_passes=args.passes)

if __name__ == "__main__":
  main()
