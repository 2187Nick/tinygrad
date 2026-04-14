#!/usr/bin/env python3
"""Capture SQTT traces from real GFX1100 hardware for all non-DRAM custom kernels.
Run on a machine with AMD 7900 XTX:  DEV=AMD AM_RESET=1 VIZ=-2 python extra/sqtt/hw_capture.py
"""
import os, subprocess, sys, shlex, shutil, json
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
CAPTURE_DIR = SCRIPT_DIR / "captures" / datetime.now().strftime("%Y%m%d_%H%M%S")

# all kernels to capture — focus on non-DRAM custom kernels
KERNELS = {
  "sync": "test/amd/test_custom_kernel.py TestCustomKernel.test_lds_sync",
  "data_deps": "test/amd/test_custom_kernel.py TestCustomKernel.test_data_deps",
  "wave_sync": "test/amd/test_custom_kernel.py TestCustomKernel.test_wave_sync",
  "plus": "test/test_tiny.py TestTiny.test_plus",
  "gemm": '-c "from tinygrad import Tensor; (Tensor.empty(N:=32, N)@Tensor.empty(N, N)).realize()"',
  "empty": "test/backend/test_custom_kernel.py TestCustomKernel.test_empty",
}
RUNS_PER_KERNEL = 3  # 3 runs each for determinism checking

def get_profile_path():
  from tinygrad.helpers import temp
  return Path(temp("profile.pkl", append_user=True))

if __name__ == "__main__":
  # verify environment
  env_check = {**os.environ, "DEBUG": "0"}
  try:
    arch = subprocess.check_output([sys.executable, "-c", "from tinygrad import Device; print(Device['AMD'].arch)"],
                                   text=True, env=env_check).strip()
  except Exception as e:
    print(f"ERROR: Cannot detect AMD device. Make sure DEV=AMD is set.\n{e}")
    sys.exit(1)

  print(f"Detected GPU arch: {arch}")
  if not arch.startswith("gfx11"):
    print(f"WARNING: Expected gfx1100 (7900 XTX), got {arch}. Results may differ.")

  CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
  profile_path = get_profile_path()
  results = {}

  for name, test_cmd in KERNELS.items():
    print(f"\n{'='*60}")
    print(f"Capturing: {name} ({RUNS_PER_KERNEL} runs)")
    print(f"{'='*60}")
    results[name] = {"runs": [], "status": "ok"}

    for run_idx in range(RUNS_PER_KERNEL):
      env = {**os.environ, "DEV": "AMD", "AM_RESET": "1", "VIZ": "-2", "PYTHONPATH": "."}
      try:
        ret = subprocess.run([sys.executable, *shlex.split(test_cmd)], cwd=str(REPO_ROOT), env=env,
                             capture_output=True, text=True, timeout=120)
        if ret.returncode != 0:
          print(f"  run_{run_idx}: FAILED (exit {ret.returncode})")
          print(f"    stderr: {ret.stderr[-500:]}" if ret.stderr else "")
          results[name]["status"] = "failed"
          results[name]["runs"].append({"run": run_idx, "status": "failed", "error": ret.stderr[-200:]})
          continue
      except subprocess.TimeoutExpired:
        print(f"  run_{run_idx}: TIMEOUT")
        results[name]["status"] = "timeout"
        results[name]["runs"].append({"run": run_idx, "status": "timeout"})
        continue

      dest = CAPTURE_DIR / f"profile_{name}_run_{run_idx}.pkl"
      if profile_path.exists():
        shutil.move(str(profile_path), str(dest))
        size_kb = dest.stat().st_size / 1024
        print(f"  run_{run_idx}: saved {dest.name} ({size_kb:.1f} KB)")
        results[name]["runs"].append({"run": run_idx, "status": "ok", "file": str(dest), "size_kb": round(size_kb, 1)})
      else:
        print(f"  run_{run_idx}: no profile.pkl generated!")
        results[name]["status"] = "no_pkl"
        results[name]["runs"].append({"run": run_idx, "status": "no_pkl"})

  # save manifest
  manifest = {"arch": arch, "capture_dir": str(CAPTURE_DIR), "kernels": results,
              "timestamp": datetime.now().isoformat(), "runs_per_kernel": RUNS_PER_KERNEL}
  manifest_path = CAPTURE_DIR / "manifest.json"
  with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

  print(f"\n{'='*60}")
  print(f"CAPTURE COMPLETE — saved to {CAPTURE_DIR}")
  print(f"{'='*60}")
  for name, info in results.items():
    ok = sum(1 for r in info["runs"] if r["status"] == "ok")
    print(f"  {name:15s}: {ok}/{RUNS_PER_KERNEL} captures successful")
  print(f"\nManifest: {manifest_path}")
  print(f"\nNext step: python extra/sqtt/hw_validate.py {CAPTURE_DIR}")
