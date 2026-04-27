#!/usr/bin/env python3
"""BEAM_EMU vs BEAM_HW selection-quality sweep across multiple workloads.

Per 94_percent.md §4, this measures the load-bearing claim of the bounty:
"BEAM_EMU picks kernels that run as fast as BEAM_HW's pick on real silicon."

Compile-time speedup is NOT measured here — Python `_simulate_sq_timing` is
~93× slower per candidate than a real GPU dispatch (validated 2026-04-27 at
128x128 matmul: HW BEAM=1.69s, EMU BEAM=156.58s). The doc's projected
speedup requires a compiled cycle counter — separate optimization track.

Each workload runs four steps; per-workload outputs live in
extra/sqtt/.beam_emu_compare/<workload>/{hw,emu,verify}.json (override via BEAM_EMU_OUT_DIR).
"""
import os, sys, json, time, pathlib, argparse
from dataclasses import replace

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

# Use a user-owned path so sudo and non-sudo steps can both write here. The wrapper script
# creates this dir + chmods it 777 before any sudo step runs, avoiding the /tmp ownership trap.
OUT_DIR = pathlib.Path(os.environ.get("BEAM_EMU_OUT_DIR", str(ROOT / "extra" / "sqtt" / ".beam_emu_compare")))
OUT_DIR.mkdir(parents=True, exist_ok=True)
try: os.chmod(OUT_DIR, 0o777)
except PermissionError: pass

# Workloads — picked to cover different op characteristics within the time budget the
# Python emulator can handle. Avoid >128 dim on matmul (emu is O(insts) and gets too slow).
def _matmul(n):
  import numpy as np
  from tinygrad import Tensor
  np.random.seed(0)
  a = Tensor(np.random.rand(n, n).astype(np.float32)).realize()
  b = Tensor(np.random.rand(n, n).astype(np.float32)).realize()
  return lambda: (a @ b).realize()

def _softmax(n):
  import numpy as np
  from tinygrad import Tensor
  np.random.seed(0)
  x = Tensor(np.random.rand(n, n).astype(np.float32)).realize()
  return lambda: x.softmax(axis=-1).realize()

def _elementwise(n):
  import numpy as np
  from tinygrad import Tensor
  np.random.seed(0)
  a = Tensor(np.random.rand(n).astype(np.float32)).realize()
  b = Tensor(np.random.rand(n).astype(np.float32)).realize()
  c = Tensor(np.random.rand(n).astype(np.float32)).realize()
  return lambda: (a + b * c).realize()

WORKLOADS: dict[str, callable] = {
  "matmul_64":     lambda: _matmul(64),
  "matmul_128":    lambda: _matmul(128),
  "softmax_64":    lambda: _softmax(64),
  "elementwise_4096": lambda: _elementwise(4096),
}


def _wkdir(name):
  d = OUT_DIR / name
  d.mkdir(parents=True, exist_ok=True)
  try: os.chmod(d, 0o777)
  except PermissionError: pass
  return d

def _write_json(path: pathlib.Path, obj):
  path.write_text(json.dumps(obj, indent=2))
  try: os.chmod(path, 0o666)
  except PermissionError: pass

def _time_kernel_n(realize_fn, n=5):
  from tinygrad import Device
  rts = []
  for _ in range(n):
    Device[Device.DEFAULT].synchronize()
    t = time.perf_counter()
    realize_fn()
    Device[Device.DEFAULT].synchronize()
    rts.append(time.perf_counter() - t)
  return rts


def step_hw(beam_n: int, names: list[str]):
  os.environ["DEV"] = "AMD"
  os.environ["BEAM"] = str(beam_n)
  os.environ["IGNORE_BEAM_CACHE"] = "1"
  os.environ.pop("MOCKGPU", None); os.environ.pop("PYTHON_REMU", None); os.environ.pop("BEAM_EMU", None)
  for name in names:
    print(f"[hw] {name} BEAM={beam_n} ...", flush=True)
    factory = WORKLOADS[name]
    realize_fn = factory()
    from tinygrad import Device
    Device[Device.DEFAULT].synchronize()
    t0 = time.perf_counter()
    realize_fn()
    Device[Device.DEFAULT].synchronize()
    beam_wall = time.perf_counter() - t0
    # warm cache so the kernel-time measurements are clean
    os.environ["BEAM"] = "0"
    realize_fn = factory()
    rts = _time_kernel_n(realize_fn, n=5)
    os.environ["BEAM"] = str(beam_n)  # restore for next workload
    out = {"step":"hw","name":name,"beam_n":beam_n,"beam_wall_s":beam_wall,
           "kernel_runtime_s_min":min(rts),"kernel_runtime_s_median":sorted(rts)[len(rts)//2],"kernel_runtimes_s":rts}
    _write_json(_wkdir(name) / "hw.json", out)
    print(f"[hw]   {name}: BEAM wall={beam_wall:.2f}s  kernel min={min(rts)*1e6:.1f}us")


def step_emu(beam_n: int, names: list[str]):
  required = {"DEV":"AMD","MOCKGPU":"1","PYTHON_REMU":"1","PROFILE":"1","SQTT":"1","BEAM_EMU":"1"}
  for k,v in required.items():
    if os.environ.get(k) != v:
      print(f"[emu] ERROR: required env {k}={v} not set (got {os.environ.get(k)!r}). Use the wrapper script.", file=sys.stderr); sys.exit(2)
  os.environ["BEAM"] = str(beam_n)
  os.environ["IGNORE_BEAM_CACHE"] = "1"
  for name in names:
    print(f"[emu] {name} BEAM={beam_n} (this can take 1-3 min)...", flush=True)
    factory = WORKLOADS[name]
    realize_fn = factory()
    from tinygrad import Device
    Device[Device.DEFAULT].synchronize()
    t0 = time.perf_counter()
    realize_fn()
    Device[Device.DEFAULT].synchronize()
    beam_wall = time.perf_counter() - t0
    out = {"step":"emu","name":name,"beam_n":beam_n,"beam_wall_s":beam_wall}
    _write_json(_wkdir(name) / "emu.json", out)
    print(f"[emu]  {name}: BEAM wall={beam_wall:.2f}s")


def step_verify(beam_n: int, names: list[str]):
  os.environ["DEV"] = "AMD"
  os.environ["BEAM"] = "0"  # reuse the cache populated by step_emu
  os.environ.pop("MOCKGPU", None); os.environ.pop("PYTHON_REMU", None); os.environ.pop("BEAM_EMU", None)
  for name in names:
    print(f"[verify] {name} ...", flush=True)
    factory = WORKLOADS[name]
    realize_fn = factory()
    rts = _time_kernel_n(realize_fn, n=5)
    out = {"step":"verify","name":name,"kernel_runtime_s_min":min(rts),
           "kernel_runtime_s_median":sorted(rts)[len(rts)//2],"kernel_runtimes_s":rts}
    _write_json(_wkdir(name) / "verify.json", out)
    print(f"[verify] {name}: emu-chosen kernel on HW min={min(rts)*1e6:.1f}us")


def step_report(names: list[str]):
  print()
  print("=" * 92)
  print(f"  BEAM_EMU vs BEAM_HW report — {len(names)} workloads")
  print("=" * 92)
  hdr = f"  {'workload':<22} {'HW BEAM':>10} {'EMU BEAM':>10} {'HW pick':>12} {'EMU pick':>12} {'ratio':>8} {'verdict':>8}"
  print(hdr); print("  " + "-"*90)
  ratios = []
  for name in names:
    d = _wkdir(name)
    hw = json.loads((d/"hw.json").read_text()) if (d/"hw.json").exists() else None
    emu = json.loads((d/"emu.json").read_text()) if (d/"emu.json").exists() else None
    verify = json.loads((d/"verify.json").read_text()) if (d/"verify.json").exists() else None
    if not (hw and verify):
      print(f"  {name:<22} (incomplete)")
      continue
    hw_us = hw["kernel_runtime_s_min"]*1e6
    em_us = verify["kernel_runtime_s_min"]*1e6
    ratio = em_us / hw_us if hw_us > 0 else float("inf")
    ratios.append(ratio)
    emu_wall = f"{emu['beam_wall_s']:9.2f}s" if emu else "      n/a"
    verdict = "OK" if 0.95 <= ratio <= 1.05 else ("FASTER" if ratio < 0.95 else "SLOWER")
    print(f"  {name:<22} {hw['beam_wall_s']:9.2f}s {emu_wall} {hw_us:11.1f}us {em_us:11.1f}us {ratio:8.3f} {verdict:>8}")
  if ratios:
    avg = sum(ratios)/len(ratios)
    pmax = max(ratios); pmin = min(ratios)
    print("  " + "-"*90)
    print(f"  ratio = EMU-chosen runtime / HW-chosen runtime on real silicon (lower = EMU picked better)")
    print(f"  mean ratio: {avg:.3f}   range: [{pmin:.3f}, {pmax:.3f}]")
  print("=" * 92)


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("step", choices=["hw","emu","verify","report"])
  ap.add_argument("--beam-n", type=int, default=4)
  ap.add_argument("--workloads", default=",".join(WORKLOADS.keys()),
                  help=f"comma-separated names; default = all. Available: {','.join(WORKLOADS.keys())}")
  args = ap.parse_args()
  names = [n.strip() for n in args.workloads.split(",") if n.strip() in WORKLOADS]
  if not names:
    print("error: no valid workloads selected", file=sys.stderr); sys.exit(2)
  if args.step == "hw":     step_hw(args.beam_n, names)
  elif args.step == "emu":  step_emu(args.beam_n, names)
  elif args.step == "verify": step_verify(args.beam_n, names)
  elif args.step == "report": step_report(names)
