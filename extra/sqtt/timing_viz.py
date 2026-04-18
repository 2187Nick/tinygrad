#!/usr/bin/env python3
"""SQTT Timing Visualizer — Interactive HTML comparison of HW vs Emulator traces.

Modes:
  1. Live comparison (runs emulator):
     DEV=AMD MOCKGPU=1 PYTHON_REMU=1 PROFILE=1 SQTT=1 PYTHONPATH=. \
       COMGR_PATH=/opt/rocm-6.4.1/lib/libamd_comgr.so \
       python extra/sqtt/timing_viz.py

  2. View-only (from previously saved JSON):
     python extra/sqtt/timing_viz.py --view timing_data.json

  3. Export JSON only (no server):
     DEV=AMD MOCKGPU=1 PYTHON_REMU=1 PROFILE=1 SQTT=1 PYTHONPATH=. \
       COMGR_PATH=/opt/rocm-6.4.1/lib/libamd_comgr.so \
       python extra/sqtt/timing_viz.py --export timing_data.json
"""
import os, sys, json, pickle, pathlib, argparse

CAPTURE_DIR = pathlib.Path("extra/sqtt/captures/rigorous")
VIZ_HTML = pathlib.Path(__file__).parent / "timing_viz.html"

def generate_comparison_data() -> dict:
  """Run emulator and compare against HW captures. Returns structured JSON data."""
  os.environ.setdefault("DEV", "AMD")
  os.environ.setdefault("PROFILE", "1")
  os.environ.setdefault("SQTT", "1")
  os.environ.setdefault("VIZ", "-2")

  from tinygrad import Device, Tensor
  from tinygrad.device import Compiled
  from extra.sqtt.rigorous_hw_test import KERNELS, _clear, extract_traces

  # Warmup
  (Tensor([1.]) + Tensor([1.])).realize()
  Device[Device.DEFAULT].synchronize()

  kernels_data = {}

  for name, (run_fn, _) in KERNELS.items():
    hw_pkl = CAPTURE_DIR / f"{name}.pkl"
    if not hw_pkl.exists(): continue

    with open(hw_pkl, "rb") as f:
      hw_traces = pickle.load(f)

    # Run emulator
    _clear()
    run_fn()
    Device[Device.DEFAULT].synchronize()
    Device[Device.DEFAULT]._at_profile_finalize()
    emu_traces, _ = extract_traces()

    hw_waves = sorted(hw_traces.keys())
    emu_waves = sorted(emu_traces.keys())
    if len(hw_waves) > len(emu_waves): continue
    n_common = min(len(hw_waves), len(emu_waves))

    waves_data = []
    for wi in range(n_common):
      hw_wid, emu_wid = hw_waves[wi], emu_waves[wi]
      hw = hw_traces[hw_wid]
      emu = emu_traces[emu_wid]
      min_len = min(len(hw), len(emu))

      pc_match = all(hw[j][0] == emu[j][0] for j in range(min_len))
      if not pc_match:
        waves_data.append({"wave_idx": wi, "pc_match": False, "instructions": [], "stats": {}})
        continue

      instructions = []
      exact = within2 = compared = 0
      for j in range(min_len):
        hd = 0 if j == 0 else hw[j][1] - hw[j-1][1]
        ed = 0 if j == 0 else emu[j][1] - emu[j-1][1]
        skip = hd > 50 or ed > 50
        diff = ed - hd
        if not skip:
          compared += 1
          if hd == ed: exact += 1; within2 += 1; status = "exact"
          elif abs(diff) <= 2: within2 += 1; status = "close"
          else: status = "mismatch"
        else:
          status = "dram_wait"

        cat = hw[j][2] if len(hw[j]) > 2 else ""
        inst = hw[j][3] if len(hw[j]) > 3 else ""
        instructions.append({
          "idx": j, "pc": f"0x{hw[j][0]:x}", "hw_time": hw[j][1], "emu_time": emu[j][1],
          "hw_delta": hd, "emu_delta": ed, "diff": diff, "status": status,
          "category": cat, "instruction": inst[:80]
        })

      waves_data.append({
        "wave_idx": wi, "pc_match": True, "instructions": instructions,
        "stats": {"exact": exact, "within2": within2, "compared": compared}
      })

    if waves_data:
      total_exact = sum(w["stats"].get("exact", 0) for w in waves_data)
      total_compared = sum(w["stats"].get("compared", 0) for w in waves_data)
      has_any_match = any(w["pc_match"] for w in waves_data)
      kernels_data[name] = {
        "waves": waves_data, "pc_comparable": has_any_match and total_compared > 0,
        "stats": {"exact": total_exact, "compared": total_compared,
                  "pct": round(100 * total_exact / total_compared, 1) if total_compared else 0}
      }

  total_exact = sum(k["stats"]["exact"] for k in kernels_data.values())
  total_compared = sum(k["stats"]["compared"] for k in kernels_data.values())
  return {
    "kernels": kernels_data,
    "summary": {"exact": total_exact, "compared": total_compared,
                "pct": round(100 * total_exact / total_compared, 1) if total_compared else 0}
  }

def generate_html(data: dict, output: str = "extra/sqtt/timing_viz_output.html"):
  """Generate a self-contained HTML file with embedded data."""
  html_template = VIZ_HTML.read_text()
  html = html_template.replace("__DATA_PLACEHOLDER__", json.dumps(data))
  out = pathlib.Path(output)
  out.write_text(html)
  print(f"Generated: {out} ({len(html)//1024}KB)")
  print(f"Open in browser: file:///{out.resolve()}")
  # In WSL, try to open with Windows browser
  wsl_path = f"\\\\wsl$\\Ubuntu{out.resolve()}"
  try:
    import subprocess
    subprocess.Popen(["cmd.exe", "/c", "start", "", f"file:///{out.resolve()}"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  except Exception: pass
  return str(out.resolve())

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="SQTT Timing Visualizer")
  parser.add_argument("--view", type=str, help="View previously exported JSON file")
  parser.add_argument("--export", type=str, help="Also export raw JSON data to file")
  args = parser.parse_args()

  if args.view:
    with open(args.view) as f: data = json.load(f)
  else:
    print("Generating comparison data (running emulator)...")
    data = generate_comparison_data()
    print(f"Generated data for {len(data['kernels'])} kernels: {data['summary']['exact']}/{data['summary']['compared']} exact ({data['summary']['pct']}%)")

  if args.export:
    with open(args.export, "w") as f: json.dump(data, f, indent=2)
    print(f"Exported JSON to {args.export}")
  
  generate_html(data)
