#!/usr/bin/env python3
"""Rigorous HW validation: capture SQTT from real GPU, then compare emulator output.

Step 1 (on real GPU): Capture traces
  sudo DEV=AMD AM_RESET=1 VIZ=-2 PROFILE=1 SQTT=1 DEBUG=0 PYTHONPATH=. .venv/bin/python extra/sqtt/rigorous_hw_test.py --capture

Step 2 (emulator): Compare
  DEV=AMD MOCKGPU=1 PYTHON_REMU=1 PROFILE=1 SQTT=1 PYTHONPATH=. .venv/bin/python extra/sqtt/rigorous_hw_test.py --compare
"""
import os, sys, pickle, pathlib, functools, argparse
os.environ.setdefault("DEV", "AMD")
os.environ.setdefault("PROFILE", "1")
os.environ.setdefault("SQTT", "1")
os.environ.setdefault("VIZ", "-2")

from tinygrad import Device, Tensor, dtypes
from tinygrad.device import Compiled, ProfileEvent, ProfileProgramEvent, ProfileDeviceEvent
from tinygrad.renderer.amd.sqtt import map_insts, WAVESTART, WAVEEND, INST, VALUINST, IMMEDIATE

TARGET = "gfx1100"
CAPTURE_DIR = pathlib.Path("extra/sqtt/captures/rigorous")

def _clear():
  Compiled.profile_events[:] = [e for e in Compiled.profile_events if isinstance(e, (ProfileProgramEvent, ProfileDeviceEvent))]

def extract_traces():
  """Extract instruction-level traces from current profile events. Returns dict[wave_id] -> [(pc, time, type_name)]."""
  sqtt_events = [e for e in Compiled.profile_events if type(e).__name__ == 'ProfileSQTTEvent']
  program_events = {e.tag: e for e in Compiled.profile_events if type(e).__name__ == 'ProfileProgramEvent'}
  all_traces = {}
  lib = None
  for ev in sqtt_events:
    if not ev.itrace: continue
    if ev.kern not in program_events: continue
    kev = program_events[ev.kern]
    lib = kev.lib
    for pkt, info in map_insts(ev.blob, kev.lib, TARGET):
      if info is None: continue
      w = info.wave
      if w not in all_traces: all_traces[w] = []
      all_traces[w].append((info.pc, pkt._time, type(pkt).__name__, str(info.inst)[:80]))
  return all_traces, lib

def run_capture(name, run_fn, max_attempts=30):
  """Run kernel on real GPU, retry until INST packets captured."""
  for attempt in range(max_attempts):
    try:
      _clear()
      run_fn()
      Device[Device.DEFAULT].synchronize()
      Device[Device.DEFAULT]._at_profile_finalize()
      traces, lib = extract_traces()
      for wid in traces:
        inst_count = sum(1 for _, _, t, _ in traces[wid] if t in ("INST", "VALUINST", "IMMEDIATE"))
        if inst_count >= 3:
          print(f"  {name}[{attempt}]: {len(traces)} wave(s), wave {wid}={inst_count} insts")
          return traces
      if attempt % 10 == 0 and attempt > 0:
        print(f"  {name}[{attempt}]: retrying...")
    except Exception as e:
      if attempt == 0: print(f"  {name}: error on attempt {attempt}: {e}")
      if attempt >= 3:
        print(f"  {name}: persistent error after {attempt+1} attempts, skipping")
        return None
  print(f"  {name}: FAILED after {max_attempts} attempts")
  return None

# ─── Kernel definitions ───────────────────────────────────────────────────────

def _run_lds_sync():
  from test.amd.test_custom_kernel import custom_lds_sync
  from test.amd.helpers import TARGET_TO_ARCH
  arch = TARGET_TO_ARCH[Device["AMD"].arch]
  a = Tensor.empty(64, dtype=dtypes.int32).contiguous().realize()
  Device[Device.DEFAULT].synchronize()
  _clear()
  return Tensor.custom_kernel(a, fxn=functools.partial(custom_lds_sync, arch=arch))[0].realize()

def _run_data_deps():
  from test.amd.test_custom_kernel import custom_data_deps
  from test.amd.helpers import TARGET_TO_ARCH
  arch = TARGET_TO_ARCH[Device["AMD"].arch]
  a = Tensor.empty(64, dtype=dtypes.float32).contiguous().realize()
  Device[Device.DEFAULT].synchronize()
  _clear()
  return Tensor.custom_kernel(a, fxn=functools.partial(custom_data_deps, arch=arch))[0].realize()

def _run_reduce256():
  Tensor.manual_seed(42)
  return Tensor.randn(256).sum().realize()

def _run_elementwise():
  a = Tensor([1., 2, 3, 4])
  b = Tensor([5., 6, 7, 8])
  c = Tensor([0.5, 0.5, 0.5, 0.5])
  return ((a + b) * c - a).realize()

def _run_plus():
  return (Tensor([1., 2, 3, 4]) + Tensor([5., 6, 7, 8])).realize()

def _run_wave_sync():
  """Wave sync exercises s_sleep + s_barrier. Use small input to limit wave count."""
  from test.amd.test_custom_kernel import custom_wave_sync
  from test.amd.helpers import TARGET_TO_ARCH
  arch = TARGET_TO_ARCH[Device["AMD"].arch]
  a = Tensor.empty(128, dtype=dtypes.int32).contiguous().realize()
  Device[Device.DEFAULT].synchronize()
  _clear()
  return Tensor.custom_kernel(a, fxn=functools.partial(custom_wave_sync, arch=arch))[0].realize()

def _run_softmax():
  Tensor.manual_seed(42)
  x = Tensor.randn(64).realize()
  Device[Device.DEFAULT].synchronize()
  _clear()
  return x.softmax().realize()

def _run_layernorm():
  Tensor.manual_seed(42)
  x = Tensor.randn(64).realize()
  Device[Device.DEFAULT].synchronize()
  _clear()
  return x.layernorm().realize()

def _run_exp_chain():
  """Multiple transcendental ops back-to-back (4-cycle VALUT pipeline)."""
  x = Tensor([1., 2., 3., 4.]).realize()
  Device[Device.DEFAULT].synchronize()
  _clear()
  return x.exp().log().sqrt().realize()

def _run_matmul_medium():
  """32x32 matmul — more compute-bound than 4x4."""
  Tensor.manual_seed(42)
  a = Tensor.randn(32, 32).realize()
  b = Tensor.randn(32, 32).realize()
  Device[Device.DEFAULT].synchronize()
  _clear()
  return (a @ b).realize()

def _run_reduce_large():
  """4096-element reduction — deep LDS tree."""
  Tensor.manual_seed(42)
  return Tensor.randn(4096).sum().realize()

def _run_where():
  """Conditional select: exercises VOPC (v_cmp) + v_cndmask."""
  a = Tensor([1., 2., 3., 4.]).realize()
  b = Tensor([5., 6., 7., 8.]).realize()
  Device[Device.DEFAULT].synchronize()
  _clear()
  return (a > 2).where(a, b).realize()

def _run_cast():
  """Type cast: float32 → int32 → float32 exercises VALU format conversion."""
  x = Tensor([1.5, 2.7, 3.1, 4.9]).realize()
  Device[Device.DEFAULT].synchronize()
  _clear()
  return x.cast(dtypes.int32).cast(dtypes.float32).realize()

def _run_probe_sgpr_cmps():
  from test.amd.test_custom_kernel import custom_probe_sgpr_cmps
  from test.amd.helpers import TARGET_TO_ARCH
  arch = TARGET_TO_ARCH[Device["AMD"].arch]
  a = Tensor.empty(64, dtype=dtypes.float32).contiguous().realize()
  Device[Device.DEFAULT].synchronize()
  _clear()
  return Tensor.custom_kernel(a, fxn=functools.partial(custom_probe_sgpr_cmps, arch=arch))[0].realize()

def _run_probe_cmp_chain():
  from test.amd.test_custom_kernel import custom_probe_cmp_chain
  from test.amd.helpers import TARGET_TO_ARCH
  arch = TARGET_TO_ARCH[Device["AMD"].arch]
  a = Tensor.empty(64, dtype=dtypes.float32).contiguous().realize()
  Device[Device.DEFAULT].synchronize()
  _clear()
  return Tensor.custom_kernel(a, fxn=functools.partial(custom_probe_cmp_chain, arch=arch))[0].realize()

def _run_probe_branch_cost():
  from test.amd.test_custom_kernel import custom_probe_branch_cost
  from test.amd.helpers import TARGET_TO_ARCH
  arch = TARGET_TO_ARCH[Device["AMD"].arch]
  a = Tensor.empty(64, dtype=dtypes.float32).contiguous().realize()
  Device[Device.DEFAULT].synchronize()
  _clear()
  return Tensor.custom_kernel(a, fxn=functools.partial(custom_probe_branch_cost, arch=arch))[0].realize()

def _run_probe_vmem_chain():
  from test.amd.test_custom_kernel import custom_probe_vmem_chain
  from test.amd.helpers import TARGET_TO_ARCH
  arch = TARGET_TO_ARCH[Device["AMD"].arch]
  a = Tensor.empty(64, dtype=dtypes.float32).contiguous().realize()
  Device[Device.DEFAULT].synchronize()
  _clear()
  return Tensor.custom_kernel(a, fxn=functools.partial(custom_probe_vmem_chain, arch=arch))[0].realize()

KERNELS = {
  # Non-DRAM dominated (best for emulator validation)
  "lds_sync":       (_run_lds_sync, 30),
  "data_deps":      (_run_data_deps, 10),
  "wave_sync":      (_run_wave_sync, 30),
  # Compute / LDS heavy
  "softmax":        (_run_softmax, 15),
  "layernorm":      (_run_layernorm, 15),
  "exp_chain":      (_run_exp_chain, 20),
  "reduce256":      (_run_reduce256, 10),
  "reduce_large":   (_run_reduce_large, 10),
  "matmul_medium":  (_run_matmul_medium, 10),
  # DRAM dominated (test VMEM forwarding)
  "elementwise":    (_run_elementwise, 10),
  "plus":           (_run_plus, 30),
  "where":          (_run_where, 20),
  "cast":           (_run_cast, 20),
  # Mechanism probes (for SGPR write-port mystery + branch cost + VMEM serialization)
  "probe_sgpr_cmps":   (_run_probe_sgpr_cmps, 30),
  "probe_cmp_chain":   (_run_probe_cmp_chain, 30),
  "probe_branch_cost": (_run_probe_branch_cost, 30),
  "probe_vmem_chain":  (_run_probe_vmem_chain, 30),
}

# ─── Capture mode ─────────────────────────────────────────────────────────────

def do_capture():
  print("=== HW Capture Mode ===")
  try:
    mode = open("/sys/class/drm/card1/device/power_dpm_force_performance_level").read().strip()
    print(f"Power mode: {mode}")
    if mode != "profile_standard":
      print("WARNING: Not in profile_standard mode!")
  except: pass

  # Warmup
  print("Warmup...")
  (Tensor([1.]) + Tensor([1.])).realize()
  Device[Device.DEFAULT].synchronize()

  CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
  captured = {}
  for name, (run_fn, max_att) in KERNELS.items():
    print(f"\n--- {name} ---")
    traces = run_capture(name, run_fn, max_att)
    if traces:
      captured[name] = traces
      out = CAPTURE_DIR / f"{name}.pkl"
      with open(out, "wb") as f:
        pickle.dump(traces, f)
      print(f"  Saved → {out}")

  print(f"\n{'='*60}")
  print(f"Captured {len(captured)}/{len(KERNELS)} kernels")
  for name, traces in captured.items():
    for wid in sorted(traces.keys()):
      n = len(traces[wid])
      inst_n = sum(1 for _, _, t, _ in traces[wid] if t in ("INST", "VALUINST", "IMMEDIATE"))
      print(f"  {name} wave {wid}: {n} total, {inst_n} inst-level")
  print("Done!")

# ─── Compare mode ─────────────────────────────────────────────────────────────

def do_compare():
  print("=== Emulator Compare Mode ===")
  assert os.environ.get("MOCKGPU") == "1", "Must run with MOCKGPU=1"

  # Warmup
  (Tensor([1.]) + Tensor([1.])).realize()
  Device[Device.DEFAULT].synchronize()

  total_exact = 0
  total_within2 = 0
  total_compared = 0
  kernel_results = {}

  for name, (run_fn, _) in KERNELS.items():
    hw_pkl = CAPTURE_DIR / f"{name}.pkl"
    if not hw_pkl.exists():
      print(f"\n--- {name}: SKIP (no HW capture) ---")
      continue

    with open(hw_pkl, "rb") as f:
      hw_traces = pickle.load(f)

    # Run through emulator
    _clear()
    run_fn()
    Device[Device.DEFAULT].synchronize()
    Device[Device.DEFAULT]._at_profile_finalize()
    emu_traces, _ = extract_traces()

    print(f"\n{'='*60}")
    print(f"  {name}: HW ({len(hw_traces)} waves) vs EMU ({len(emu_traces)} waves)")
    print(f"{'='*60}")

    hw_waves = sorted(hw_traces.keys())
    emu_waves = sorted(emu_traces.keys())
    # Skip kernels where HW has MORE waves than EMU — indicates genuinely different workgroup size (e.g. matmul: HW=6, EMU=1)
    # When EMU >= HW, we compare the overlapping waves
    if len(hw_waves) > len(emu_waves):
      print(f"  SKIP (wave count mismatch: HW={len(hw_waves)} waves, EMU={len(emu_waves)} waves)")
      continue
    n_common = min(len(hw_waves), len(emu_waves))

    k_exact = 0
    k_within2 = 0
    k_compared = 0

    for i in range(n_common):
      hw_wid, emu_wid = hw_waves[i], emu_waves[i]
      hw = hw_traces[hw_wid]
      emu = emu_traces[emu_wid]
      min_len = min(len(hw), len(emu))

      # Normalize PC to offset from first instruction (HW and EMU load code at different VRAM addresses).
      hw_pc0 = hw[0][0] if hw else 0
      emu_pc0 = emu[0][0] if emu else 0
      hw_off = [h[0] - hw_pc0 for h in hw]
      emu_off = [e[0] - emu_pc0 for e in emu]
      pc_match = all(hw_off[j] == emu_off[j] for j in range(min_len))
      if not pc_match:
        for j in range(min_len):
          if hw_off[j] != emu_off[j]:
            print(f"  Wave {i}: PC-offset mismatch at idx {j}: HW+0x{hw_off[j]:x} EMU+0x{emu_off[j]:x} (HW=0x{hw[j][0]:x}, EMU=0x{emu[j][0]:x})")
            print(f"    HW offs:  {[hex(hw_off[k]) for k in range(max(0,j-2), min(j+3, len(hw)))]}")
            print(f"    EMU offs: {[hex(emu_off[k]) for k in range(max(0,j-2), min(j+3, len(emu)))]}")
            print(f"    HW insts: {[hw[k][3][:30] for k in range(max(0,j-2), min(j+3, len(hw)))]}")
            print(f"    EMU insts:{[emu[k][3][:30] for k in range(max(0,j-2), min(j+3, len(emu)))]}")
            break
        print(f"  Wave {i}: SKIP (different instruction stream)")
        continue

      # Compare deltas — skip DRAM waits (>50 cycles)
      exact = within2 = compared = 0
      mismatches = []
      for j in range(min_len):
        hd = 0 if j == 0 else hw[j][1] - hw[j-1][1]
        ed = 0 if j == 0 else emu[j][1] - emu[j-1][1]
        if hd > 50 or ed > 50: continue
        compared += 1
        if hd == ed:
          exact += 1; within2 += 1
        elif abs(hd - ed) <= 2:
          within2 += 1
          mismatches.append((j, hd, ed, hw[j][2], hw[j][3] if len(hw[j]) > 3 else ""))
        else:
          mismatches.append((j, hd, ed, hw[j][2], hw[j][3] if len(hw[j]) > 3 else ""))

      k_exact += exact
      k_within2 += within2
      k_compared += compared

      if compared > 0:
        pct = 100*exact/compared
        pct2 = 100*within2/compared
        print(f"  Wave {i}: {exact}/{compared} exact ({pct:.1f}%), {within2}/{compared} ±2 ({pct2:.1f}%)")
        if mismatches:
          for j, hd, ed, typ, inst in mismatches[:25]:
            print(f"    [{j:2d}] HW={hd:3d} EMU={ed:3d} diff={ed-hd:+d} {typ} {inst[:60]}")
          if len(mismatches) > 25:
            print(f"    ... {len(mismatches)-25} more mismatches")

    total_exact += k_exact
    total_within2 += k_within2
    total_compared += k_compared
    if k_compared > 0:
      kernel_results[name] = (k_exact, k_within2, k_compared)

  print(f"\n{'='*60}")
  print(f"  SUMMARY")
  print(f"{'='*60}")
  for name, (ex, w2, comp) in kernel_results.items():
    print(f"  {name:20s}: {ex:3d}/{comp:3d} exact ({100*ex/comp:5.1f}%), {w2:3d}/{comp:3d} ±2 ({100*w2/comp:5.1f}%)")
  if total_compared > 0:
    print(f"  {'TOTAL':20s}: {total_exact:3d}/{total_compared:3d} exact ({100*total_exact/total_compared:5.1f}%), "
          f"{total_within2:3d}/{total_compared:3d} ±2 ({100*total_within2/total_compared:5.1f}%)")
  print("Done!")

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--capture", action="store_true", help="Capture from real GPU")
  parser.add_argument("--compare", action="store_true", help="Compare emulator vs captured")
  args = parser.parse_args()

  if args.capture:
    do_capture()
  elif args.compare:
    do_compare()
  else:
    print("Usage: --capture (on real GPU) or --compare (with MOCKGPU=1)")
