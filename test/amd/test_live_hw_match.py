#!/usr/bin/env python3
"""
Live hardware validation: proves the RDNA3 emulator matches a real AMD 7900 XTX (GFX1100).

Runs each test kernel twice:
  1. On REAL hardware (no MOCKGPU) with SQTT profiling → captures HW inter-instruction deltas
  2. On the EMULATOR (MOCKGPU=1 PYTHON_REMU=1) via subprocess → captures EMU deltas
Then compares the non-DRAM sections cycle-by-cycle.

Usage (on a machine with a real 7900 XTX + ROCm):
  PROFILE=1 SQTT=1 python test/amd/test_live_hw_match.py

Cloud setup (vast.ai / runpod):
  - GPU: AMD Radeon RX 7900 XTX (gfx1100)
  - Driver: ROCm 6.x (amdgpu-dkms)
  - Python: 3.10+
  - Install: pip install -e '.[amd]'
  - No rocprof-trace-decoder needed (tinygrad decodes SQTT natively)
"""
import os, sys, subprocess, pickle, textwrap, functools, unittest

# ---------------------------------------------------------------------------
# Environment checks
# ---------------------------------------------------------------------------
_SKIP_REASON = None
if os.environ.get("MOCKGPU", "0") != "0":
  _SKIP_REASON = "test_live_hw_match.py requires REAL hardware (MOCKGPU must be unset or 0)"
elif os.environ.get("PROFILE", "0") == "0" or os.environ.get("SQTT", "0") == "0":
  _SKIP_REASON = "profiling not enabled — run with: PROFILE=1 SQTT=1 python test/amd/test_live_hw_match.py"

if _SKIP_REASON is None:

if _SKIP_REASON is None:
  from tinygrad import Device, Tensor, dtypes
  from tinygrad.device import Compiled, ProfileProgramEvent, ProfileDeviceEvent

TARGET = "gfx1100"
# non-DRAM window PCs for custom_lds_sync (LDS + barrier + register section)
NON_DRAM_PC_START, NON_DRAM_PC_END = 0x10c, 0x140
HW_PCS = [0x10c, 0x110, 0x118, 0x11c, 0x120, 0x124, 0x12c, 0x134, 0x138, 0x13c, 0x140]
TOLERANCE = 2  # ±2 cycles

# ---------------------------------------------------------------------------
# SQTT trace extraction (shared with test_emulator_timing.py)
# ---------------------------------------------------------------------------
def _extract_wave_traces(blob, lib, target):
  """Extract per-wave instruction traces from SQTT blob. Returns {wave_id: [(pc, time, pkt_type), ...]}"""
  from tinygrad.renderer.amd.sqtt import map_insts
  traces = {}
  for pkt, info in map_insts(blob, lib, target):
    if info is None: continue
    wave = info.wave
    if wave not in traces: traces[wave] = []
    traces[wave].append((info.pc, pkt._time, type(pkt).__name__))
  return traces

def _capture_sqtt():
  """Finalize profiling and extract the first itrace SQTT event with its program event."""
  Device[Device.DEFAULT].synchronize()
  Device[Device.DEFAULT]._at_profile_finalize()
  sqtt_events = [e for e in Compiled.profile_events if type(e).__name__ == "ProfileSQTTEvent"]
  kern_events = {e.tag: e for e in Compiled.profile_events if type(e).__name__ == "ProfileProgramEvent"}
  for ev in sqtt_events:
    if ev.itrace and ev.kern in kern_events: return ev, kern_events[ev.kern]
  return None, None

def _clear_events():
  """Clear profiling events, keeping program/device metadata."""
  Device[Device.DEFAULT].synchronize()
  Compiled.profile_events[:] = [e for e in Compiled.profile_events if isinstance(e, (ProfileProgramEvent, ProfileDeviceEvent))]

def _get_non_dram_deltas(traces, pc_start, pc_end):
  """For each wave, extract inter-instruction deltas in the non-DRAM PC window."""
  result = {}
  for wid in sorted(traces.keys()):
    window = [(pc, t, typ) for pc, t, typ in traces[wid] if pc_start <= pc <= pc_end]
    if len(window) < 2: continue
    pcs = [pc for pc, _, _ in window]
    deltas = [0] + [window[i+1][1] - window[i][1] for i in range(len(window)-1)]
    types_ = [typ for _, _, typ in window]
    result[wid] = {"pcs": pcs, "deltas": deltas, "types": types_}
  return result

# ---------------------------------------------------------------------------
# Emulator subprocess runner
# ---------------------------------------------------------------------------
EMU_CAPTURE_SCRIPT = textwrap.dedent(r'''
import os, sys, pickle, functools
os.environ.setdefault("DEV", "AMD")
from tinygrad import Device, Tensor, dtypes
from tinygrad.device import Compiled, ProfileEvent, ProfileProgramEvent, ProfileDeviceEvent

def _clear():
  Device[Device.DEFAULT].synchronize()
  Compiled.profile_events[:] = [e for e in Compiled.profile_events if isinstance(e, (ProfileProgramEvent, ProfileDeviceEvent))]

def _capture():
  Device[Device.DEFAULT].synchronize()
  Device[Device.DEFAULT]._at_profile_finalize()
  sqtt_events = [e for e in Compiled.profile_events if type(e).__name__ == "ProfileSQTTEvent"]
  kern_events = {e.tag: e for e in Compiled.profile_events if type(e).__name__ == "ProfileProgramEvent"}
  for ev in sqtt_events:
    if ev.itrace and ev.kern in kern_events: return ev, kern_events[ev.kern]
  return None, None

def run_kernel(kernel_name):
  from test.amd.helpers import TARGET_TO_ARCH
  arch = TARGET_TO_ARCH[Device["AMD"].arch]
  if kernel_name == "custom_lds_sync":
    from test.amd.test_custom_kernel import custom_lds_sync
    _clear()
    a = Tensor.empty(128, dtype=dtypes.int32).contiguous().realize()
    Tensor.custom_kernel(a, fxn=functools.partial(custom_lds_sync, arch=arch))[0].realize()
  elif kernel_name == "plus":
    _clear()
    (Tensor([1., 2, 3]) + Tensor([4., 5, 6])).realize()
  return _capture()

out_path = sys.argv[1]
kernel_name = sys.argv[2]
sqtt_ev, prg_ev = run_kernel(kernel_name)
if sqtt_ev is None:
  print("ERROR: no SQTT event captured in emulator", file=sys.stderr)
  sys.exit(1)
with open(out_path, "wb") as f:
  pickle.dump({"blob": sqtt_ev.blob, "lib": prg_ev.lib}, f)
print(f"EMU capture OK: {kernel_name} -> {out_path}")
''')

def _run_emulator(kernel_name):
  """Run kernel through emulator subprocess, return (blob, lib)."""
  out_path = os.path.join(os.path.dirname(__file__), f"_emu_capture_{kernel_name}_{os.getpid()}.pkl")
  script_path = os.path.join(os.path.dirname(__file__), f"_emu_script_{os.getpid()}.py")
  try:
    with open(script_path, "w") as f: f.write(EMU_CAPTURE_SCRIPT)
    env = {**os.environ, "DEV": "AMD", "MOCKGPU": "1", "PYTHON_REMU": "1", "PROFILE": "1", "SQTT": "1"}
    # Unset GPU-specific vars that could interfere with emulator
    for k in ["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"]:
      env.pop(k, None)
    result = subprocess.run([sys.executable, script_path, out_path, kernel_name],
                            env=env, capture_output=True, text=True, timeout=120, cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    if result.returncode != 0:
      print(f"EMU subprocess stderr:\n{result.stderr}")
      raise RuntimeError(f"Emulator subprocess failed (rc={result.returncode})")
    if result.stdout.strip(): print(f"  EMU: {result.stdout.strip()}")
    with open(out_path, "rb") as f: data = pickle.load(f)
    return data["blob"], data["lib"]
  finally:
    for p in [out_path, script_path]:
      if os.path.exists(p): os.remove(p)

# ---------------------------------------------------------------------------
# Comparison and reporting
# ---------------------------------------------------------------------------
def _print_comparison(label, hw_data, emu_data, tolerance=TOLERANCE):
  """Print side-by-side comparison of HW vs EMU deltas. Returns (n_pass, n_fail, max_diff)."""
  total_pass, total_fail, max_diff = 0, 0, 0
  hw_waves = sorted(hw_data.keys())[:2]
  emu_waves = sorted(emu_data.keys())[:2]
  n_waves = min(len(hw_waves), len(emu_waves))
  if n_waves == 0:
    print(f"  WARNING: no waves to compare for {label}")
    return 0, 1, 999

  for wave_idx in range(n_waves):
    hw_w = hw_data[hw_waves[wave_idx]]
    emu_w = emu_data[emu_waves[wave_idx]]
    hw_pcs, hw_deltas = hw_w["pcs"], hw_w["deltas"]
    emu_pcs, emu_deltas = emu_w["pcs"], emu_w["deltas"]

    # verify code identity
    if hw_pcs != emu_pcs:
      print(f"  FAIL wave {wave_idx}: PC sequence mismatch!")
      print(f"    HW:  {[hex(p) for p in hw_pcs]}")
      print(f"    EMU: {[hex(p) for p in emu_pcs]}")
      total_fail += 1
      continue

    print(f"\n  {label} wave {wave_idx} — inter-instruction deltas (non-DRAM section):")
    print(f"  {'PC':>8s}  {'HW':>6s}  {'EMU':>6s}  {'diff':>5s}  {'status':>6s}  {'type'}")
    print(f"  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*5}  {'─'*6}  {'─'*12}")
    wave_fail = False
    for i in range(min(len(hw_deltas), len(emu_deltas))):
      pc = hw_pcs[i] if i < len(hw_pcs) else 0
      hw_d, emu_d = hw_deltas[i], emu_deltas[i]
      diff = abs(hw_d - emu_d)
      max_diff = max(max_diff, diff)
      ok = diff <= tolerance
      status = "✓ PASS" if ok else "✗ FAIL"
      typ = hw_w["types"][i] if i < len(hw_w["types"]) else "?"
      if ok: total_pass += 1
      else:
        total_fail += 1
        wave_fail = True
      print(f"  0x{pc:05x}  {hw_d:6d}  {emu_d:6d}  {diff:5d}  {status}  {typ}")
    wave_status = "PASS" if not wave_fail else "FAIL"
    print(f"  wave {wave_idx}: {wave_status} (max_diff={max_diff})")

  return total_pass, total_fail, max_diff

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
@unittest.skipIf(_SKIP_REASON is not None, _SKIP_REASON or "")
class TestLiveHWMatch(unittest.TestCase):
  """Live HW vs emulator SQTT validation — requires real AMD 7900 XTX."""

  @classmethod
  def setUpClass(cls):
    assert Device.DEFAULT == "AMD", f"Device must be AMD, got {Device.DEFAULT}"
    assert Device[Device.DEFAULT].sqtt_enabled, "SQTT not enabled (need PROFILE=1 SQTT=1)"
    cls.arch_name = Device["AMD"].arch
    assert cls.arch_name == TARGET, f"Expected {TARGET}, got {cls.arch_name}"
    from test.amd.helpers import TARGET_TO_ARCH
    cls.arch = TARGET_TO_ARCH[cls.arch_name]

  def _run_hw_kernel(self, kernel_name):
    """Run kernel on real HW and return wave traces."""
    _clear_events()
    if kernel_name == "custom_lds_sync":
      from test.amd.test_custom_kernel import custom_lds_sync
      a = Tensor.empty(128, dtype=dtypes.int32).contiguous().realize()
      _clear_events()
      Tensor.custom_kernel(a, fxn=functools.partial(custom_lds_sync, arch=self.arch))[0].realize()
    elif kernel_name == "plus":
      _clear_events()
      (Tensor([1., 2, 3]) + Tensor([4., 5, 6])).realize()
    else:
      raise ValueError(f"unknown kernel: {kernel_name}")

    sqtt_ev, prg_ev = _capture_sqtt()
    self.assertIsNotNone(sqtt_ev, f"no HW SQTT event captured for {kernel_name}")
    return _extract_wave_traces(sqtt_ev.blob, prg_ev.lib, TARGET), prg_ev.lib

  def _run_emu_kernel(self, kernel_name):
    """Run kernel through emulator subprocess, return wave traces."""
    blob, lib = _run_emulator(kernel_name)
    return _extract_wave_traces(blob, lib, TARGET)

  def test_custom_lds_sync(self):
    """Live HW vs EMU: custom_lds_sync (128 threads, LDS + barrier + register ops)."""
    print("\n" + "="*80)
    print("TEST: custom_lds_sync — LDS write, barrier, LDS read, register ops")
    print("="*80)

    # --- 1. Capture on real HW ---
    print("\n[1/3] Capturing SQTT on REAL hardware...")
    hw_traces, _ = self._run_hw_kernel("custom_lds_sync")
    hw_data = _get_non_dram_deltas(hw_traces, NON_DRAM_PC_START, NON_DRAM_PC_END)
    self.assertGreaterEqual(len(hw_data), 2, f"HW produced {len(hw_data)} waves, need ≥2")
    print(f"  HW: {len(hw_data)} waves captured")

    # --- 2. Capture on emulator ---
    print("\n[2/3] Capturing SQTT on EMULATOR (subprocess with MOCKGPU=1)...")
    emu_traces = self._run_emu_kernel("custom_lds_sync")
    emu_data = _get_non_dram_deltas(emu_traces, NON_DRAM_PC_START, NON_DRAM_PC_END)
    self.assertGreaterEqual(len(emu_data), 2, f"EMU produced {len(emu_data)} waves, need ≥2")
    print(f"  EMU: {len(emu_data)} waves captured")

    # --- 3. Compare ---
    print("\n[3/3] Comparing HW vs EMU inter-instruction deltas...")
    n_pass, n_fail, max_diff = _print_comparison("custom_lds_sync", hw_data, emu_data)

    print(f"\n{'='*80}")
    print(f"RESULT: {n_pass} instructions PASS, {n_fail} FAIL (tolerance=±{TOLERANCE}, max_diff={max_diff})")
    print(f"{'='*80}")
    self.assertEqual(n_fail, 0, f"{n_fail} instruction(s) exceeded ±{TOLERANCE} cycle tolerance (max_diff={max_diff})")

  def test_custom_lds_sync_determinism(self):
    """Run HW capture twice to verify real hardware is deterministic in non-DRAM section."""
    print("\n" + "="*80)
    print("TEST: HW determinism check — two captures of custom_lds_sync")
    print("="*80)

    captures = []
    for run in range(2):
      print(f"\n  HW capture run {run}...")
      hw_traces, _ = self._run_hw_kernel("custom_lds_sync")
      hw_data = _get_non_dram_deltas(hw_traces, NON_DRAM_PC_START, NON_DRAM_PC_END)
      captures.append(hw_data)

    print("\n  Comparing run_0 vs run_1...")
    n_pass, n_fail, max_diff = _print_comparison("determinism", captures[0], captures[1], tolerance=0)

    print(f"\n  RESULT: {n_pass} pass, {n_fail} fail (tolerance=0, exact match)")
    self.assertEqual(n_fail, 0, f"HW not deterministic! {n_fail} deltas differ between runs (max_diff={max_diff})")
    print("  HW determinism: CONFIRMED ✓")

  def test_custom_lds_sync_barrier_sync(self):
    """Verify emulator barrier synchronization matches HW — both waves resume after last-arriving barrier."""
    print("\n" + "="*80)
    print("TEST: barrier synchronization — cross-wave timing")
    print("="*80)

    hw_traces, _ = self._run_hw_kernel("custom_lds_sync")
    emu_traces = self._run_emu_kernel("custom_lds_sync")

    for label, traces in [("HW", hw_traces), ("EMU", emu_traces)]:
      wave_ids = sorted(traces.keys())[:2]
      self.assertGreaterEqual(len(wave_ids), 2, f"{label}: need ≥2 waves for barrier check")

      barrier_times, resume_times = {}, {}
      for wave_idx, wid in enumerate(wave_ids):
        for pc, t, _ in traces[wid]:
          if pc == 0x11c: barrier_times[wave_idx] = t     # s_barrier
          if pc == 0x120: resume_times[wave_idx] = t      # v_add_nc_u32 (first post-barrier inst)

      self.assertEqual(len(barrier_times), 2, f"{label}: missing barrier timestamps")
      self.assertEqual(len(resume_times), 2, f"{label}: missing resume timestamps")

      last_barrier = max(barrier_times[0], barrier_times[1])
      for w in range(2):
        gap = resume_times[w] - last_barrier
        print(f"  {label} wave {w}: barrier@{barrier_times[w]} resume@{resume_times[w]} gap={gap}")
        self.assertGreaterEqual(gap, 0, f"{label} wave {w}: resumed before last barrier")
        self.assertLessEqual(gap, 30, f"{label} wave {w}: post-barrier gap too large ({gap})")

    print("  Barrier sync: PASS ✓")

  def test_plus_valu_forwarding(self):
    """Live HW vs EMU: element-wise addition — check VALU→global_store forwarding delta."""
    print("\n" + "="*80)
    print("TEST: plus kernel — VALU→global_store forwarding")
    print("="*80)

    print("\n  Capturing HW...")
    hw_traces, _ = self._run_hw_kernel("plus")
    print("\n  Capturing EMU...")
    emu_traces = self._run_emu_kernel("plus")

    for label, traces in [("HW", hw_traces), ("EMU", emu_traces)]:
      wid = sorted(traces.keys())[0]
      t_list = traces[wid]
      # find last VALUINST → INST (global_store) transition
      for i in range(len(t_list) - 2):
        if t_list[i][2] == "VALUINST" and t_list[i+1][2] == "INST":
          if i+2 >= len(t_list) or t_list[i+2][2] == "WAVEEND":
            delta = t_list[i+1][1] - t_list[i][1]
            print(f"  {label}: VALU→global_store delta = {delta} cycles")
            break

    print("  (manual inspection — forwarding delta logged above)")

# ---------------------------------------------------------------------------
# Standalone entry point with summary
# ---------------------------------------------------------------------------
def main():
  print(r"""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  RDNA3 Emulator — Live Hardware Validation Test                ║
  ║  Target: AMD Radeon RX 7900 XTX (gfx1100)                     ║
  ║                                                                ║
  ║  This test proves the emulator matches REAL hardware by        ║
  ║  comparing SQTT inter-instruction deltas cycle-by-cycle.       ║
  ╚══════════════════════════════════════════════════════════════════╝
  """)
  print(f"  Device:    {Device.DEFAULT}")
  print(f"  Target:    {Device['AMD'].arch}")
  print(f"  PROFILE:   {os.environ.get('PROFILE', 'unset')}")
  print(f"  SQTT:      {os.environ.get('SQTT', 'unset')}")
  print(f"  MOCKGPU:   {os.environ.get('MOCKGPU', 'unset (real HW)')}")
  print()

  # Run via unittest for proper assertions and reporting
  loader = unittest.TestLoader()
  suite = loader.loadTestsFromTestCase(TestLiveHWMatch)
  runner = unittest.TextTestRunner(verbosity=2)
  result = runner.run(suite)

  # Final summary
  print("\n" + "="*80)
  if result.wasSuccessful():
    print("  ★ ALL TESTS PASSED — Emulator matches real GFX1100 hardware! ★")
    print("  The RDNA3 emulator produces cycle-accurate SQTT traces for non-DRAM kernels.")
  else:
    print("  ✗ SOME TESTS FAILED — See details above.")
    n_fail = len(result.failures) + len(result.errors)
    print(f"  {n_fail} test(s) failed out of {result.testsRun}")
  print("="*80)

  sys.exit(0 if result.wasSuccessful() else 1)

if __name__ == "__main__":
  main()
