"""
Validates that the RDNA3 mockgpu SQTT emulator generates cycle-accurate timing that
matches real hardware GFX1100 SQTT captures for non-DRAM kernels.

Run with: DEV=AMD MOCKGPU=1 PYTHON_REMU=1 PROFILE=1 SQTT=1 python -m pytest test/amd/test_emulator_timing.py
"""
import unittest, pickle, functools
from pathlib import Path
from tinygrad import Device, Tensor, dtypes
from tinygrad.uop.ops import UOp, KernelInfo
from tinygrad.device import Compiled, ProfileProgramEvent, ProfileDeviceEvent
from tinygrad.renderer.amd.sqtt import map_insts

import tinygrad
EXAMPLES_DIR = Path(tinygrad.__file__).parent.parent / "extra/sqtt/examples"
TARGET = "gfx1100"

def extract_wave_timing(blob, lib, target):
  """Decode SQTT blob and return per-wave instruction timing normalized to first instruction per wave.
  Returns dict: wave_id -> list of (relative_cycle, InstructionInfo).
  """
  wave_t0, waves = {}, {}
  for pkt, info in map_insts(blob, lib, target):
    if info is None: continue
    w, t = info.wave, pkt._time
    if w not in wave_t0: wave_t0[w] = t
    waves.setdefault(w, []).append((t - wave_t0[w], info))
  return waves

@unittest.skipUnless(Device.DEFAULT == "AMD", "only runs on AMD")
class TestEmulatorTiming(unittest.TestCase):
  """Compare mockgpu emulator SQTT timing against real GFX1100 hardware SQTT captures."""

  @classmethod
  def setUpClass(cls):
    if not Device[Device.DEFAULT].sqtt_enabled:
      raise unittest.SkipTest("device must be in SQTT profiling mode (PROFILE=1 SQTT=1)")
    hw_dir = EXAMPLES_DIR / TARGET
    if not hw_dir.exists(): raise unittest.SkipTest(f"hardware examples not found at {hw_dir}")
    cls.hw_examples = {}
    for pkl_path in sorted(hw_dir.glob("*.pkl")):
      with open(pkl_path, "rb") as f: data = pickle.load(f)
      sqtt_events = [e for e in data if type(e).__name__ == "ProfileSQTTEvent"]
      kern_events = {e.tag: e for e in data if type(e).__name__ == "ProfileProgramEvent"}
      if sqtt_events and kern_events: cls.hw_examples[pkl_path.stem] = (sqtt_events, kern_events)

  def setUp(self):
    Device[Device.DEFAULT].synchronize()
    Compiled.profile_events[:] = [e for e in Compiled.profile_events if isinstance(e, (ProfileProgramEvent, ProfileDeviceEvent))]

  def _capture(self):
    """Finalize profiling and extract SQTT + program events."""
    Device[Device.DEFAULT].synchronize()
    Device[Device.DEFAULT]._at_profile_finalize()
    sqtt = [e for e in Compiled.profile_events if type(e).__name__ == "ProfileSQTTEvent"]
    kerns = {e.tag: e for e in Compiled.profile_events if type(e).__name__ == "ProfileProgramEvent"}
    return sqtt, kerns

  def _match_event(self, hw_prg, emu_sqtts, emu_kerns):
    """Find emulator SQTT event whose kernel lib matches the hardware program."""
    for e in emu_sqtts:
      if not e.itrace or e.kern not in emu_kerns: continue
      if emu_kerns[e.kern].lib == hw_prg.lib: return e, emu_kerns[e.kern]
    for e in emu_sqtts:
      if not e.itrace or e.kern not in emu_kerns: continue
      if emu_kerns[e.kern].name == hw_prg.name: return e, emu_kerns[e.kern]
    return None, None

  def _compare_timing(self, emu_blob, emu_lib, hw_blob, hw_lib, label):
    """Compare per-instruction relative timing between emulator and hardware SQTT blobs.
    Returns (mismatches, total_instructions). Mismatches are dicts with wave/idx/emu/hw/diff/inst keys.
    """
    emu_waves = extract_wave_timing(emu_blob, emu_lib, TARGET)
    hw_waves = extract_wave_timing(hw_blob, hw_lib, TARGET)
    self.assertEqual(len(emu_waves), len(hw_waves), f"{label}: wave count emu={len(emu_waves)} hw={len(hw_waves)}")

    mismatches, total = [], 0
    for ew, hw in zip(sorted(emu_waves), sorted(hw_waves)):
      el, hl = emu_waves[ew], hw_waves[hw]
      self.assertEqual(len(el), len(hl), f"{label}: wave {ew} instruction count emu={len(el)} hw={len(hl)}")
      for idx, ((et, ei), (ht, _hi)) in enumerate(zip(el, hl)):
        total += 1
        if et != ht: mismatches.append(dict(wave=ew, idx=idx, emu=et, hw=ht, diff=et - ht, inst=str(ei.inst)))
    if mismatches:
      print(f"\n{label}: {len(mismatches)}/{total} timing mismatches:")
      for m in mismatches[:20]:
        print(f"  wave={m['wave']} inst#{m['idx']}: emu_rel={m['emu']} hw_rel={m['hw']} diff={m['diff']}  {m['inst']}")
    return mismatches, total

  def _run_comparison(self, run_kernel, hw_key, label, must_match):
    """Run a kernel, capture emulator SQTT, compare against hardware pkl."""
    self.assertIn(hw_key, self.hw_examples, f"hardware example '{hw_key}' not found")
    hw_sqtts, hw_kerns = self.hw_examples[hw_key]
    hw_ev = next((e for e in hw_sqtts if e.itrace), None)
    self.assertIsNotNone(hw_ev, f"no itrace event in {hw_key}")
    hw_prg = hw_kerns[hw_ev.kern]

    run_kernel()
    emu_sqtts, emu_kerns = self._capture()
    self.assertGreater(len(emu_sqtts), 0, "no emulator SQTT events captured")

    emu_ev, emu_prg = self._match_event(hw_prg, emu_sqtts, emu_kerns)
    self.assertIsNotNone(emu_ev, f"no emulator event matches {hw_key} (kernel binary mismatch — hardware pkl may need regeneration)")

    mismatches, total = self._compare_timing(emu_ev.blob, emu_prg.lib, hw_ev.blob, hw_prg.lib, label)
    if must_match:
      self.assertEqual(len(mismatches), 0, f"{label}: {len(mismatches)}/{total} timing mismatches (non-DRAM kernel must match exactly)")
    return mismatches, total

  def test_empty(self):
    """Empty kernel (no DRAM) — emulator timing must match hardware exactly."""
    def run():
      a = Tensor.empty(1)
      Tensor.custom_kernel(a, fxn=lambda _: UOp.sink(arg=KernelInfo()))[0].realize()
    self._run_comparison(run, "profile_empty_run_0", "empty", must_match=True)

  def test_plus(self):
    """Elementwise addition (has global load/store DRAM ops) — compare and report differences."""
    def run(): (Tensor([1., 2, 3]) + Tensor([4., 5, 6])).realize()
    self._run_comparison(run, "profile_plus_run_0", "plus", must_match=False)

  def test_sync(self):
    """LDS barrier sync kernel (has SMEM + global store) — compare and report differences."""
    from test.amd.test_custom_kernel import custom_lds_sync
    from test.amd.helpers import TARGET_TO_ARCH
    arch = TARGET_TO_ARCH[Device["AMD"].arch]
    def run():
      a = Tensor.empty(128, dtype=dtypes.int32).contiguous().realize()
      Device[Device.DEFAULT].synchronize()
      Compiled.profile_events[:] = [e for e in Compiled.profile_events if isinstance(e, (ProfileProgramEvent, ProfileDeviceEvent))]
      Tensor.custom_kernel(a, fxn=functools.partial(custom_lds_sync, arch=arch))[0].realize()
    self._run_comparison(run, "profile_sync_run_0", "sync", must_match=False)

if __name__ == "__main__":
  unittest.main()
