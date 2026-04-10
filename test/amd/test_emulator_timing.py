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
from tinygrad.runtime.autogen.amd.rdna3.ins import s_endpgm

import tinygrad
EXAMPLES_DIR = Path(tinygrad.__file__).parent.parent / "extra/sqtt/examples"
TARGET = "gfx1100"

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

  def _compare_timing(self, emu_blob, emu_lib, hw_sqtt_ev, hw_prg, label):
    """Compare per-instruction relative timing: emulator map_insts vs hardware rocprof decoder.

    Uses rocprof-trace-decoder to extract hardware timing (handles arbitrary CU/SIMD dispatch),
    then compares against emulator timing from map_insts. Both sides are normalized to the first
    instruction in each wave so absolute clock offsets cancel out.

    Returns (mismatches, total_instructions).
    """
    from tinygrad.viz.serve import amd_decode
    from extra.sqtt.roc import decode as roc_decode

    # Build absolute-address disasm map for rocprof ISA callback
    addr_table = amd_decode(hw_prg.lib, TARGET)
    disasm_map = {addr + hw_prg.base: inst for addr, inst in addr_table.items()}

    try:
      rctx = roc_decode([hw_sqtt_ev], {hw_prg.tag: disasm_map})
    except RuntimeError as e:
      self.skipTest(f"{label}: rocprof-trace-decoder unavailable: {e}")

    rwaves = rctx.inst_execs.get((hw_sqtt_ev.kern, hw_sqtt_ev.exec_tag), [])
    if not rwaves:
      self.skipTest(f"{label}: rocprof returned no waves (SQTT blob may lack itrace data)")

    # Build per-wave iterators keyed by wave_id (0-15)
    rwaves_iter: dict[int, list] = {}
    for w in rwaves: rwaves_iter.setdefault(w.wave_id, []).append(iter(w.unpack_insts()))

    mismatches, total = [], 0
    emu_wave_t0: dict[int, int] = {}
    hw_wave_t0: dict[int, int] = {}

    for pkt, info in map_insts(emu_blob, emu_lib, TARGET):
      if info is None: continue
      wave = info.wave
      # s_endpgm marks wave completion; its timing semantics differ from issued instructions
      if info.inst == s_endpgm(): continue
      if wave not in rwaves_iter or not rwaves_iter[wave]: continue

      rocprof_inst = next(rwaves_iter[wave][0], None)
      if rocprof_inst is None: continue

      emu_t = pkt._time
      hw_t = rocprof_inst.time + rocprof_inst.stall

      # Normalize both sides to the first instruction in the wave
      if wave not in emu_wave_t0: emu_wave_t0[wave] = emu_t
      if wave not in hw_wave_t0: hw_wave_t0[wave] = hw_t

      emu_rel = emu_t - emu_wave_t0[wave]
      hw_rel = hw_t - hw_wave_t0[wave]

      total += 1
      if emu_rel != hw_rel:
        mismatches.append(dict(wave=wave, idx=total - 1, emu=emu_rel, hw=hw_rel, diff=emu_rel - hw_rel, inst=str(info.inst)))

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

    mismatches, total = self._compare_timing(emu_ev.blob, emu_prg.lib, hw_ev, hw_prg, label)
    self.assertGreater(total, 0, f"{label}: no instructions were compared")
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
