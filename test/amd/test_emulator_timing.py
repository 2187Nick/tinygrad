"""
Self-consistency test: validates that the emulator's SQTT blob is correctly decodable by rocprof
and that pkt._time == rocprof.time + rocprof.stall for every instruction.

Run with: DEV=AMD MOCKGPU=1 PYTHON_REMU=1 PROFILE=1 SQTT=1 python -m pytest test/amd/test_emulator_timing.py
"""
import unittest, functools
from tinygrad import Device, Tensor, dtypes
from tinygrad.device import Compiled, ProfileProgramEvent, ProfileDeviceEvent

TARGET = "gfx1100"

@unittest.skipUnless(Device.DEFAULT == "AMD", "only runs on AMD")
class TestEmulatorTiming(unittest.TestCase):
  """Validate emulator SQTT self-consistency: pkt._time == rocprof.time + stall for all instructions."""

  @classmethod
  def setUpClass(cls):
    if not Device[Device.DEFAULT].sqtt_enabled:
      raise unittest.SkipTest("device must be in SQTT profiling mode (PROFILE=1 SQTT=1)")

  def setUp(self):
    Device[Device.DEFAULT].synchronize()
    Compiled.profile_events[:] = [e for e in Compiled.profile_events if isinstance(e, (ProfileProgramEvent, ProfileDeviceEvent))]

  def _capture(self):
    """Finalize profiling and extract the first SQTT event with its program event."""
    Device[Device.DEFAULT].synchronize()
    Device[Device.DEFAULT]._at_profile_finalize()
    sqtt_events = [e for e in Compiled.profile_events if type(e).__name__ == "ProfileSQTTEvent"]
    kern_events = {e.tag: e for e in Compiled.profile_events if type(e).__name__ == "ProfileProgramEvent"}
    # find first SQTT event with itrace data and a matching program
    for ev in sqtt_events:
      if ev.itrace and ev.kern in kern_events: return ev, kern_events[ev.kern]
    return None, None

  def _validate_timing(self, label, run_kernel, must_match=True):
    """Run kernel, capture emulator SQTT, validate via rocprof_inst_traces_match."""
    from test.amd.test_sqttmap import rocprof_inst_traces_match
    run_kernel()
    sqtt_ev, prg_ev = self._capture()
    self.assertIsNotNone(sqtt_ev, f"{label}: no emulator SQTT event with itrace captured")
    passed_insts, n_waves, n_units = rocprof_inst_traces_match(sqtt_ev, prg_ev, TARGET)
    # rocprof must be able to decode the emulator's blob — 0 waves means the SQTT format is broken
    self.assertGreater(n_waves, 0, f"{label}: rocprof returned 0 waves — emulator SQTT blob format is broken")
    self.assertGreater(passed_insts, 0, f"{label}: 0 instructions validated — emulator SQTT blob has no instruction trace data")
    print(f"{label}: {passed_insts} instructions validated across {n_waves} waves on {n_units} units "
          f"(pkt._time == rocprof.time + stall for all)")

  def test_plus_timing_consistent(self):
    """Elementwise addition (has DRAM ops) — validate emulator SQTT is rocprof-decodable."""
    def run(): (Tensor([1., 2, 3]) + Tensor([4., 5, 6])).realize()
    self._validate_timing("plus", run)

  def test_sync_timing_consistent(self):
    """LDS barrier sync kernel — validate emulator SQTT is rocprof-decodable."""
    from test.amd.test_custom_kernel import custom_lds_sync
    from test.amd.helpers import TARGET_TO_ARCH
    arch = TARGET_TO_ARCH[Device["AMD"].arch]
    def run():
      a = Tensor.empty(128, dtype=dtypes.int32).contiguous().realize()
      Device[Device.DEFAULT].synchronize()
      Compiled.profile_events[:] = [e for e in Compiled.profile_events if isinstance(e, (ProfileProgramEvent, ProfileDeviceEvent))]
      Tensor.custom_kernel(a, fxn=functools.partial(custom_lds_sync, arch=arch))[0].realize()
    self._validate_timing("sync", run)

if __name__ == "__main__":
  unittest.main()
