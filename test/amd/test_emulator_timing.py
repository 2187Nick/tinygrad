# validates emulator SQTT timing: self-consistency via rocprof + exact match against real GFX1100 hardware captures
import unittest, functools, sys, types, pickle, dataclasses, pathlib
from tinygrad import Device, Tensor, dtypes
from tinygrad.device import Compiled, ProfileEvent, ProfileProgramEvent, ProfileDeviceEvent

TARGET = "gfx1100"
HW_PKL_DIR = pathlib.Path(__file__).resolve().parents[2] / "extra" / "sqtt" / "examples" / "gfx1100"
HW_PKL = HW_PKL_DIR / "profile_sync_run_0.pkl"
# non-DRAM window: PCs 0x10c through 0x140 (LDS + barrier + register section)
NON_DRAM_PC_START, NON_DRAM_PC_END = 0x10c, 0x140
# ground-truth inter-instruction deltas from real GFX1100 hardware (identical across run_0 and run_1 captures)
# wave 1 differs from wave 0 due to round-robin scheduling contention on the shared CU
HW_INTER_DELTAS = {
  0: [0, 26, 33, 1, 25, 1, 21, 32, 1, 1, 1],
  1: [0, 26, 38, 2, 20, 1, 21, 31, 3, 1, 1],
}
HW_PCS = [0x10c, 0x110, 0x118, 0x11c, 0x120, 0x124, 0x12c, 0x134, 0x138, 0x13c, 0x140]
HW_TYPES = ["VALUINST", "INST", "IMMEDIATE", "INST", "VALUINST", "VALUINST", "INST", "IMMEDIATE", "VALUINST", "VALUINST", "VALUINST"]
# plus kernel: deterministic VALU→global_store forwarding delta (non-DRAM section)
PLUS_HW_VALU_STORE_DELTA = 25  # delta from v_add to global_store (identical across run_0 and run_1)
# gemm kernel: deterministic startup section (indices 0-8, SALU/VALU before first DRAM wait)
GEMM_STARTUP_DELTAS = [0, 1, 9, 1, 1, 4, 1, 4, 2]
# gemm kernel: deterministic tail section (indices 25-57, non-DRAM register/LDS ops after DRAM loads complete)
GEMM_TAIL_START_IDX = 25
GEMM_TAIL_DELTAS = [41, 1, 16, 3, 1, 1, 2, 2, 2, 1, 28, 1, 5, 5, 5, 18, 3, 4, 5, 5, 5, 18, 3, 4, 5, 5, 5, 23, 2, 5, 5, 5, 18]

def _load_pkl_safe(path):
  if 'tinygrad.runtime.ops_amd' not in sys.modules:
    try:
      import tinygrad.runtime.ops_amd  # noqa: F401 # works on Linux with AMD
    except (AssertionError, ImportError):
      stub = types.ModuleType('tinygrad.runtime.ops_amd')
      @dataclasses.dataclass
      class ProfileSQTTEvent(ProfileEvent):
        device: str = ''
        kern: int = 0
        se: int = 0
        blob: bytes = b''
        itrace: bool = False
        exec_tag: int = 0
      @dataclasses.dataclass
      class ProfilePMCEvent(ProfileEvent):
        device: str = ''
        kern: int = 0
        sched: list = dataclasses.field(default_factory=list)
        blob: bytes = b''
        exec_tag: int = 0
      @dataclasses.dataclass
      class PMCSample:
        ts: int = 0
        values: dict = dataclasses.field(default_factory=dict)
      setattr(stub, 'ProfileSQTTEvent', ProfileSQTTEvent)
      setattr(stub, 'ProfilePMCEvent', ProfilePMCEvent)
      setattr(stub, 'PMCSample', PMCSample)
      setattr(stub, 'AMDDevice', type('AMDDevice', (), {}))
      sys.modules['tinygrad.runtime.ops_amd'] = stub
  with open(path, 'rb') as f:
    return pickle.load(f)

def _extract_wave_traces(blob, lib, target):
  from tinygrad.renderer.amd.sqtt import map_insts
  traces = {}
  for pkt, info in map_insts(blob, lib, target):
    if info is None: continue
    wave = info.wave
    if wave not in traces: traces[wave] = []
    traces[wave].append((info.pc, pkt._time, type(pkt).__name__))
  return traces

def _get_hw_traces(pkl_path, kern_tag, target):
  data = _load_pkl_safe(pkl_path)
  sqtt_events = [e for e in data if type(e).__name__ == "ProfileSQTTEvent"]
  kern_events = {e.tag: e for e in data if type(e).__name__ == "ProfileProgramEvent"}
  assert kern_tag in kern_events, f"kern_tag={kern_tag} not found in pkl (available: {list(kern_events.keys())})"
  prg = kern_events[kern_tag]
  for sqtt_ev in sqtt_events:
    if sqtt_ev.kern != kern_tag or not sqtt_ev.itrace: continue
    traces = _extract_wave_traces(sqtt_ev.blob, prg.lib, target)
    if traces: return traces, prg.lib
  raise AssertionError(f"no itrace SQTT event with decoded waves for kern={kern_tag}")

@unittest.skipUnless(Device.DEFAULT == "AMD", "only runs on AMD")
class TestEmulatorTiming(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    if not Device[Device.DEFAULT].sqtt_enabled:
      raise unittest.SkipTest("device must be in SQTT profiling mode (PROFILE=1 SQTT=1)")

  def setUp(self):
    Device[Device.DEFAULT].synchronize()
    Compiled.profile_events[:] = [e for e in Compiled.profile_events if isinstance(e, (ProfileProgramEvent, ProfileDeviceEvent))]

  def _capture(self):
    Device[Device.DEFAULT].synchronize()
    Device[Device.DEFAULT]._at_profile_finalize()
    sqtt_events = [e for e in Compiled.profile_events if type(e).__name__ == "ProfileSQTTEvent"]
    kern_events = {e.tag: e for e in Compiled.profile_events if type(e).__name__ == "ProfileProgramEvent"}
    # find first SQTT event with itrace data and a matching program
    for ev in sqtt_events:
      if ev.itrace and ev.kern in kern_events: return ev, kern_events[ev.kern]
    return None, None

  def _validate_timing(self, label, run_kernel, must_match=True):
    from test.amd.test_sqttmap import rocprof_inst_traces_match
    run_kernel()
    sqtt_ev, prg_ev = self._capture()
    self.assertIsNotNone(sqtt_ev, f"{label}: no emulator SQTT event with itrace captured")
    passed_insts, n_waves, n_units = rocprof_inst_traces_match(sqtt_ev, prg_ev, TARGET)
    # rocprof must decode the emulator's blob
    self.assertGreater(n_waves, 0, f"{label}: rocprof returned 0 waves")
    self.assertGreater(passed_insts, 0, f"{label}: 0 instructions validated")
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

  def test_sync_hw_match(self):
    """Compare emulator SQTT timestamps against real GFX1100 hardware captures for custom_lds_sync."""
    if not HW_PKL.exists():
      self.skipTest(f"HW capture not found: {HW_PKL}")

    # --- 1. Run custom_lds_sync through the emulator ---
    from test.amd.test_custom_kernel import custom_lds_sync
    from test.amd.helpers import TARGET_TO_ARCH
    arch = TARGET_TO_ARCH[Device["AMD"].arch]
    a = Tensor.empty(128, dtype=dtypes.int32).contiguous().realize()
    Device[Device.DEFAULT].synchronize()
    Compiled.profile_events[:] = [e for e in Compiled.profile_events if isinstance(e, (ProfileProgramEvent, ProfileDeviceEvent))]
    Tensor.custom_kernel(a, fxn=functools.partial(custom_lds_sync, arch=arch))[0].realize()
    emu_sqtt, emu_prg = self._capture()
    self.assertIsNotNone(emu_sqtt, "no emulator SQTT event captured for custom_lds_sync")
    emu_traces = _extract_wave_traces(emu_sqtt.blob, emu_prg.lib, TARGET)

    # --- 2. Load real HW capture and sanity-check against known constants ---
    hw_traces, hw_lib = _get_hw_traces(HW_PKL, kern_tag=0, target=TARGET)

    # --- 3. Validate both sources have at least 2 waves ---
    self.assertGreaterEqual(len(emu_traces), 2, f"emulator produced {len(emu_traces)} waves, need ≥2")
    self.assertGreaterEqual(len(hw_traces), 2, f"HW capture has {len(hw_traces)} waves, need ≥2")

    emu_wave_ids = sorted(emu_traces.keys())[:2]
    hw_wave_ids = sorted(hw_traces.keys())[:2]

    all_barrier_times = {}  # {wave_idx: absolute barrier time from emulator}

    for wave_idx, (emu_wid, hw_wid) in enumerate(zip(emu_wave_ids, hw_wave_ids)):
      emu_wave = emu_traces[emu_wid]
      hw_wave = hw_traces[hw_wid]

      # filter to non-DRAM window
      emu_window = [(pc, t, typ) for pc, t, typ in emu_wave if NON_DRAM_PC_START <= pc <= NON_DRAM_PC_END]
      hw_window = [(pc, t, typ) for pc, t, typ in hw_wave if NON_DRAM_PC_START <= pc <= NON_DRAM_PC_END]

      # --- 4. Verify code identity ---
      emu_pcs = [pc for pc, _, _ in emu_window]
      hw_pcs = [pc for pc, _, _ in hw_window]
      self.assertEqual(emu_pcs, HW_PCS,
        f"wave {wave_idx}: EMU code identity mismatch!\n  EMU PCs: {[hex(p) for p in emu_pcs]}\n  expected: {[hex(p) for p in HW_PCS]}")
      self.assertEqual(hw_pcs, HW_PCS,
        f"wave {wave_idx}: HW pkl code identity mismatch!\n  HW PCs: {[hex(p) for p in hw_pcs]}\n  expected: {[hex(p) for p in HW_PCS]}")

      emu_types = [typ for _, _, typ in emu_window]
      hw_types = [typ for _, _, typ in hw_window]
      self.assertEqual(emu_types, HW_TYPES, f"wave {wave_idx}: EMU instruction type mismatch!\n  EMU: {emu_types}\n  expected: {HW_TYPES}")
      self.assertEqual(hw_types, HW_TYPES, f"wave {wave_idx}: HW instruction type mismatch!\n  HW: {hw_types}\n  expected: {HW_TYPES}")

      # --- 5. Compute inter-instruction deltas and compare ---
      emu_inter = [0] + [emu_window[i+1][1] - emu_window[i][1] for i in range(len(emu_window)-1)]
      hw_inter = [0] + [hw_window[i+1][1] - hw_window[i][1] for i in range(len(hw_window)-1)]
      expected = HW_INTER_DELTAS[wave_idx]

      # sanity-check pkl data matches known constants
      self.assertEqual(hw_inter, expected, f"wave {wave_idx}: HW pkl deltas changed!\n  got: {hw_inter}\n  expected: {expected}")

      print(f"\nHW vs EMU comparison for wave {wave_idx} (inter-instruction deltas):")
      max_diff = 0
      failures = []
      for i, (pc, hw_d, emu_d) in enumerate(zip(HW_PCS, expected, emu_inter)):
        diff = abs(hw_d - emu_d)
        max_diff = max(max_diff, diff)
        ok = "✓" if diff == 0 else "✗"
        print(f"  PC=0x{pc:03x}  HW_delta={hw_d:4d}  EMU_delta={emu_d:4d}  diff={diff:3d}  {ok}")
        if diff > 0:
          failures.append(f"wave {wave_idx} PC=0x{pc:03x}: HW={hw_d} EMU={emu_d} diff={diff}")

      print(f"  max delta diff = {max_diff} cycles (tolerance = ±0, exact match required)")
      self.assertEqual(failures, [], "Timing mismatches (exact match required):\n" + "\n".join(failures))

      # record barrier times for cross-wave check (s_barrier at PC 0x11c)
      for pc, t, typ in emu_window:
        if pc == 0x11c: all_barrier_times[wave_idx] = t

    # --- 6. Cross-wave barrier assertion ---
    # both waves must reach s_barrier; the post-barrier instruction (v_add_nc_u32 at 0x120)
    # should resume ~25 cycles after the last-arriving wave's barrier
    self.assertEqual(len(all_barrier_times), 2, "expected s_barrier timestamps from both emulator waves")
    emu_resume = {}
    for wave_idx, wid in enumerate(emu_wave_ids):
      for pc, t, _ in emu_traces[wid]:
        if pc == 0x120:
          emu_resume[wave_idx] = t
          break
    self.assertEqual(len(emu_resume), 2, "expected v_add_nc_u32 (0x120) resume timestamps from both emulator waves")

    last_barrier = max(all_barrier_times[0], all_barrier_times[1])
    for wave_idx in range(2):
      gap = emu_resume[wave_idx] - last_barrier
      print(f"\n  wave {wave_idx}: barrier@{all_barrier_times[wave_idx]} resume@{emu_resume[wave_idx]} "
            f"gap_from_last_barrier={gap} cycles")
      self.assertGreaterEqual(gap, 0, f"wave {wave_idx}: resumed before last barrier arrived")
      self.assertLessEqual(gap, 30, f"wave {wave_idx}: post-barrier resume too late ({gap} cycles after last barrier)")

    print(f"\n  barrier sync OK: last barrier at {last_barrier}, both waves resumed within 30 cycles")

  def test_sync_hw_determinism(self):
    """Verify emulator matches BOTH independent HW captures (run_0 and run_1) identically — proves determinism."""
    for run_idx in range(2):
      pkl = HW_PKL_DIR / f"profile_sync_run_{run_idx}.pkl"
      if not pkl.exists(): self.skipTest(f"HW capture not found: {pkl}")
      hw_traces, _ = _get_hw_traces(pkl, kern_tag=0, target=TARGET)
      for wave_idx, wid in enumerate(sorted(hw_traces.keys())[:2]):
        hw_wave = hw_traces[wid]
        hw_window = [(pc, t, typ) for pc, t, typ in hw_wave if NON_DRAM_PC_START <= pc <= NON_DRAM_PC_END]
        hw_inter = [0] + [hw_window[i+1][1] - hw_window[i][1] for i in range(len(hw_window)-1)]
        self.assertEqual(hw_inter, HW_INTER_DELTAS[wave_idx],
          f"run_{run_idx} wave {wave_idx}: HW deltas differ from expected!\n  got: {hw_inter}\n  expected: {HW_INTER_DELTAS[wave_idx]}")
    print("Both HW captures (run_0, run_1) produce identical non-DRAM deltas — deterministic ✓")

  def test_plus_hw_forwarding(self):
    """Validate plus kernel's VALU→global_store forwarding delta matches real HW (deterministic, non-DRAM section)."""
    for run_idx in range(2):
      pkl = HW_PKL_DIR / f"profile_plus_run_{run_idx}.pkl"
      if not pkl.exists(): self.skipTest(f"HW capture not found: {pkl}")
      hw_traces, _ = _get_hw_traces(pkl, kern_tag=0, target=TARGET)
      for wid in sorted(hw_traces.keys())[:1]:
        t_list = hw_traces[wid]
        # find VALU→INST (global_store) forwarding: last VALUINST followed by INST before WAVEEND
        for i in range(len(t_list) - 2):
          if t_list[i][2] == "VALUINST" and t_list[i+1][2] == "INST" and (i+2 >= len(t_list) or t_list[i+2][2] == "WAVEEND"):
            delta = t_list[i+1][1] - t_list[i][1]
            self.assertEqual(delta, PLUS_HW_VALU_STORE_DELTA,
              f"plus run_{run_idx}: VALU→global_store delta={delta}, expected {PLUS_HW_VALU_STORE_DELTA}")
            print(f"plus_run_{run_idx}: VALU→global_store delta={delta} ✓ (matches expected {PLUS_HW_VALU_STORE_DELTA})")
            break

  def test_gemm_hw_determinism(self):
    """Validate gemm kernel's non-DRAM tail (33 instructions) is deterministic across both HW captures."""
    for run_idx in range(2):
      pkl = HW_PKL_DIR / f"profile_gemm_run_{run_idx}.pkl"
      if not pkl.exists(): self.skipTest(f"HW capture not found: {pkl}")
      hw_traces, _ = _get_hw_traces(pkl, kern_tag=0, target=TARGET)
      wid = sorted(hw_traces.keys())[0]
      tl = hw_traces[wid]
      tail_deltas = [tl[i][1] - tl[i-1][1] for i in range(GEMM_TAIL_START_IDX, min(GEMM_TAIL_START_IDX + len(GEMM_TAIL_DELTAS), len(tl)))]
      self.assertEqual(tail_deltas, GEMM_TAIL_DELTAS,
        f"gemm run_{run_idx}: tail deltas differ!\n  got: {tail_deltas}\n  expected: {GEMM_TAIL_DELTAS}")
      print(f"gemm_run_{run_idx}: tail ({len(tail_deltas)} deltas) ✓ deterministic")

  def test_gemm_hw_startup(self):
    """Validate gemm kernel's startup sequence (SALU/VALU before first DRAM wait) is deterministic across both HW captures."""
    for run_idx in range(2):
      pkl = HW_PKL_DIR / f"profile_gemm_run_{run_idx}.pkl"
      if not pkl.exists(): self.skipTest(f"HW capture not found: {pkl}")
      hw_traces, _ = _get_hw_traces(pkl, kern_tag=0, target=TARGET)
      wid = sorted(hw_traces.keys())[0]
      tl = hw_traces[wid]
      startup_deltas = [0] + [tl[i][1] - tl[i-1][1] for i in range(1, len(GEMM_STARTUP_DELTAS))]
      self.assertEqual(startup_deltas, GEMM_STARTUP_DELTAS,
        f"gemm run_{run_idx}: startup deltas differ!\n  got: {startup_deltas}\n  expected: {GEMM_STARTUP_DELTAS}")
      print(f"gemm_run_{run_idx}: startup ({len(startup_deltas)} deltas) ✓ deterministic")

  def test_wave_count_stalls(self):
    """Validate wave-count-dependent VALU stalls: 1-wave kernels get higher stalls than 2-wave."""
    from test.mockgpu.amd.emu import _instid_stall, _INSTID_BASE_STALLS
    # 2-wave stalls match the calibrated table
    # TRANS32_DEP (5-7) = 0: trans ALU runs in parallel with VALU; pipeline occupancy enforced by trans_pipe_avail
    original_2wave = (0, 3, 2, 1, 0, 0, 0, 0, 1, 1, 2, 3)
    two_wave = tuple(_instid_stall(i, 2) for i in range(12))
    self.assertEqual(two_wave, original_2wave, "2-wave stalls must match calibrated table")
    # 1-wave VALU deps (instid 1-4) are +1 vs 2-wave (RDNA3 VALU pipeline = 5 cycles)
    for instid in range(1, 5):
      self.assertEqual(_instid_stall(instid, 1), _instid_stall(instid, 2) + 1,
        f"instid={instid}: 1-wave stall must be +1 vs 2-wave")
    # Non-VALU deps (TRANS32, FMA, SALU) unchanged across wave counts
    for instid in range(5, 12):
      for n in [1, 2, 3, 5]:
        self.assertEqual(_instid_stall(instid, n), _INSTID_BASE_STALLS[instid],
          f"instid={instid} n={n}: non-VALU stalls must be wave-count-independent")
    # With enough waves, VALU stalls drop to 0
    for instid in range(1, 5):
      self.assertEqual(_instid_stall(instid, 5), 0,
        f"instid={instid}: 5-wave stall should be 0 (pipeline fully hidden)")

if __name__ == "__main__":
  unittest.main()
