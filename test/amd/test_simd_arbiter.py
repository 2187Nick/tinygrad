"""Unit tests for SimdArbiter — pure state-holder semantics.

Exercises test/mockgpu/amd/sq_timing/simd_arbiter.py directly (no MOCKGPU).
The arbiter is a skeleton in Step 1 of the SIMD-arbiter refactor — these
tests lock in the contract so later cutover steps can't silently break the
invariants that the wiring in emu.py relies on.

Shadow-wire integration (the arbiter receives updates when _simulate_sq_timing
runs under SIMD_ARB_SHADOW=1) is covered by running `rigorous_hw_test.py
--compare` with that env set and inspecting `_simd_arb_shadow_log`; not
re-tested here since it requires the full SQTT profiling stack.
"""
import unittest
from test.mockgpu.amd.sq_timing.simd_arbiter import SimdArbiter, N_SIMDS, NO_OWNER


class TestSimdForWave(unittest.TestCase):
  def test_mapping_matches_hw_id(self):
    # emu.py:3365 emits HW_ID[5:4] = wave_idx & 3. The arbiter mapping MUST
    # track that: a divergence here would mean simulation-time arbitration
    # contradicts run-time HW_ID reporting.
    for w in range(64):
      self.assertEqual(SimdArbiter.simd_for_wave(w), w & 0x3)

  def test_n_simds_constant(self):
    self.assertEqual(N_SIMDS, 4)


class TestPeersOnSimd(unittest.TestCase):
  def test_four_waves_each_on_distinct_simd(self):
    a = SimdArbiter(4)
    for w in range(4):
      self.assertEqual(a.peers_on_simd(w), [])

  def test_sixteen_waves_modulo_four_groupings(self):
    a = SimdArbiter(16)
    # Wave 0 shares SIMD 0 with waves 4, 8, 12
    self.assertEqual(a.peers_on_simd(0), [4, 8, 12])
    # Wave 5 shares SIMD 1 with waves 1, 9, 13
    self.assertEqual(a.peers_on_simd(5), [1, 9, 13])
    # Wave 15 shares SIMD 3 with waves 3, 7, 11
    self.assertEqual(a.peers_on_simd(15), [3, 7, 11])

  def test_excludes_self(self):
    a = SimdArbiter(16)
    for w in range(16):
      self.assertNotIn(w, a.peers_on_simd(w))


class TestPortAvail(unittest.TestCase):
  def test_default_zero(self):
    a = SimdArbiter(16)
    for s in range(N_SIMDS):
      self.assertEqual(a.port_avail(s), 0)

  def test_setter_persists(self):
    a = SimdArbiter(16)
    a.set_port_avail(2, 1234)
    self.assertEqual(a.port_avail(2), 1234)
    # Other SIMDs untouched
    for s in (0, 1, 3):
      self.assertEqual(a.port_avail(s), 0)

  def test_port_avail_for_wave_lookup(self):
    a = SimdArbiter(16)
    a.set_port_avail(1, 42)
    # Wave 5 is on SIMD 1 (5 & 3 == 1)
    self.assertEqual(a.port_avail_for_wave(5), 42)
    # Wave 2 is on SIMD 2 — still zero
    self.assertEqual(a.port_avail_for_wave(2), 0)


class TestOwnerTracking(unittest.TestCase):
  def test_default_no_owner(self):
    a = SimdArbiter(16)
    for s in range(N_SIMDS):
      self.assertEqual(a.owner_wave(s), NO_OWNER)
      self.assertEqual(NO_OWNER, -1)

  def test_set_and_clear(self):
    a = SimdArbiter(16)
    a.set_owner_wave(0, 5)
    self.assertEqual(a.owner_wave(0), 5)
    a.clear_owner(0)
    self.assertEqual(a.owner_wave(0), NO_OWNER)


class TestLastIssueCycle(unittest.TestCase):
  def test_default_minus_one(self):
    a = SimdArbiter(16)
    for s in range(N_SIMDS):
      self.assertEqual(a.last_issue_cycle(s), -1)

  def test_setter(self):
    a = SimdArbiter(16)
    a.set_last_issue_cycle(3, 99)
    self.assertEqual(a.last_issue_cycle(3), 99)


class TestSnapshot(unittest.TestCase):
  def test_snapshot_initial(self):
    a = SimdArbiter(16)
    snap = a.snapshot()
    self.assertEqual(snap, {
      "port_avail": [0, 0, 0, 0],
      "owner_wave": [-1, -1, -1, -1],
      "last_issue_cycle": [-1, -1, -1, -1],
    })

  def test_snapshot_after_updates(self):
    a = SimdArbiter(16)
    a.set_port_avail(0, 10)
    a.set_port_avail(2, 20)
    a.set_owner_wave(0, 4)
    a.set_last_issue_cycle(2, 19)
    snap = a.snapshot()
    self.assertEqual(snap["port_avail"], [10, 0, 20, 0])
    self.assertEqual(snap["owner_wave"], [4, -1, -1, -1])
    self.assertEqual(snap["last_issue_cycle"], [-1, -1, 19, -1])

  def test_snapshot_is_copy_not_reference(self):
    # Callers mutating the snapshot MUST NOT affect arbiter state
    a = SimdArbiter(16)
    snap = a.snapshot()
    snap["port_avail"][0] = 999
    self.assertEqual(a.port_avail(0), 0)


if __name__ == "__main__":
  unittest.main()
