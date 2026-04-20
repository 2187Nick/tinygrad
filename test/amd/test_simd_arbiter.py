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
  def test_even_odd_wgp_split(self):
    # HW_ID probe 2026-04-20 (16-wave launches, 2600+ obs): every compute wave
    # has SIMD_ID=0, but 16-wave workgroups split across TWO WGPs in an
    # even/odd pattern. simd_for_wave() returns wave_idx & 0x1 as a peer-group
    # proxy — even waves on one WGP's port, odd waves on the other.
    for w in range(64):
      self.assertEqual(SimdArbiter.simd_for_wave(w), w & 0x1)

  def test_n_simds_constant(self):
    self.assertEqual(N_SIMDS, 2)


class TestPeersOnSimd(unittest.TestCase):
  def test_four_waves_even_odd_split(self):
    a = SimdArbiter(4)
    # Even waves (0, 2) share one peer group; odd waves (1, 3) share the other
    self.assertEqual(a.peers_on_simd(0), [2])
    self.assertEqual(a.peers_on_simd(2), [0])
    self.assertEqual(a.peers_on_simd(1), [3])
    self.assertEqual(a.peers_on_simd(3), [1])

  def test_sixteen_waves_even_odd_split(self):
    a = SimdArbiter(16)
    # Wave 0 (even) peers = all other even waves {2, 4, 6, 8, 10, 12, 14}
    self.assertEqual(a.peers_on_simd(0), [2, 4, 6, 8, 10, 12, 14])
    # Wave 5 (odd) peers = all other odd waves {1, 3, 7, 9, 11, 13, 15}
    self.assertEqual(a.peers_on_simd(5), [1, 3, 7, 9, 11, 13, 15])
    self.assertEqual(a.peers_on_simd(15), [1, 3, 5, 7, 9, 11, 13])

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
    a.set_port_avail(1, 1234)
    self.assertEqual(a.port_avail(1), 1234)
    # Other SIMD untouched
    self.assertEqual(a.port_avail(0), 0)

  def test_port_avail_for_wave_lookup(self):
    a = SimdArbiter(16)
    a.set_port_avail(0, 42)
    a.set_port_avail(1, 99)
    # Even waves -> SIMD 0, odd waves -> SIMD 1
    self.assertEqual(a.port_avail_for_wave(4), 42)
    self.assertEqual(a.port_avail_for_wave(5), 99)


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
    a.set_last_issue_cycle(1, 99)
    self.assertEqual(a.last_issue_cycle(1), 99)


class TestSnapshot(unittest.TestCase):
  def test_snapshot_initial(self):
    a = SimdArbiter(16)
    snap = a.snapshot()
    self.assertEqual(snap, {
      "port_avail": [0, 0],
      "owner_wave": [-1, -1],
      "last_issue_cycle": [-1, -1],
    })

  def test_snapshot_after_updates(self):
    a = SimdArbiter(16)
    a.set_port_avail(0, 10)
    a.set_port_avail(1, 20)
    a.set_owner_wave(0, 4)
    a.set_last_issue_cycle(1, 19)
    snap = a.snapshot()
    self.assertEqual(snap["port_avail"], [10, 20])
    self.assertEqual(snap["owner_wave"], [4, -1])
    self.assertEqual(snap["last_issue_cycle"], [-1, 19])

  def test_snapshot_is_copy_not_reference(self):
    # Callers mutating the snapshot MUST NOT affect arbiter state
    a = SimdArbiter(16)
    snap = a.snapshot()
    snap["port_avail"][0] = 999
    self.assertEqual(a.port_avail(0), 0)


if __name__ == "__main__":
  unittest.main()
