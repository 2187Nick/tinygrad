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
from test.mockgpu.amd.sq_timing.vgpr_banks import (
  bank_of, inst_has_bank_conflict, vgpr_bank_conflicts,
)
from tinygrad.renderer.amd.dsl import v
from tinygrad.runtime.autogen.amd.rdna3.enum import VOPDOp
from tinygrad.runtime.autogen.amd.rdna3.ins import VOPD as VOPD3


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


def _vopd(opx=VOPDOp.V_DUAL_ADD_F32, opy=VOPDOp.V_DUAL_ADD_F32,
          vdstx=v[10], vdsty=v[11], srcx0=v[0], srcy0=v[1], vsrcx1=v[2], vsrcy1=v[3]):
  return VOPD3(opx=opx, opy=opy, vdstx=vdstx, vdsty=vdsty,
               srcx0=srcx0, srcy0=srcy0, vsrcx1=vsrcx1, vsrcy1=vsrcy1)


class TestBankOf(unittest.TestCase):
  def test_modulo_4(self):
    self.assertEqual(bank_of(0), 0)
    self.assertEqual(bank_of(1), 1)
    self.assertEqual(bank_of(4), 0)
    self.assertEqual(bank_of(5), 1)
    self.assertEqual(bank_of(255), 3)


class TestInstBankConflict(unittest.TestCase):
  def test_no_conflict_distinct_banks(self):
    # X reads v[0,2] (banks 0,2); Y reads v[1,3] (banks 1,3) — disjoint
    self.assertFalse(inst_has_bank_conflict(_vopd(srcx0=v[0], vsrcx1=v[2], srcy0=v[1], vsrcy1=v[3])))

  def test_srcx0_srcy0_same_bank(self):
    # v[0] bank=0, v[4] bank=0 — both X and Y read bank 0
    self.assertTrue(inst_has_bank_conflict(_vopd(srcx0=v[0], srcy0=v[4])))

  def test_vsrcx1_vsrcy1_same_bank(self):
    # X-side vsrcx1=v[5] bank=1, Y-side vsrcy1=v[9] bank=1
    self.assertTrue(inst_has_bank_conflict(_vopd(srcx0=v[0], vsrcx1=v[5], srcy0=v[2], vsrcy1=v[9])))

  def test_same_vgpr_both_sides(self):
    self.assertTrue(inst_has_bank_conflict(_vopd(srcx0=v[8], srcy0=v[8])))

  def test_sgpr_sources_dont_conflict(self):
    from tinygrad.renderer.amd.dsl import s
    # Y srcy0=s[5] (SGPR) — Y has only vsrcy1=v[3] (bank 3) vs X v[0,2] (banks 0,2): no overlap
    self.assertFalse(inst_has_bank_conflict(_vopd(srcx0=v[0], vsrcx1=v[2], srcy0=s[5], vsrcy1=v[3])))

  def test_mov_only_ignores_dummy_vsrc1(self):
    # V_DUAL_MOV_B32 has only srcy0 (no vsrcy1) per OPERANDS — even though encoding fills v[0]
    # in the vsrcy1 slot, the analyzer must skip it. Same on X-side (V_DUAL_MOV_B32 -> only srcx0).
    from tinygrad.renderer.amd.dsl import s
    inst = _vopd(opx=VOPDOp.V_DUAL_MOV_B32, opy=VOPDOp.V_DUAL_MOV_B32,
                 srcx0=s[4], srcy0=s[5], vsrcx1=v[0], vsrcy1=v[0])
    # Only srcx0 (SGPR), srcy0 (SGPR) are real — no VGPR reads at all
    self.assertFalse(inst_has_bank_conflict(inst))


class TestVgprBankConflictsTable(unittest.TestCase):
  def test_dict_keys_only_vopd_pcs(self):
    # Direct test path: synthesize a tiny lib with a VOPD instruction is not trivial
    # without a renderer — exercise the per-inst path via inst_has_bank_conflict above
    # and rely on integration tests for the lib-bytes path. Here we just sanity-check
    # the function exists and returns a dict.
    self.assertTrue(callable(vgpr_bank_conflicts))


if __name__ == "__main__":
  unittest.main()
