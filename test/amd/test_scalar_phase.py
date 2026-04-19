"""Unit tests for ScalarPhaseMachine — exercises the transition table in
test/mockgpu/amd/sq_timing/scalar_phase.py without needing MOCKGPU.

Each test sends a deterministic sequence of events and asserts the resulting
`(phase_shifted, chain, chain_position)` state. Delta suggestions are placeholder
(0) in Step 1 of the refactor; this file locks in the transition contract so
that wiring the machine into emu.py later cannot silently regress it.

Mapped to HW scenarios:
  - DEPCTR → phase_shifted=True (see exp_chain [8], [27], [48])
  - cmp_lit after DEPCTR inherits phase (exp_chain [9-12])
  - cndmask consuming phase-shifted cmp chain stays in phase (exp_chain [13-15])
  - VOPD closing cndmask chain under phase = VOPD_TAIL (exp_chain [37], [61])
  - TRANS breaks chain (and per 2026-04-19 analysis should clear phase; currently
    tests assert the CURRENT behavior then we'll tighten once fix lands)
  - WAITCNT/NOP drain clears phase
"""
import unittest
from test.mockgpu.amd.sq_timing.scalar_phase import ScalarPhaseMachine, Event, Chain, State, classify_event


class TestScalarPhaseTransitions(unittest.TestCase):
  def test_initial_state_is_idle(self):
    m = ScalarPhaseMachine()
    self.assertEqual(m.state, State.IDLE)
    self.assertFalse(m.phase_shifted)
    self.assertEqual(m.chain, Chain.NONE)
    self.assertEqual(m.chain_position, 0)

  def test_depctr_enters_post_depctr_ready(self):
    m = ScalarPhaseMachine()
    m.advance(Event.DEPCTR)
    self.assertTrue(m.phase_shifted)
    self.assertEqual(m.chain, Chain.NONE)
    self.assertEqual(m.state, State.POST_DEPCTR_READY)

  def test_waitcnt_clears_phase(self):
    m = ScalarPhaseMachine()
    m.advance(Event.DEPCTR)       # phase on
    m.advance(Event.WAITCNT)      # should clear
    self.assertFalse(m.phase_shifted)
    self.assertEqual(m.state, State.IDLE)

  def test_nop_clears_phase(self):
    m = ScalarPhaseMachine()
    m.advance(Event.DEPCTR)
    m.advance(Event.NOP)
    self.assertFalse(m.phase_shifted)
    self.assertEqual(m.state, State.IDLE)

  # ── cmp_lit chains ──────────────────────────────────────────────────────────

  def test_cmp_chain_preserves_phase_from_depctr(self):
    """exp_chain [8-11]: depctr → cmp_lit chain stays phase-shifted."""
    m = ScalarPhaseMachine()
    m.advance(Event.DEPCTR)
    m.advance(Event.CMP_LIT_VCC)
    self.assertTrue(m.phase_shifted)
    self.assertEqual(m.chain, Chain.CMP_CHAIN)
    self.assertEqual(m.chain_position, 0)
    self.assertEqual(m.state, State.PHASE_CMP_CHAIN)
    m.advance(Event.CMP_LIT_SGPR)
    self.assertEqual(m.chain_position, 1)

  def test_cmp_chain_without_prior_depctr_is_normal(self):
    m = ScalarPhaseMachine()
    m.advance(Event.CMP_LIT_VCC)
    self.assertFalse(m.phase_shifted)
    self.assertEqual(m.state, State.NORMAL_CMP_CHAIN)

  def test_cmp_chain_position_increments(self):
    m = ScalarPhaseMachine()
    m.advance(Event.DEPCTR)
    for i in range(4):
      m.advance(Event.CMP_LIT_SGPR)
      self.assertEqual(m.chain_position, i)

  # ── cndmask chains ──────────────────────────────────────────────────────────

  def test_cmp_to_cndmask_transition_preserves_phase(self):
    """exp_chain [12→13]: cmp chain flips to cndmask chain, phase preserved."""
    m = ScalarPhaseMachine()
    m.advance(Event.DEPCTR)
    m.advance(Event.CMP_LIT_VCC)
    m.advance(Event.CMP_LIT_SGPR)
    # transition into cndmask chain
    m.advance(Event.CNDMASK_VCC)
    self.assertTrue(m.phase_shifted)
    self.assertEqual(m.chain, Chain.CNDMASK_CHAIN)
    self.assertEqual(m.chain_position, 0)

  def test_cndmask_chain_position_increments(self):
    m = ScalarPhaseMachine()
    m.advance(Event.DEPCTR)
    m.advance(Event.CMP_LIT_VCC)
    for i in range(4):
      m.advance(Event.CNDMASK_SGPR)
      self.assertEqual(m.chain_position, i)

  # ── VOPD tail & chain ───────────────────────────────────────────────────────

  def test_vopd_closing_cndmask_chain_is_tail(self):
    """exp_chain [37]: VOPD after cndmask chain = VOPD_TAIL, phase preserved."""
    m = ScalarPhaseMachine()
    m.advance(Event.DEPCTR)
    m.advance(Event.CMP_LIT_VCC)
    m.advance(Event.CNDMASK_VCC)
    m.advance(Event.CNDMASK_SGPR)
    m.advance(Event.VOPD)
    self.assertTrue(m.phase_shifted)
    self.assertEqual(m.chain, Chain.VOPD_TAIL)
    self.assertEqual(m.state, State.PHASE_VOPD_TAIL)

  def test_vopd_after_tail_becomes_chain(self):
    """Second VOPD after a tail: VOPD_CHAIN with pos=1."""
    m = ScalarPhaseMachine()
    m.advance(Event.DEPCTR)
    m.advance(Event.CMP_LIT_VCC)
    m.advance(Event.CNDMASK_VCC)
    m.advance(Event.VOPD)        # tail
    m.advance(Event.VOPD)        # chain
    self.assertEqual(m.chain, Chain.VOPD_CHAIN)
    self.assertEqual(m.chain_position, 1)

  def test_vopd_from_idle_is_chain(self):
    m = ScalarPhaseMachine()
    m.advance(Event.VOPD)
    self.assertEqual(m.chain, Chain.VOPD_CHAIN)
    self.assertEqual(m.chain_position, 0)
    self.assertFalse(m.phase_shifted)

  def test_vopd_lit_same_treatment_as_vopd(self):
    m = ScalarPhaseMachine()
    m.advance(Event.DEPCTR)
    m.advance(Event.CNDMASK_VCC)
    m.advance(Event.VOPD_LIT)
    self.assertEqual(m.chain, Chain.VOPD_TAIL)

  def test_vopd_mov_same_treatment_as_vopd(self):
    m = ScalarPhaseMachine()
    m.advance(Event.VOPD_MOV)
    self.assertEqual(m.chain, Chain.VOPD_CHAIN)

  # ── TRANS handling ──────────────────────────────────────────────────────────

  def test_trans_breaks_chain(self):
    m = ScalarPhaseMachine()
    m.advance(Event.DEPCTR)
    m.advance(Event.CMP_LIT_VCC)
    m.advance(Event.CNDMASK_VCC)
    m.advance(Event.TRANS)
    self.assertEqual(m.chain, Chain.NONE)

  def test_four_trans_after_chain(self):
    """exp_chain [19-22]: 4× v_exp between two cndmask groups. Chain broken."""
    m = ScalarPhaseMachine()
    m.advance(Event.DEPCTR)
    m.advance(Event.CMP_LIT_VCC)
    for _ in range(4):
      m.advance(Event.TRANS)
    self.assertEqual(m.chain, Chain.NONE)

  # ── Break events ────────────────────────────────────────────────────────────

  def test_valu_other_breaks_chain_and_clears_phase(self):
    """Per _transition's catch-all: plain VALU clears phase+chain."""
    m = ScalarPhaseMachine()
    m.advance(Event.DEPCTR)
    m.advance(Event.CMP_LIT_VCC)
    m.advance(Event.VALU_OTHER)
    self.assertFalse(m.phase_shifted)
    self.assertEqual(m.chain, Chain.NONE)

  def test_vmem_breaks_chain(self):
    m = ScalarPhaseMachine()
    m.advance(Event.DEPCTR)
    m.advance(Event.CMP_LIT_VCC)
    m.advance(Event.VMEM_WRITE)
    self.assertEqual(m.chain, Chain.NONE)

  # ── Full exp_chain trace (canonical scenario) ───────────────────────────────

  def test_exp_chain_canonical_trace(self):
    """Walks an exp_chain-like sequence and verifies the state at each key point."""
    m = ScalarPhaseMachine()
    # depctr
    m.advance(Event.DEPCTR);                                self.assertEqual(m.state, State.POST_DEPCTR_READY)
    # cmp_lit chain
    m.advance(Event.CMP_LIT_VCC);                           self.assertEqual(m.state, State.PHASE_CMP_CHAIN)
    m.advance(Event.CMP_LIT_SGPR);                          self.assertEqual(m.state, State.PHASE_CMP_CHAIN)
    m.advance(Event.CMP_LIT_SGPR);                          self.assertEqual(m.state, State.PHASE_CMP_CHAIN)
    # cndmask chain
    m.advance(Event.CNDMASK_VCC);                           self.assertEqual(m.state, State.PHASE_CNDMASK_CHAIN)
    m.advance(Event.CNDMASK_SGPR);                          self.assertEqual(m.state, State.PHASE_CNDMASK_CHAIN)
    # VOPD tail
    m.advance(Event.VOPD);                                  self.assertEqual(m.state, State.PHASE_VOPD_TAIL)
    m.advance(Event.VOPD);                                  self.assertEqual(m.state, State.PHASE_VOPD_CHAIN)
    # trans breaks
    m.advance(Event.TRANS);                                 self.assertEqual(m.chain, Chain.NONE)


class TestEventClassifier(unittest.TestCase):
  def test_depctr_categories(self):
    self.assertEqual(classify_event('depctr'), Event.DEPCTR)
    self.assertEqual(classify_event('waitcnt'), Event.WAITCNT)
    self.assertEqual(classify_event('nop'), Event.NOP)
    self.assertEqual(classify_event('barrier'), Event.BARRIER)
    self.assertEqual(classify_event('branch'), Event.BRANCH)

  def test_memory_categories(self):
    self.assertEqual(classify_event('smem'), Event.SMEM)
    self.assertEqual(classify_event('vmem_rd'), Event.VMEM_READ)
    self.assertEqual(classify_event('vmem_wr'), Event.VMEM_WRITE)
    self.assertEqual(classify_event('ds_rd'), Event.DS_READ)
    self.assertEqual(classify_event('ds_wr'), Event.DS_WRITE)

  def test_salu_and_other(self):
    self.assertEqual(classify_event('salu'), Event.SALU)
    self.assertEqual(classify_event('bogus'), Event.OTHER)

  def test_vopd_flavors(self):
    self.assertEqual(classify_event('valu', is_vopd=True), Event.VOPD)
    self.assertEqual(classify_event('valu', is_vopd=True, is_vopd_lit=True), Event.VOPD_LIT)
    self.assertEqual(classify_event('valu', is_vopd=True, is_vopd_mov=True), Event.VOPD_MOV)

  def test_trans_from_trans_name(self):
    self.assertEqual(classify_event('valu', trans_name='v_exp'), Event.TRANS)
    self.assertEqual(classify_event('valu', trans_name='v_log'), Event.TRANS)

  def test_cndmask_vcc_vs_sgpr(self):
    # cndmask with cond_sgpr=106 (VCC) → CNDMASK_VCC
    self.assertEqual(classify_event('valu', cond_sgpr=106), Event.CNDMASK_VCC)
    # cndmask with cond_sgpr=0 (non-VCC) → CNDMASK_SGPR
    self.assertEqual(classify_event('valu', cond_sgpr=0), Event.CNDMASK_SGPR)
    # cndmask detected via sgpr_reads containing 106
    self.assertEqual(classify_event('valu', sgpr_reads=(106,)), Event.CNDMASK_VCC)

  def test_cmp_lit_vcc_vs_sgpr(self):
    # cmp_lit writing only VCC (implicit) → CMP_LIT_VCC
    self.assertEqual(classify_event('valu', is_cmp_lit=True, sgpr_writes=(106,)), Event.CMP_LIT_VCC)
    self.assertEqual(classify_event('valu', is_cmp_lit=True, sgpr_writes=()), Event.CMP_LIT_VCC)
    # cmp_lit writing explicit SGPR → CMP_LIT_SGPR
    self.assertEqual(classify_event('valu', is_cmp_lit=True, sgpr_writes=(4,)), Event.CMP_LIT_SGPR)

  def test_valu_other_fallthrough(self):
    self.assertEqual(classify_event('valu'), Event.VALU_OTHER)


class TestTraceLogging(unittest.TestCase):
  def test_trace_captures_events_and_states(self):
    m = ScalarPhaseMachine()
    m.advance(Event.DEPCTR)
    m.advance(Event.CMP_LIT_VCC)
    self.assertEqual(len(m.trace), 2)
    ev0, prev0, cur0, d0 = m.trace[0]
    self.assertEqual(ev0, Event.DEPCTR)
    self.assertEqual(prev0, State.IDLE)
    self.assertEqual(cur0, State.POST_DEPCTR_READY)
    self.assertEqual(d0, 0)  # placeholder in Step 1


if __name__ == '__main__':
  unittest.main()
