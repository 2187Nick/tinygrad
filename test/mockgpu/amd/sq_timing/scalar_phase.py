"""ScalarPhaseMachine — explicit state machine for SQ scalar-pipe phase tracking.

Replaces the scattered flags (`next_cmp_lit_phase_offset`, `in_phase_shifted_chain`,
`_vopd_paid_phase_warmup`) on SgprScoreboard and VAluPipe with a single cohesive
state machine. The machine observes per-instruction events (categorized from the
emu's existing extras tuple) and returns:
  - `current state` — introspectable for the caller
  - `extra_cycles(event)` — stall adjustment for this event given current state

States model the three orthogonal phase axes that HW cares about:
  - "Post-depctr" vs "Normal": does the depctr drain leave the scalar pipe in
    a shifted phase that affects the subsequent cmp→cndmask→VOPD chain?
  - Chain-type: are we in a v_cmp_lit writer chain, a v_cndmask reader chain,
    or a VOPD back-to-back chain?
  - Chain-position: how many consecutive instructions of the same chain-type
    have we seen (for position-dependent rules like "first cndmask pays A[n],
    subsequent pay 1cy").

Design non-goal for Step 1: wire into emu. This module is standalone, pure Python,
unit-testable without MOCKGPU. Step 3 wires it as a SHADOW alongside existing
flags; Steps 4-5 migrate each rule.

HW calibration source data (already captured):
  - exp_chain [5-80] — the canonical post-depctr phase-shift scenario
  - Batch C mb_c2_depctr_cmp{2,3,4}_cnd{2,3,4}_vopd — sweeps chain depth
  - Batch C mb_c4_depctr_chain_n{1,2,3,4}_vcc_first — cndmask taper
  - Batch B mb_vopd_chain_n4_{raw,no_raw} — VOPD chain behavior
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Event type classifier ────────────────────────────────────────────────────

class Event(Enum):
  """One event per instruction that the scalar-pipe machine observes.

  Coarse enough that the state machine stays small; fine enough to distinguish
  HW-different behaviors (CNDMASK_VCC vs CNDMASK_SGPR; VOPD vs VOPD_LIT; etc.).
  """
  DEPCTR = "depctr"                    # s_waitcnt_depctr — clears everything, enters post_depctr
  WAITCNT = "waitcnt"                  # s_waitcnt (vmcnt/lgkm/etc) — drain, resets chain state
  NOP = "nop"                          # s_nop — drain, resets chain state
  CMP_LIT_SGPR = "cmp_lit_sgpr"        # v_cmp_*_e64 with explicit SGPR write + literal source
  CMP_LIT_VCC = "cmp_lit_vcc"          # v_cmp_*_e32 with literal source (writes VCC implicitly)
  CNDMASK_VCC = "cndmask_vcc"          # v_cndmask reading VCC
  CNDMASK_SGPR = "cndmask_sgpr"        # v_cndmask reading explicit non-VCC SGPR (cmp-produced)
  VOPD = "vopd"                        # VOPD with VGPR reads (can be self-fwd or not)
  VOPD_LIT = "vopd_lit"                # VOPD_LIT with literal
  VOPD_MOV = "vopd_mov"                # VOPD V_DUAL_MOV on both lanes (no true VGPR reads)
  TRANS = "trans"                      # v_exp/v_log/v_rcp/v_sqrt/v_rsq
  VALU_OTHER = "valu_other"            # plain v_add/v_mul/v_mov — generic VALU
  SALU = "salu"                        # s_mov/s_cmp/s_and — any SALU
  BRANCH = "branch"                    # s_cbranch
  VMEM_READ = "vmem_rd"                # global_load
  VMEM_WRITE = "vmem_wr"               # global_store
  SMEM = "smem"                        # s_load_b*
  DS_READ = "ds_rd"                    # LDS read
  DS_WRITE = "ds_wr"                   # LDS write
  BARRIER = "barrier"                  # s_barrier
  OTHER = "other"                      # anything else


# ── State enum ───────────────────────────────────────────────────────────────

class Chain(Enum):
  """What chain we're currently building/consuming. Orthogonal to `phase_shifted`."""
  NONE = "none"                    # no active chain (idle, after a break)
  CMP_CHAIN = "cmp_chain"          # in a v_cmp_lit writer chain
  CNDMASK_CHAIN = "cndmask_chain"  # in a v_cndmask reader chain
  VOPD_CHAIN = "vopd_chain"        # consecutive VOPDs
  VOPD_TAIL = "vopd_tail"          # VOPD closing a cndmask chain (first, before chain)


class State(Enum):
  """Legacy enum kept for code that wants a single compact label. The authoritative
  state is `(phase_shifted, chain)` in ScalarPhaseMachine. Labels below map cleanly
  for introspection / logging / testing.
  """
  IDLE = "idle"
  POST_DEPCTR_READY = "post_depctr_ready"
  PHASE_CMP_CHAIN = "phase_cmp"
  PHASE_CNDMASK_CHAIN = "phase_cndmask"
  PHASE_VOPD_TAIL = "phase_vopd_tail"
  PHASE_VOPD_CHAIN = "phase_vopd_chain"   # VOPD chain in phase-shifted window (e.g. exp_chain [28-28])
  NORMAL_CMP_CHAIN = "normal_cmp"
  NORMAL_CNDMASK_CHAIN = "normal_cndmask"
  VOPD_CHAIN = "vopd_chain"
  POST_IDLE = "post_idle"


# ── Main class ───────────────────────────────────────────────────────────────

@dataclass
class ScalarPhaseMachine:
  """Per-wave phase state tracker.

  Authoritative state is the tuple `(phase_shifted, chain, chain_position)`:
    phase_shifted: bool — set True when depctr fires, preserved through the
      following VOPD/cmp/cndmask chain, cleared by any breaking instruction
      OR by a fresh drain event that doesn't start a phase-shifted chain.
    chain: Chain enum — what chain (if any) we're currently in.
    chain_position: int — 0-indexed position within the current chain.

  The `state` property exposes a compact label for introspection / tests.

  Usage:
    m = ScalarPhaseMachine()
    for event, info in events_for_this_wave:
      delta = m.advance(event, info)
      # emu adds `delta` to its base prediction for this inst.

  `info` is optional per-event extra (e.g. issue cycle, sgpr writes). For Step 1
  we keep it minimal; deltas are placeholders (return 0) until Step 4.
  """
  phase_shifted: bool = False
  chain: Chain = Chain.NONE
  chain_position: int = 0
  last_event: Optional[Event] = None
  # For debugging & unit tests — trace of (event, state_before, state_after, delta)
  trace: list = field(default_factory=list)

  @property
  def state(self) -> State:
    """Map the internal (phase_shifted, chain) to a compact State label."""
    if self.phase_shifted:
      if self.chain == Chain.NONE:          return State.POST_DEPCTR_READY
      if self.chain == Chain.CMP_CHAIN:     return State.PHASE_CMP_CHAIN
      if self.chain == Chain.CNDMASK_CHAIN: return State.PHASE_CNDMASK_CHAIN
      if self.chain == Chain.VOPD_TAIL:     return State.PHASE_VOPD_TAIL
      if self.chain == Chain.VOPD_CHAIN:    return State.PHASE_VOPD_CHAIN
    else:
      if self.chain == Chain.NONE:          return State.IDLE
      if self.chain == Chain.CMP_CHAIN:     return State.NORMAL_CMP_CHAIN
      if self.chain == Chain.CNDMASK_CHAIN: return State.NORMAL_CNDMASK_CHAIN
      if self.chain == Chain.VOPD_CHAIN:    return State.VOPD_CHAIN
      if self.chain == Chain.VOPD_TAIL:     return State.VOPD_CHAIN  # unreachable normally
    return State.IDLE

  def advance(self, event: Event, info: Optional[dict] = None) -> int:
    """Apply an event, update state, return cycle-delta suggestion (0 for Step 1)."""
    prev_state = self.state
    delta = 0
    new_phase, new_chain, new_pos = self._transition(event, info or {})
    self.phase_shifted = new_phase
    self.chain = new_chain
    self.chain_position = new_pos
    self.last_event = event
    self.trace.append((event, prev_state, self.state, delta))
    return delta

  def _transition(self, event: Event, info: dict) -> tuple[bool, Chain, int]:
    """Pure function: (state, event, info) → (new_phase_shifted, new_chain, new_position).

    Kept separate from `advance` so unit tests can assert transitions without
    invoking delta logic. Uses the orthogonal (phase_shifted, chain, pos) model.
    """
    ps, ch, pos = self.phase_shifted, self.chain, self.chain_position

    # ── Drain events ─────────────────────────────────────────────────────────
    if event == Event.DEPCTR:
      # depctr always enters phase-shifted mode. Any prior chain is dropped.
      return (True, Chain.NONE, 0)
    if event == Event.WAITCNT:
      # waitcnt drain — clears phase (different drain semantics than depctr).
      return (False, Chain.NONE, 0)
    if event == Event.NOP:
      return (False, Chain.NONE, 0)

    # ── Chain builders: CMP_LIT ──────────────────────────────────────────────
    if event in (Event.CMP_LIT_SGPR, Event.CMP_LIT_VCC):
      # Phase preserved if we're entering from POST_DEPCTR or continuing the chain.
      if ch == Chain.CMP_CHAIN:
        return (ps, Chain.CMP_CHAIN, pos + 1)
      # Entering a new cmp chain — inherit phase_shifted from current state.
      return (ps, Chain.CMP_CHAIN, 0)

    # ── Chain consumers: CNDMASK ─────────────────────────────────────────────
    if event in (Event.CNDMASK_VCC, Event.CNDMASK_SGPR):
      if ch == Chain.CNDMASK_CHAIN:
        return (ps, Chain.CNDMASK_CHAIN, pos + 1)
      # Transition from cmp chain or other state into cndmask chain.
      # Phase preserved if we were in a phase-shifted cmp chain.
      return (ps, Chain.CNDMASK_CHAIN, 0)

    # ── VOPD handling ────────────────────────────────────────────────────────
    if event in (Event.VOPD, Event.VOPD_LIT, Event.VOPD_MOV):
      # VOPD closing a cndmask chain = VOPD_TAIL (pays the +2cy warmup if phase-shifted).
      if ch == Chain.CNDMASK_CHAIN:
        return (ps, Chain.VOPD_TAIL, 0)
      # Subsequent VOPDs after the tail, or VOPDs chaining directly: VOPD_CHAIN.
      if ch in (Chain.VOPD_TAIL, Chain.VOPD_CHAIN):
        return (ps, Chain.VOPD_CHAIN, pos + 1)
      # First VOPD from IDLE / POST_DEPCTR / CMP_CHAIN etc. — phase preserved.
      return (ps, Chain.VOPD_CHAIN, 0)

    # ── Trans ops ────────────────────────────────────────────────────────────
    if event == Event.TRANS:
      # Trans ops break cmp/cndmask chains. They don't necessarily clear phase
      # (exp_chain [6-7] trans ops between VOPD chain segments).
      # Conservative: break the chain, keep phase for now. Refine in Step 4.
      return (ps, Chain.NONE, 0)

    # ── All other events break chains and clear phase ────────────────────────
    # (salu, valu_other, branch, vmem, smem, ds, barrier)
    return (False, Chain.NONE, 0)


# ── Event classifier ─────────────────────────────────────────────────────────

def classify_event(cat: str, *, is_vopd: bool = False, is_vopd_lit: bool = False,
                   is_vopd_mov: bool = False, is_cmp_lit: bool = False,
                   sgpr_writes: tuple = (), sgpr_reads: tuple = (),
                   cond_sgpr: int = -1, trans_name: str = "",
                   nop_cycles: Optional[int] = None) -> Event:
  """Map the emu's categorized instruction info to a single `Event`.

  `cat` is the emu's cat string ('valu', 'salu', 'ds_rd', ...). The other args
  come from the extras tuple that `_simulate_sq_timing` already destructures.
  """
  if cat == 'depctr': return Event.DEPCTR
  if cat == 'waitcnt': return Event.WAITCNT
  if cat == 'nop': return Event.NOP
  if cat == 'barrier': return Event.BARRIER
  if cat == 'branch': return Event.BRANCH
  if cat == 'smem': return Event.SMEM
  if cat == 'ds_rd': return Event.DS_READ
  if cat == 'ds_wr': return Event.DS_WRITE
  if cat == 'vmem_rd': return Event.VMEM_READ
  if cat == 'vmem_wr': return Event.VMEM_WRITE
  if cat == 'salu': return Event.SALU
  if cat != 'valu': return Event.OTHER
  # valu cases
  if is_vopd_mov:    return Event.VOPD_MOV
  if is_vopd_lit:    return Event.VOPD_LIT
  if is_vopd:        return Event.VOPD
  if trans_name:     return Event.TRANS
  # v_cndmask detection: reads VCC (sgpr 106) or has cond_sgpr set
  _is_cndmask = (cond_sgpr >= 0) or (sgpr_reads and 106 in sgpr_reads)
  if _is_cndmask:
    return Event.CNDMASK_VCC if (cond_sgpr < 0 or cond_sgpr == 106) else Event.CNDMASK_SGPR
  # cmp_lit detection
  if is_cmp_lit:
    # Non-VCC explicit SGPR write → CMP_LIT_SGPR; implicit VCC-only → CMP_LIT_VCC.
    _nonvcc_writes = [r for r in sgpr_writes if r != 106]
    return Event.CMP_LIT_SGPR if _nonvcc_writes else Event.CMP_LIT_VCC
  return Event.VALU_OTHER
