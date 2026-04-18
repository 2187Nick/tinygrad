"""Per-wave instruction-buffer / IB-fetch tracker.

Owns the small amount of state that tracks drain events (waitcnt / depctr /
s_nop) inside `_drain_zero_cost`. Extracted from `_simulate_sq_timing` per
EMU_REWRITE_DESIGN.md §1.8 / §5 Step 2. Pure refactor — zero behaviour change
at Step 2. `last_nop_before_valu_extra` is a stub that returns 0 here and will
be flipped to +4 in Step 7 (MISMATCH_ANALYSIS §B.B1 / PROBE_FINDINGS §B).
"""
from test.mockgpu.amd.sq_timing.constants import CONST, TimingConstants


class IbFetch:
  """Per-wave instruction-buffer / IB-fetch tracker.

  Owns: last_drain_stamp, had_drain_nop.
  Exposes: set_drain, reset_drain, mark_nop_in_chain, clear_nop_chain,
           resume_penalty, last_nop_before_valu_extra (stub, returns 0 for
           now — flipped in Step 7), on_non_nop_issue,
           last_drain_stamp (property), had_drain_nop (property).
  """
  def __init__(self, const: TimingConstants = CONST):
    self._const = const
    self._last_drain_stamp: int = -1
    self._had_drain_nop: bool = False

  @property
  def last_drain_stamp(self) -> int: return self._last_drain_stamp
  @property
  def had_drain_nop(self) -> bool: return self._had_drain_nop

  def set_drain(self, stamp: int) -> None:
    self._last_drain_stamp = stamp

  def reset_drain(self) -> None:
    self._last_drain_stamp = -1

  def mark_nop_in_chain(self) -> None:
    self._had_drain_nop = True

  def clear_nop_chain(self) -> None:
    self._had_drain_nop = False

  def resume_penalty(self) -> int:
    """+1 after a drain-nop chain; 0 otherwise (current behavior)."""
    return 1 if self._had_drain_nop else 0

  def last_nop_before_valu_extra(self) -> int:
    """Step 7 will land +4 here; Step 2 is a pure refactor → 0."""
    return 0

  def on_non_nop_issue(self) -> None:
    """Called after a non-nop instruction issues — clears the nop-chain flag."""
    self._had_drain_nop = False
