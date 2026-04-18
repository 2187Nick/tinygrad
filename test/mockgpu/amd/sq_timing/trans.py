"""Per-wave trans ALU pipeline tracker.

Extracted from `_simulate_sq_timing` per EMU_REWRITE_DESIGN.md §1.3 / §5 Step 5a.
Pure refactor — zero behaviour change. Owns the per-wave trans pipeline
availability, trans-written VGPR readiness map, scalar-after-trans visibility
cycle, and the pending-trans completion queue used by s_waitcnt_depctr.

Scope note: this class is a simple state-holder (getters, setters, prune)
following the Step 4 VmemPipe pattern. Logic stays in emu.py for now — later
steps may refactor methods like `trans_read_stall` / `on_trans_issue` in.
"""
from test.mockgpu.amd.sq_timing.constants import CONST, TimingConstants


class TransPipe:
  """Per-wave trans ALU pipeline tracker.

  Owns:
    - `trans_pipe_avail`: next cycle the trans unit is free for a trans op.
    - `trans_vgpr_ready[reg]`: ready cycle for trans-written VGPRs (only
      non-trans consumers wait for full writeback; trans→trans forwards via
      the pipeline).
    - `scalar_after_trans_ready`: cycle at which scalar path (waitcnt, s_nop)
      can observe trans completion (trans_pipe_avail - 1).
    - `valu_pend`: pending multi-cycle VALU (trans) completion cycles for
      s_waitcnt_depctr drain.
  """
  def __init__(self, const: TimingConstants = CONST):
    self._c = const
    self._pipe_avail = 0
    self._scalar_ready = 0
    self._vgpr_ready: dict[int, int] = {}
    self._valu_pend: list[int] = []

  # ── Read-only accessors ────────────────────────────────────────────────────
  @property
  def pipe_avail(self) -> int: return self._pipe_avail
  @property
  def scalar_ready(self) -> int: return self._scalar_ready

  def vgpr_ready_map(self) -> dict[int, int]:
    """Return the trans-written VGPR readiness dict (caller may read/pop)."""
    return self._vgpr_ready

  def valu_pending(self) -> list[int]:
    """Return the raw pending trans-completion list (caller may sort/index)."""
    return self._valu_pend

  # ── Mutations routed through methods so future steps can swap logic ────────
  def set_pipe_avail(self, cycle: int) -> None: self._pipe_avail = cycle
  def set_scalar_ready(self, cycle: int) -> None: self._scalar_ready = cycle

  def set_vgpr_ready(self, reg: int, cycle: int) -> None: self._vgpr_ready[reg] = cycle
  def pop_vgpr_ready(self, reg: int) -> None: self._vgpr_ready.pop(reg, None)

  def on_trans_issue(self, completion_cycle: int) -> None:
    """Append a trans-completion cycle to the wave's valu_pend queue."""
    self._valu_pend.append(completion_cycle)

  def prune_valu_pend(self, stall_until: int) -> None:
    """Drop trans completions ≤ stall_until after s_waitcnt_depctr resolves."""
    self._valu_pend = [c for c in self._valu_pend if c > stall_until]
