"""Per-wave VMEM pipeline tracker.

Extracted from `_simulate_sq_timing` per EMU_REWRITE_DESIGN.md §1.6 / §5 Step 4.
Pure refactor — zero behaviour change. Owns the per-wave VMEM forwarding
deadlines, the VMEM drain deadline, and the `vm_pend` queue used by
s_waitcnt_vmcnt drain.

Scope note: `_vmem_wr_bypass_active` (the inter-wave 17/21 bypass check) reads
peer-wave VmemPipe instances (via the per-wave `vmem` list) but stays in emu.py
for this step. Moving it into VmemPipe requires a `PeerSnapshot` type that
also exposes pc/at_barrier/wave_done/ready — that peer-state machinery is a
Step 5+ concern per the design doc (§4.3).

This class is a simple state-holder: getters, setters, capped-setters, and a
prune() helper. Later steps can refactor logic INTO the class.
"""
from test.mockgpu.amd.sq_timing.constants import CONST, TimingConstants


class VmemPipe:
  """Per-wave VMEM pipeline tracker.

  Owns:
    - `valu_vmem_wr_deadline`: VALU→VMEM_WR forwarding deadline.
    - `valu_vmem_wr_set_time`: issue cycle of the VALU that set the VMEM_WR
      deadline (used by `_vmem_wr_bypass_active` in emu.py).
    - `valu_vmem_wr_slow_ext`: slow-fresh extension baked into VMEM_WR deadline
      (used to shrink VMEM drain when forwarding stalled).
    - `valu_vmem_rd_deadline`: VALU→VMEM_RD forwarding deadline.
    - `vmem_drain_deadline`: VMEM pipeline drain — blocks s_nop/s_endpgm until
      the VMEM unit has accepted the op.
    - `vm_pend`: pending VMEM completion cycles for s_waitcnt_vmcnt drain.
  """
  def __init__(self, const: TimingConstants = CONST):
    self._c = const
    self._wr_deadline = 0
    self._wr_set_time = 0
    self._wr_slow_ext = 0
    self._rd_deadline = 0
    self._drain_deadline = 0
    self._vm_pend: list[int] = []

  # ── Simple accessors for read-only queries from emu.py ─────────────────────
  @property
  def wr_deadline(self) -> int: return self._wr_deadline
  @property
  def wr_set_time(self) -> int: return self._wr_set_time
  @property
  def wr_slow_ext(self) -> int: return self._wr_slow_ext
  @property
  def rd_deadline(self) -> int: return self._rd_deadline
  @property
  def drain_deadline(self) -> int: return self._drain_deadline

  def vm_pending(self) -> list[int]:
    """Return the raw pending VMEM completion list (caller may sort/index)."""
    return self._vm_pend

  # ── Mutations routed through methods so future steps can swap logic ────────
  def set_wr_deadline(self, cycle: int) -> None: self._wr_deadline = cycle
  def set_wr_set_time(self, cycle: int) -> None: self._wr_set_time = cycle
  def set_wr_slow_ext(self, v: int) -> None: self._wr_slow_ext = v
  def set_rd_deadline(self, cycle: int) -> None: self._rd_deadline = cycle
  def set_drain_deadline(self, cycle: int) -> None: self._drain_deadline = cycle

  def cap_wr_deadline(self, cycle: int) -> None:
    """Clamp wr_deadline to min(current, cycle) — used by s_nop residual cap."""
    self._wr_deadline = min(self._wr_deadline, cycle)

  def cap_rd_deadline(self, cycle: int) -> None:
    """Clamp rd_deadline to min(current, cycle) — used by s_nop residual cap."""
    self._rd_deadline = min(self._rd_deadline, cycle)

  def on_vmem_issue(self, completion_cycle: int) -> None:
    """Append a VMEM completion cycle to the wave's vm_pend queue."""
    self._vm_pend.append(completion_cycle)

  def prune(self, stall_until: int) -> None:
    """Drop completions ≤ stall_until after s_waitcnt_vmcnt resolves."""
    self._vm_pend = [c for c in self._vm_pend if c > stall_until]
