"""Per-wave scalar ALU + branch state holder.

Extracted from `_simulate_sq_timing` per EMU_REWRITE_DESIGN.md §1.4 / §5 Step 6.
Pure refactor — zero behaviour change. Owns the per-wave SCC-write and
EXEC-write cycle trackers used by s_cbranch_scc0/scc1 and
s_cbranch_execz/nz dependency enforcement.

Scope note: this class is a simple state-holder (getters, setters) following
the Step 4 VmemPipe pattern. SALU cost, s_cbranch cost, and s_nop cost
functions stay in emu.py for this step — Step 7 will land the HW-confirmed
constant fixes (CBRANCH_TIGHT/COLD, NOP_AFTER_VMCNT_DRAIN, etc.) and may
then refactor those cost helpers onto ScalarPipe.

Intentionally out of scope for Step 6:
  - `at_barrier`, `barrier_issue`, `wave_done`, `pc`, `ready` — WaveScheduler.
  - `last_drain_stamp`, `had_drain_nop` — IbFetch (Step 2).
  - `scalar_after_trans_ready` — TransPipe (Step 5a).
"""
from test.mockgpu.amd.sq_timing.constants import CONST, TimingConstants


class ScalarPipe:
  """Per-wave scalar ALU + branch state holder.

  Owns:
    - `scc_write_time`: last cycle when SCC was written (by s_cmp/s_cmpk/SALU).
      Read by s_cbranch_scc0/scc1 issue (+SCC_READ_LATENCY).
    - `exec_write_time`: last cycle when EXEC was written (by v_cmpx, and also
      stamped on any SALU issue to match HW behaviour). Read by
      s_cbranch_execz/nz issue (+EXEC_WRITE_LATENCY).
  """
  def __init__(self, const: TimingConstants = CONST):
    self._c = const
    self._scc_write_time: int = 0
    self._exec_write_time: int = 0
    self._first_branch_after_drain: bool = False

  # ── Read-only accessors ────────────────────────────────────────────────────
  @property
  def scc_write_time(self) -> int: return self._scc_write_time
  @property
  def exec_write_time(self) -> int: return self._exec_write_time
  @property
  def first_branch_after_drain(self) -> bool: return self._first_branch_after_drain

  # ── Mutations routed through methods so future steps can swap logic ────────
  def set_scc_write_time(self, cycle: int) -> None: self._scc_write_time = cycle
  def set_exec_write_time(self, cycle: int) -> None: self._exec_write_time = cycle
  def mark_drain(self) -> None: self._first_branch_after_drain = True
  def consume_drain_branch(self) -> None: self._first_branch_after_drain = False
