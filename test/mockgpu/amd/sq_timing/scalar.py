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
    self._last_salu_issue: int = 0
    # Dedicated SALU-only write-time map (separate from sgpr[i].write_time which models
    # the VALU-visible 4cy SGPR latency). SALU→SALU RAW-chain uses this shorter map.
    self._salu_write_time: dict[int, int] = {}
    # Consecutive SALU RAW-chain length. Incremented when a SALU reads an sdst written
    # by the immediately-previous SALU; reset otherwise. Used to gate the wave-credit
    # rule so short chains (n≤5, HW mb_g4_s_{or,and,xor,bfe}_n4 all-waves dt=1) stay
    # fast while long chains (n≥6, HW mb_g4_s_add_u32_n8 wave 1+ stalls at pos 6+).
    self._salu_raw_chain_depth: int = 0

  # ── Read-only accessors ────────────────────────────────────────────────────
  @property
  def scc_write_time(self) -> int: return self._scc_write_time
  @property
  def exec_write_time(self) -> int: return self._exec_write_time
  @property
  def first_branch_after_drain(self) -> bool: return self._first_branch_after_drain
  @property
  def last_salu_issue(self) -> int: return self._last_salu_issue

  def salu_write_time_map(self) -> dict[int, int]: return self._salu_write_time
  def set_salu_write_time(self, reg: int, cycle: int) -> None: self._salu_write_time[reg] = cycle
  @property
  def salu_raw_chain_depth(self) -> int: return self._salu_raw_chain_depth
  def inc_salu_raw_chain_depth(self) -> None: self._salu_raw_chain_depth += 1
  def reset_salu_raw_chain_depth(self) -> None: self._salu_raw_chain_depth = 0

  # ── Mutations routed through methods so future steps can swap logic ────────
  def set_scc_write_time(self, cycle: int) -> None: self._scc_write_time = cycle
  def set_exec_write_time(self, cycle: int) -> None: self._exec_write_time = cycle
  def mark_drain(self) -> None: self._first_branch_after_drain = True
  def consume_drain_branch(self) -> None: self._first_branch_after_drain = False
  def set_last_salu_issue(self, cycle: int) -> None: self._last_salu_issue = cycle
