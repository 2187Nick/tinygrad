"""Per-wave VALU pipeline tracker.

Extracted from `_simulate_sq_timing` per EMU_REWRITE_DESIGN.md §1.2 / §5 Step 5c.
Pure refactor — zero behaviour change. Owns per-wave non-trans VALU issue history,
VOPD dual-issue pipeline state (pipe-available, last-issue, 4-bank write times),
consecutive-VALU run-length counters, and the VGPR scoreboard (ready / slow-fresh
/ write-time).

Scope note: this class is a simple state-holder (getters, setters, prune)
following the Step 4/5a/5b pattern. Logic stays in emu.py for this step —
later steps may refactor methods like `valu_dep_stall` / `vgpr_read_stall`
/ `vopd_bank_port_stall` / `on_valu_issue` in. `trans_vgpr_ready` is
intentionally NOT owned here — it lives in TransPipe (Step 5a).
"""
from test.mockgpu.amd.sq_timing.constants import CONST, TimingConstants


class VAluPipe:
  """Per-wave VALU pipeline tracker.

  Owns:
    - `valu_issue_hist`: last 4 non-trans VALU issue cycles (for time-based
      delay_alu resolution).
    - `vopd_pipe_avail`: next cycle VOPD unit is free for next VOPD.
    - `last_vopd_issue`: last VOPD issue cycle (-1000 init; used for
      warm/cold dual-issue classification).
    - `bank_vopd_write_time[bank]`: 4-element list tracking the last VOPD
      issue cycle that wrote each bank (bank = reg & 3) — used for
      inter-VOPD bank port pressure.
    - `consecutive_single_valu`: non-trans non-VOPD VALU run-length.
    - `consecutive_selffwd_vgprs`: count of VGPRs written by consecutive
      self-forwarding non-VOPC VALUs (for VMEM store forwarding).
    - `consecutive_vgprs_written`: count of VGPRs written by consecutive
      non-VOPC VALUs.
    - `cndmask_cluster_vgprs`: count of VGPRs written by a cndmask chain
      (V_CNDMASK_* ops). Unlike `consecutive_vgprs_written`, VOPCs don't
      break this cluster — HW `where` kernel [18] shows cmp-interleaved
      cndmask writes feed a subsequent b128 store at `deadline` (no
      width_extra, no scatter +1). Any non-cndmask non-VOPC VALU resets.
    - `vgpr_ready[reg]`: VGPR readiness scoreboard (reg → cycle when
      result is available).
    - `vgpr_slow_fresh_until[reg]`: VGPR slow-freshness window end —
      consuming before this cycle yields 9cy VALU latency.
    - `vgpr_write_time[reg]`: VGPR last VALU-write cycle (for VMEM address
      forwarding).
  """
  def __init__(self, const: TimingConstants = CONST):
    self._c = const
    self._issue_hist: list[int] = []
    self._vopd_pipe_avail = 0
    self._last_vopd_issue = -1000
    self._bank_vopd_write_time: list[int] = [0, 0, 0, 0]
    self._consecutive_single_valu = 0
    self._consecutive_selffwd_vgprs = 0
    self._consecutive_vgprs_written = 0
    self._cndmask_cluster_vgprs = 0
    self._vgpr_ready: dict[int, int] = {}
    self._vgpr_slow_fresh_until: dict[int, int] = {}
    self._vgpr_write_time: dict[int, int] = {}

  # ── VALU issue history (time-based delay_alu) ──────────────────────────────
  def issue_hist(self) -> list[int]:
    """Return the raw last-4 non-trans VALU issue-cycle list."""
    return self._issue_hist

  # ── VOPD state ─────────────────────────────────────────────────────────────
  @property
  def vopd_pipe_avail(self) -> int: return self._vopd_pipe_avail
  def set_vopd_pipe_avail(self, cycle: int) -> None: self._vopd_pipe_avail = cycle

  @property
  def last_vopd_issue(self) -> int: return self._last_vopd_issue
  def set_last_vopd_issue(self, cycle: int) -> None: self._last_vopd_issue = cycle

  def bank_vopd_write_time(self) -> list[int]:
    """Return the raw 4-element VOPD bank-write-time list (caller may read/write index)."""
    return self._bank_vopd_write_time

  def set_bank_vopd_write_time(self, bank: int, cycle: int) -> None:
    self._bank_vopd_write_time[bank] = cycle

  # ── VALU run-length counters ───────────────────────────────────────────────
  @property
  def consecutive_single_valu(self) -> int: return self._consecutive_single_valu
  def set_consecutive_single_valu(self, v: int) -> None: self._consecutive_single_valu = v
  def inc_consecutive_single_valu(self) -> None: self._consecutive_single_valu += 1

  @property
  def consecutive_selffwd_vgprs(self) -> int: return self._consecutive_selffwd_vgprs
  def set_consecutive_selffwd_vgprs(self, v: int) -> None: self._consecutive_selffwd_vgprs = v
  def add_consecutive_selffwd_vgprs(self, v: int) -> None: self._consecutive_selffwd_vgprs += v

  @property
  def consecutive_vgprs_written(self) -> int: return self._consecutive_vgprs_written
  def set_consecutive_vgprs_written(self, v: int) -> None: self._consecutive_vgprs_written = v
  def add_consecutive_vgprs_written(self, v: int) -> None: self._consecutive_vgprs_written += v

  @property
  def cndmask_cluster_vgprs(self) -> int: return self._cndmask_cluster_vgprs
  def set_cndmask_cluster_vgprs(self, v: int) -> None: self._cndmask_cluster_vgprs = v
  def add_cndmask_cluster_vgprs(self, v: int) -> None: self._cndmask_cluster_vgprs += v

  # ── VGPR scoreboard (ready / slow-fresh / write-time) ──────────────────────
  def vgpr_ready_map(self) -> dict[int, int]:
    """Return the raw VGPR readiness dict (caller may read/pop/write)."""
    return self._vgpr_ready

  def vgpr_slow_fresh_map(self) -> dict[int, int]:
    """Return the raw VGPR slow-freshness dict (caller may read/pop/write)."""
    return self._vgpr_slow_fresh_until

  def vgpr_write_time_map(self) -> dict[int, int]:
    """Return the raw VGPR write-time dict (caller may read/pop/write)."""
    return self._vgpr_write_time
