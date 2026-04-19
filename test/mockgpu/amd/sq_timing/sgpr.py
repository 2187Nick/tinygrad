"""Per-wave SGPR scoreboard tracker.

Extracted from `_simulate_sq_timing` per EMU_REWRITE_DESIGN.md §1.5 / §5 Step 5b.
Pure refactor — zero behaviour change. Owns per-wave SGPR write-time scoreboard,
the LIT v_cmp completion buffer (answer.md model), and the SMEM-return map.

Scope note: this class is a simple state-holder (getters, setters, prune)
following the Step 4 VmemPipe pattern. Logic (read_stall, writer kind,
pending_nonvcc_drain) stays in emu.py for this step — later steps may
refactor methods in.
"""
from test.mockgpu.amd.sq_timing.constants import CONST, TimingConstants


class SgprScoreboard:
  """Per-wave SGPR scoreboard tracker.

  Owns:
    - `sgpr_write_time[reg]`: last VALU / SALU SGPR write cycle per reg.
    - `sgpr_cmp_lit_read_ready[reg]`: A[n] per LIT v_cmp-written SGPR
      (answer.md completion-buffer reader latency).
    - `sgpr_cmp_lit_last_commit`: last C[n] in the current LIT v_cmp chain
      (for commit-gap serialization).
    - `sgpr_cmp_lit_hist`: recent LIT v_cmp I[n] issue cycles (depth-2 writer
      stall — N-th LIT v_cmp must wait for (n-2)th to propagate).
    - `smem_sgpr_ready[reg]`: SGPR index → cycle when SMEM result available.
  """
  def __init__(self, const: TimingConstants = CONST):
    self._c = const
    self._write_time: dict[int, int] = {}
    self._cmp_lit_read_ready: dict[int, int] = {}
    self._cmp_lit_last_commit: int = 0
    self._cmp_lit_hist: list[int] = []
    self._smem_ready: dict[int, int] = {}
    # Phase offset applied to next cmp_lit chain's A[n]. Set to +3 after s_waitcnt_depctr
    # (exp_chain [33-36] confirms: depctr-drain + VOPD + cmp chain shifts SGPR read-ready by ~3cy
    # vs a VALU-prefix cmp chain like exp_chain [8-11]). Consumed on first cmp_lit A[n] update.
    self._next_cmp_lit_phase_offset: int = 0

  @property
  def next_cmp_lit_phase_offset(self) -> int: return self._next_cmp_lit_phase_offset
  def set_next_cmp_lit_phase_offset(self, v: int) -> None: self._next_cmp_lit_phase_offset = v

  # ── Write-time scoreboard ──────────────────────────────────────────────────
  def write_time_map(self) -> dict[int, int]:
    """Return the raw sgpr_write_time dict (caller may read/iter/replace)."""
    return self._write_time

  def set_write_time(self, reg: int, cycle: int) -> None:
    self._write_time[reg] = cycle

  def replace_write_time(self, new_map: dict[int, int]) -> None:
    """Wholesale replace the write-time dict (used for drain-prune filters)."""
    self._write_time = new_map

  # ── LIT v_cmp completion buffer ────────────────────────────────────────────
  def cmp_lit_read_ready_map(self) -> dict[int, int]:
    """Return the LIT v_cmp reader-ready dict (caller may read/pop/clear)."""
    return self._cmp_lit_read_ready

  def set_cmp_lit_read_ready(self, reg: int, cycle: int) -> None:
    self._cmp_lit_read_ready[reg] = cycle

  def pop_cmp_lit_read_ready(self, reg: int) -> None:
    self._cmp_lit_read_ready.pop(reg, None)

  def clear_cmp_lit_read_ready(self) -> None:
    self._cmp_lit_read_ready.clear()

  @property
  def cmp_lit_last_commit(self) -> int: return self._cmp_lit_last_commit
  def set_cmp_lit_last_commit(self, cycle: int) -> None: self._cmp_lit_last_commit = cycle

  def cmp_lit_hist(self) -> list[int]:
    """Return the raw LIT v_cmp issue-cycle history (caller may read/append/pop/clear)."""
    return self._cmp_lit_hist

  def clear_cmp_lit_hist(self) -> None: self._cmp_lit_hist.clear()

  # ── SMEM-return map ────────────────────────────────────────────────────────
  def smem_ready_map(self) -> dict[int, int]:
    """Return the raw SMEM-return dict (caller may read/filter)."""
    return self._smem_ready

  def set_smem_ready(self, reg: int, cycle: int) -> None:
    self._smem_ready[reg] = cycle

  def replace_smem_ready(self, new_map: dict[int, int]) -> None:
    """Wholesale replace the SMEM-return dict (used for drain-prune filters)."""
    self._smem_ready = new_map
