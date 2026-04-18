"""CU-shared LDS pipeline tracker.

Extracted from `_simulate_sq_timing` per EMU_REWRITE_DESIGN.md §1.7 / §5 Step 3.
Pure refactor — zero behaviour change. Owns the three CU-shared counters
(`cu_lds_available`, `cu_lds_last_was_write`, `cu_lds_rd_available`) plus the
per-wave `lgkm_pend` queue (used by s_waitcnt_lgkmcnt drain and by SMEM too —
SMEM completions also live on this queue since lgkmcnt spans both LDS and
scalar-memory ops on RDNA3).

Scope note: the b128 VGPR-stagger / slow-fresh logic referenced by the LDS
branch in emu.py writes `vgpr_ready` / `vgpr_slow_fresh_until` — that is VGPR
scoreboard state and belongs in `VAluPipe` (Step 5). We return the computed
`lds_complete` cycle and an `is_serialized` flag so the caller can still apply
the VGPR stagger without LdsPipe depending on VGPR state.
"""
from test.mockgpu.amd.sq_timing.constants import CONST, TimingConstants


class LdsPipe:
  """CU-shared LDS pipeline tracker.

  Owns (CU-shared):
    - `cu_lds_available`: next cycle the LDS unit is free for a write.
    - `cu_lds_last_was_write`: True if the last DS op was a write (used for
      write→read mode-switch penalty).
    - `cu_lds_rd_available`: next cycle the LDS unit is free for a b128 read
      (b128 reads serialize more aggressively than normal reads).
  Owns (per-wave):
    - `lgkm_pend[wave]`: pending LDS / SMEM completion cycles for
      s_waitcnt_lgkmcnt stall computation.
  """
  def __init__(self, n_waves: int, const: TimingConstants = CONST):
    self._const = const
    self._n = n_waves
    self._cu_lds_available: int = 0
    self._cu_lds_last_was_write: bool = False
    self._cu_lds_rd_available: int = 0
    self._lgkm_pend: list[list[int]] = [[] for _ in range(n_waves)]

  # ── Pending-queue access (used by s_waitcnt_lgkmcnt) ───────────────────────
  def lgkm_pending(self, wave: int) -> list[int]:
    """Return the raw pending list for `wave` (caller may sort/index)."""
    return self._lgkm_pend[wave]

  def prune(self, wave: int, stall_until: int) -> None:
    """Drop completions ≤ stall_until after a waitcnt resolves."""
    self._lgkm_pend[wave] = [c for c in self._lgkm_pend[wave] if c > stall_until]

  # ── SMEM (smem completions also live on lgkm_pend) ─────────────────────────
  def on_smem_issue(self, wave: int, completion_cycle: int) -> None:
    """Append an SMEM completion cycle to `wave`'s lgkm queue."""
    self._lgkm_pend[wave].append(completion_cycle)

  # ── DS write ───────────────────────────────────────────────────────────────
  def on_ds_write_issue(self, wave: int, issue_cycle: int) -> int:
    """Account for a ds_write at `issue_cycle`. Returns the LDS start cycle
    (after CU contention), so the caller can trace/debug. Updates CU-shared
    state: cu_lds_available += LDS_SERVICE_COST, cu_lds_last_was_write=True.
    Appends `lds_start + LDS_WR_LATENCY` to the wave's lgkm queue.
    """
    c = self._const
    lds_start = max(issue_cycle, self._cu_lds_available)
    self._lgkm_pend[wave].append(lds_start + c.LDS_WR_LATENCY)
    self._cu_lds_available = lds_start + c.LDS_SERVICE_COST
    self._cu_lds_last_was_write = True
    return lds_start

  # ── DS read ────────────────────────────────────────────────────────────────
  def on_ds_read_issue(self, wave: int, issue_cycle: int, *, ds_bytes: int) -> tuple[int, bool]:
    """Account for a ds_read at `issue_cycle` with `ds_bytes` width.
    Returns `(lds_complete, is_serialized)`:
      - lds_complete: the cycle the read result lands (appended to lgkm queue).
      - is_serialized: True iff this b128 read was delayed waiting for the LDS
        port (used by caller for b128 VGPR stagger on dest VGPRs).
    Updates CU-shared state:
      - cu_lds_rd_available bumped on b128 reads.
      - cu_lds_last_was_write cleared (write→read switch consumed).
    """
    c = self._const
    mode_switch_penalty = 1 if self._cu_lds_last_was_write else 0
    width_extra = c.LDS_B128_EXTRA if ds_bytes >= 16 else 0
    is_serialized = False
    if ds_bytes >= 16:
      lds_rd_start = max(issue_cycle, self._cu_lds_rd_available)
      lds_complete = lds_rd_start + c.LDS_RD_LATENCY + mode_switch_penalty + width_extra
      self._cu_lds_rd_available = issue_cycle + c.LDS_B128_RD_SERVICE
      is_serialized = (lds_rd_start > issue_cycle)
    else:
      lds_complete = issue_cycle + c.LDS_RD_LATENCY + mode_switch_penalty + width_extra
    self._lgkm_pend[wave].append(lds_complete)
    if self._cu_lds_last_was_write: self._cu_lds_last_was_write = False
    return lds_complete, is_serialized

  # ── Debug / introspection ──────────────────────────────────────────────────
  @property
  def cu_lds_available(self) -> int: return self._cu_lds_available
  @property
  def cu_lds_last_was_write(self) -> bool: return self._cu_lds_last_was_write
  @property
  def cu_lds_rd_available(self) -> int: return self._cu_lds_rd_available
