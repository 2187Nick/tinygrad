"""CU-shared SIMD VALU issue arbiter (state holder — NOT YET WIRED).

Step 1 of the SIMD-arbiter refactor (see extra/sqtt/HANDOFF_OVERNIGHT.md
Option A). This file introduces the state container only; emu.py still uses
the existing per-wave heuristics. Wiring happens in a later step once
shadow-mode comparison confirms the arbiter reproduces the heuristics'
outputs on the calibrated microbenches.

Architecture recap
------------------
RDNA3 CU has 4 SIMDs × 32 lanes. A wave64 wave runs on a single SIMD across
two back-to-back 32-lane issues. Wave→SIMD mapping is fixed at dispatch and
observable via HW_ID[5:4] — emu.py:3365 already uses
`SIMD_ID = wave_idx % 4`, so we adopt the same rule here.

Each SIMD owns an independent VALU issue port. Waves sharing a SIMD serialize
through that port; waves on different SIMDs issue in parallel. VOPD dual-
issue consumes BOTH lanes of a single SIMD in one cycle — it does NOT multi-
SIMD serialize (HW `mb_vopd_chain_n4_raw` pipelines at 1cy even when all 4
waves fall on SIMD 0 via i%4 for n=4).

Heuristics this arbiter is intended to eventually subsume
--------------------------------------------------------
The following rules currently approximate per-SIMD arbitration. They are
listed here (file:line) so a future cutover step can replace them one at a
time with arbiter calls and validate per-kernel that behaviour is unchanged.

  1. VALU burst priority              emu.py:291, 532-544, 1129-1138
     Keeps one wave issuing VALUs back-to-back on "its" port, and on burst
     end forces peer ready[j] ≥ prev_issue_cycle+1. This is a coarse
     oldest-wave-monopolizes proxy.

  2. Per-wave s_cbranch_scc split     emu.py:696-701
     Wave 0 issues -1cy (wins scalar-pipe slot after drain); wave 1+ pays
     +1cy. Calibrated: HW probe_branch_cost [7] W0=8, W1=10.

  3. Wave-credit VGPR RAW stall       emu.py:703-724
     i > 0 AND (backward chain_depth ≥ 4 OR long_raw_chain[pc]) AND
     immediate-predecessor RAW → issue_cycle = max(…, last_valu + 5).
     Calibrated: mb_valu_add_n16 waves 1-15 unanimous dt=5; mb_valu_add_n4
     all waves dt=1. In SIMD-arbiter terms: port is owned by oldest wave,
     other SIMD-peers queue behind.

  4. Long-chain pre-pass              emu.py:218-246 (produces long_raw_chain)
     Flags positions inside a same-reg RAW chain of length ≥ 6. Feeds
     rules 3 and 6. In arbiter terms: chain length gates how quickly the
     port owner can drain the issue queue before peers get a slot.

  5. Trans chain wave-stagger         emu.py:743-753
     i ≥ 4 AND trans_pipe_avail busy AND long_raw_chain → +10cy. The 4-wave
     boundary matches a SIMD boundary under i%4 when n≥8 (waves 4-7 on
     SIMDs 0-3 start queueing behind waves 0-3).

  6. VOPD last-cndmask floor          emu.py:774-783
     in_phase_shifted_chain AND cndmask_cluster ≥ 4 → issue no sooner than
     last_cndmask_issue + 3. Uses per-wave state, not peer-wave state, so
     this is semantically scalar-pipe-side and NOT an arbiter concern —
     listed only to rule it out.

  7. simd_valu_avail stub             emu.py:297
     Declared but unused. The arbiter will take over this responsibility.

Refactor ordering (planned)
--------------------------
  Step 1 (this file)   scaffold + inventory.
  Step 2               shadow mode: at each VALU issue compute
                       arbiter.would_issue(i, issue_cycle) and log the
                       delta to a side channel.
  Step 3               unit tests vs mb_valu_add_n16 / mb_vopd_chain_n4_raw
                       / mb_f2_raw_all_banks_n4.
  Step 4+              cutover one heuristic at a time, verifying strict
                       totals ≥54830 and MODAL ≥65259 after each.

Design notes
-----------
- State is CU-shared (one arbiter per workgroup simulation), not per-wave.
  Every other class under this package is per-wave; this one is not.
- The only per-SIMD state we actually need is `port_avail[s]` (next cycle
  the VALU port is free) plus `owner_wave[s]` (the wave currently holding
  the port, used to implement the "oldest-wave monopolizes until drain"
  policy). VOPD is modelled as consuming the same port for 1cy.
- Wave→SIMD mapping is immutable per-dispatch and exposed as a method for
  future experimentation (e.g. verifying via RGP placement data).
"""
from __future__ import annotations

from test.mockgpu.amd.sq_timing.constants import CONST, TimingConstants


N_SIMDS = 4          # RDNA3 CUs have 4 SIMDs — HW_ID[5:4] confirms (emu.py:3365)
NO_OWNER = -1


class SimdArbiter:
  """CU-shared SIMD VALU issue arbiter.

  Owns:
    - `port_avail[s]`: next cycle VALU issue port on SIMD `s` is free.
    - `owner_wave[s]`: wave index currently holding the port (oldest-wave
      priority), or NO_OWNER when no wave is actively streaming.
    - `last_issue_cycle[s]`: cycle of the most recent VALU issue on SIMD
      `s` (for tie-breaks / telemetry).

  Skeleton only — no logic methods yet. Wiring lands in Step 2+.
  """
  __slots__ = ('_c', '_n_waves', '_port_avail', '_owner_wave', '_last_issue_cycle')

  def __init__(self, n_waves: int, const: TimingConstants = CONST):
    self._c = const
    self._n_waves = n_waves
    self._port_avail: list[int] = [0] * N_SIMDS
    self._owner_wave: list[int] = [NO_OWNER] * N_SIMDS
    self._last_issue_cycle: list[int] = [-1] * N_SIMDS

  # ── Wave → SIMD mapping ────────────────────────────────────────────────────
  @staticmethod
  def simd_for_wave(wave_idx: int) -> int:
    """Return the SIMD index a wave lives on. Matches HW_ID[5:4] = wave_idx & 3."""
    return wave_idx & 0x3

  def peers_on_simd(self, wave_idx: int) -> list[int]:
    """Return all wave indices that share a SIMD with `wave_idx` (excluding self)."""
    s = self.simd_for_wave(wave_idx)
    return [w for w in range(self._n_waves) if w != wave_idx and self.simd_for_wave(w) == s]

  # ── Port availability ──────────────────────────────────────────────────────
  def port_avail(self, simd: int) -> int: return self._port_avail[simd]
  def set_port_avail(self, simd: int, cycle: int) -> None: self._port_avail[simd] = cycle

  def port_avail_for_wave(self, wave_idx: int) -> int:
    return self._port_avail[self.simd_for_wave(wave_idx)]

  # ── Owner tracking (oldest-wave priority) ──────────────────────────────────
  def owner_wave(self, simd: int) -> int: return self._owner_wave[simd]
  def set_owner_wave(self, simd: int, wave_idx: int) -> None: self._owner_wave[simd] = wave_idx
  def clear_owner(self, simd: int) -> None: self._owner_wave[simd] = NO_OWNER

  # ── Last-issue telemetry ───────────────────────────────────────────────────
  def last_issue_cycle(self, simd: int) -> int: return self._last_issue_cycle[simd]
  def set_last_issue_cycle(self, simd: int, cycle: int) -> None: self._last_issue_cycle[simd] = cycle

  # ── Debug / test snapshot ──────────────────────────────────────────────────
  def snapshot(self) -> dict:
    """Return a plain-dict snapshot for unit tests and shadow-mode diffs."""
    return {
      'port_avail': list(self._port_avail),
      'owner_wave': list(self._owner_wave),
      'last_issue_cycle': list(self._last_issue_cycle),
    }
