# Stochastic Wave Scheduler — Plan & Progress

## Problem

The emulator currently simulates each wave with identical state so every
wave's SQTT trace has the same per-token dt. HW does NOT: the same
instruction at the same position produces different dts on different
waves due to inter-wave SQ arbitration, scoreboard contention, and
wave-launch stagger.

Today's gap:
- Reference strict (MODAL=0): 319/340 → same code, per-wave divergence
  accounts for the 21-token delta between MODAL=0 and MODAL=1.
- Microbench strict: 34047/44126 (77.2%) vs MODAL 41456/44126 (93.9%)
  → **~7400 tokens** of wave-variance in the microbench suite.

Goal: reach 100% strict (per-wave exact) — reproduce HW's per-wave dt
distribution deterministically so every wave matches its HW counterpart
rather than relying on MODAL to accept any wave's dt.

## Starting state (2026-04-19 after the 6-hour push session)

- Reference: 339/340 exact (99.7% MODAL), 340/340 ±2 (100%). Strict 319/340.
- Microbench: 41456/44126 exact (93.9% MODAL), 42836/44126 ±2 (97.1%).
  Strict 34047/44126 (77.2%).
- All 8 bounty tests pass.
- Last commit: `6140d5e4b` on master, pushed to origin.

## What we know from prior probing

1. **Waves launch staggered** — `_WAVESTART_GAP=1cy` already modeled. But
   HW observed stagger is larger for later waves in 16-wave dispatches.
2. **Shared-SIMD model falsified** — 2-wave WGs land on different SIMDs
   (RGP captures confirm). So same-SIMD serialization is not the cause.
3. **Wave-variance seen on:**
   - `probe_branch_cost` [7] s_cbranch: W0=8, W1=10 (±2cy jitter)
   - `probe_sgpr_cmps` [16] v_cndmask: W0=2, W1=5
   - `where` [18] b128 store: W0=21, W1=25
   - `mb_valu_add_n4` across 16 waves: most at dt=1, later waves at dt=3-5
4. **MODAL catches it** because EMU picks SOME valid wave's dt.
5. The variance is not random — it's a function of **wave index** and
   **position in the kernel** (later waves → bigger stalls at specific
   points; SQ queue drains unevenly).

## Hypothesis (to validate)

The HW SQ uses a **credit-based round-robin** across wave slots:
- At launch, each wave consumes a launch slot staggered by 1cy per wave.
- When waves hit a waitcnt/cmp-chain/store, the ones that reach the
  contention point earlier win the bypass slot; later arrivals pay the
  penalty cycles.
- Since waves are identical programs, later-launched waves reach EVERY
  contention point later by a fixed offset — **but that offset can
  COMPOUND** through vmcnt drains (where slow waves wait extra cycles
  for VMEM issue bandwidth).

Simplest candidate model: **per-wave arbitration delay** that grows with
wave index at each scheduling-contention point.

## Plan

### Phase 0 — Measurement & classification (45 min)

Build an analyzer that takes HW captures and produces, per kernel, the
per-wave dt profile. Classify each token:

- **INVARIANT** — all 16 waves have the same dt (already matched).
- **LINEAR(slope)** — dt grows linearly with wave_idx (or some subset).
- **BIMODAL(a, b)** — exactly two dt values across waves, 50/50 split.
- **CLUSTERED(N)** — waves group into N clusters of same dt.
- **CHAOTIC** — no clean pattern; each wave independent. (Hard to fix.)

Output goes to `STOCHASTIC_FINGERPRINTS.md`. This tells us which
microbenches have tractable variance vs which are CHAOTIC (likely
unreachable without full HW arbitration simulation).

Tool: new script `extra/sqtt/rgp/analyze_wave_variance.py`.

### Phase 1 — Wave-index arbitration offset (1-2 hr)

Model the simplest variance pattern: **later waves arrive at contention
points with a per-wave offset that accumulates through drains.**

Implementation:

1. Add `wave_idx` parameter to `_simulate_sq_timing` state (already
   tracked per-wave via `wave_ids`; just expose it in stall rules).
2. At **waitcnt-effective drain**, add per-wave stagger:
   `stall_until += wave_idx * WAVE_ARB_SLOPE` where
   `WAVE_ARB_SLOPE` is calibrated (likely 1-2cy per wave).
3. At **cmp-chain → cndmask boundary** (post-drain), add an additional
   arbitration jitter calibrated against `probe_sgpr_cmps [16]`.
4. At **VMEM store bypass check**, use the wave_idx to decide which
   wave gets the 17cy fast path vs 21cy slow path (currently all-or-
   nothing based on n; model the split that HW shows).

Constants live in `sq_timing/constants.py` so they can be tuned from
measurement data without touching logic.

### Phase 2 — Validate & regression-guard (30 min)

Run:
- Reference `--compare` (MODAL on): must stay 339-340 exact.
- Reference MODAL=0: target 327+/340 (up from 319).
- Microbench MODAL=0: target 37000+/44126 (up from 34047).
- 8 bounty tests must pass.

If any phase regresses, roll back that phase's constants to 0 (effectively
disabling it) rather than disabling the mechanism — keeps the hook in
place for future tuning.

### Phase 3 — Harder variance patterns (remaining time)

If the simple wave-idx linear model closes 30-50% of the gap, iterate on
the remaining CHAOTIC patterns:
- Per-CU placement (RGP data shows CU assignments are deterministic
  per dispatch but depend on workgroup geometry).
- VMEM credit window (store `vmem_issue_credits[i]` per wave, drain on
  use, refill at inter-wave bypass events).

## Risk register

- **Reference regression**: current reference matches MODAL at 100%±2;
  any change that shifts per-wave dts risks breaking those. Mitigation:
  all new constants default to 0 behavior until positively calibrated.
- **Batch D rule interactions**: phase_shift_armed / VOPD self-fwd
  rules already depend on per-wave state. Adding arb_delay here could
  double-count. Mitigation: apply arb_delay at a single well-defined
  integration point (waitcnt drain) rather than sprinkling through.
- **Running out of time**: if Phase 0 measurement shows most variance is
  CHAOTIC, Phase 1 won't help much — pivot to documenting the ceiling.

## Progress log

### 2026-04-19 session start
- Plan doc created. Starting Phase 0 measurement.

(Updates below; newest last.)
