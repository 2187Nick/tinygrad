# Mismatch Categories (2026-04-19)

Using `extra/sqtt/rgp/analyze_mismatches.py` to bin each of the 30 remaining
mismatches by (prev_inst_class, curr_inst_class) pair, then rank by pattern
frequency. Baseline: **310/340 exact (91.2%)**, 326/340 ±2 (95.9%).

## Pair patterns (prev→curr) sorted by count

| Count | Prev → Curr              | HW dts                | EMU dts              | Δ mode | Category |
|---|---|---|---|---|---|
| 7 | V_CNDMASK → V_CNDMASK       | 1×3, 4×1, 3×1         | 1×4, 3×2, 2×1        | +2 (2×) | **CNDMASK taper** |
| 3 | V_ADDSUB → GSTORE           | 17×3                  | 21×3                 | +4 (3×) | Wave-variance |
| 3 | V_CNDMASK → OTHER           | 3×2, 1×1              | 1×2, 3×1             | −2 (2×) | CNDMASK taper |
| 2 | S_MOV → S_CMP → S_CBR       | 8, 10                 | 9, 9                 | ±1      | Wave-variance |
| 2 | S_NOP → S_NOP               | 20×2                  | 16×2                 | −4      | Wave-variance (probe_sgpr_cmps) |
| 2 | V_CMP_LIT → V_CMP_LIT       | 3, 1                  | 4, 4                 | +1      | Writer-stall wave-variance |
| 2 | OTHER → OTHER               | 2, 1                  | 4, 5                 | +2      | mixed |
| 1 | V_TRANS → V_TRANS           | 10                    | 4                    | −6      | **Trans chain RAW positional** |
| 1 | V_CMP_LIT → S_NOP           | 18                    | 22                   | +4      | singleton |
| 1 | V_CNDMASK → GSTORE          | 21                    | 25                   | +4      | singleton (wave-variance) |
| 1 | WAIT → S_NOP                | 20                    | 16                   | −4      | Wave-variance |
| 1 | WAIT_LGKM → V_SHIFT         | 4                     | 1                    | −3      | singleton |
| ~ | …misc singletons            |                       |                      |         |          |

## Category summary

| Category | Count | Actionable? |
|---|---|---|
| Wave-variance (opposite-direction ±1 on matched waves) | ~15 | No — deterministic emu can't match both |
| CNDMASK chain taper | ~9 | Partially — requires precise commit-buffer model |
| Trans chain RAW positional | ~1 | Yes — Batch A `mb_trans_exp_n4` shows [1,14,14,4] |
| Isolated singletons | ~5 | Low priority |

## Wave-variance details (confirmed not-actionable)

Same instruction pattern, same kernel — HW wave 0 ≠ HW wave 1:

```
V_ADDSUB → GSTORE (v_add_f32 v[1], 1.0, v[1] → global_store):
  data_deps:         W0 HW=21  W1 HW=17  (W1 is "fast")
  probe_cmp_chain:   W0 HW=17  W1 HW=21  (W0 is "fast")
  probe_branch_cost: W0 HW=17  W1 HW=21  (W0 is "fast")
  probe_sgpr_cmps:   W0 HW=21  W1 HW=21  (both match EMU)
```

The "fast wave" flips between W0 and W1 across kernels. EMU always
predicts 21 — matches the "slow" wave. No constant change fixes both:
forcing bypass would trade the 21cy-matches for 17cy-mismatches.

Same picture for `S_MOV→S_CMP→S_CBR` (HW W0=8 W1=10, EMU=9 for both)
and `S_NOP→S_NOP→S_NOP` (modal=16 vs single-wave=20).

## CNDMASK chain taper (the largest actionable category)

From `exp_chain` around [33-36] (modal across 1 wave only):

```
[33] v_cndmask reads VCC       HW dt=1  EMU dt=1  ✓
[34] v_cndmask reads s[0]      HW dt=4  EMU dt=1  Δ−3  (first cmp_lit consumer)
[35] v_cndmask reads s[1]      HW dt=1  EMU dt=3  Δ+2  (subsequent 1cy)
[36] v_cndmask reads s[2]      HW dt=1  EMU dt=2  Δ+1  (subsequent 1cy)
```

**HW pattern:** first cndmask reading a cmp_lit SGPR pays the completion-
buffer A[n] latency, subsequent cndmasks pipeline at 1cy.

**Emu pattern:** applies A[n] uniformly — so the first cndmask under-predicts
(because some prior issue already advanced the clock past A[0]) and
subsequent cndmasks over-predict (because A[n] stacks via SGPR_COMMIT_GAP).

### Attempted fix (reverted)

Added `prev_cndmask_read_cmp_lit` flag to SgprScoreboard. When the previous
VALU was a cndmask reading a cmp_lit SGPR, skip the A[n] wait. Net impact:
**310 → 307 (−3)**. Regressed because:

- At exp_chain [14-15] (different chain), HW shows dt=[3, 2, 1] not [3, 1, 1] —
  so the "warm → 1cy forever" rule is too aggressive.
- The real HW pattern is a multi-cycle taper, not binary.

### What the fix needs to be

The commit-buffer model needs to distinguish:
1. **Entry cost** (first consumer of a cmp_lit chain): pay A[n] minus some
   overlap — the waits on [33]/[34] are absorbed into scalar/pre-cndmask state.
2. **In-chain taper** (2nd, 3rd consumer): dt=1 for most cases, but 2cy for
   the 3rd position of a short chain (already reflected in Batch B k-sweep).
3. **Chain exit cost** (last cndmask → non-cndmask): observed at `V_CNDMASK → OTHER`
   cases where HW=3 EMU=1.

Implementing this cleanly would require a per-chain state machine in
SgprScoreboard — not a constant change. Estimated effort: ~4 hours,
expected gain: 4-6 of the 9 cndmask-taper mismatches close.

## Trans chain RAW positional — NOT actionable against reference suite

Batch A `mb_trans_exp_n4` showed HW=[1,14,14,4] for 4× same-VGPR v_exp RAW,
vs EMU=[1,4,4,4]. Investigated whether this applies to reference kernels.

**Finding:** the [1,14,14,4] pattern does NOT appear in any reference kernel.
Survey of all V_TRANS→V_TRANS pairs in the suite:

| Kernel | Wave | Idx | HW | EMU | Same VGPR? | Status |
|---|---|---|---|---|---|---|
| exp_chain | W0 | [20-22] | 4,4,4 | 4,4,4 | ❌ (v[0]→v[1]→v[2]→v[3]) | ✓ match |
| exp_chain | W0 | [41-43] | 4,4,4 | 4,4,4 | ❌ (v_log chain) | ✓ match |
| exp_chain | W0 | [69-70] | 4,4 | 4,4 | ❌ (v_sqrt v[4]→v[6]→v[8]) | ✓ match |
| probe_sgpr_cmps | W0 | [18-19] | 4,4 | 4,4 | ✓ (v_exp→v_log→v_sqrt all v[10]) | ✓ match |
| probe_sgpr_cmps | W1 | [18] | **10** | 4 | ✓ (same as W0) | ✗ wave-variance |
| softmax | W0 | [34] | 4 | 5 | ❌ (v[0]→v[3]) | ✗ Δ−1 |

The reference kernels use DIFFERENT VGPRs per trans in their chains, so the
same-VGPR 14cy interlock never fires. The only same-VGPR case (probe_sgpr_cmps)
matches at 4cy on W0 and shows 10cy on W1 — which is wave-variance, not a
missing model. **No fix needed; no mismatches closable via this path.**

## Conclusion for the 100% plan

### Hard stochastic ceiling

The reference .pkl files are per-wave captures (typically 2 waves).
Wave-0 and wave-1 of the **same instruction in the same kernel** often
produce different dts — this is real HW arbitration jitter, not emu bug.

Examples with confirmed wave-variance:

| Pattern | Kernel | W0 HW | W1 HW | EMU | Comment |
|---|---|---|---|---|---|
| V_ADDSUB→GSTORE | data_deps | 21 | 17 | 21 | W1 "fast" |
| V_ADDSUB→GSTORE | probe_cmp_chain | 17 | 21 | 21 | W0 "fast" |
| V_ADDSUB→GSTORE | probe_branch_cost | 17 | 21 | 21 | W0 "fast" |
| S_MOV→S_CMP→S_CBR | probe_branch_cost | 8 | 10 | 9 | both ±1 |
| V_TRANS→V_TRANS (v_exp→v_log v[10]) | probe_sgpr_cmps | 4 | 10 | 4 | W1 slow |

A deterministic emu with one constant choice per transition matches at
most one of each pair. This sets a **hard ceiling of ~320/340 (94.1%)**
against the current reference .pkl files — higher is mathematically
impossible without a seeded wave-scheduler model.

### Realistic path to the ceiling

With only one category remaining actionable (cndmask chain taper, ~9 cases),
and best-case 4-6 of those closable via a per-chain state machine:

- **Optimistic ceiling: 316/340 (92.9%)**
- **Realistic landing: 313-314/340 (92.1%)** — the Batch B k-sweep
  microbenches currently match emu behavior; the state machine must not
  regress those.

### Beyond the ceiling: re-capture as modal

The one path to 100% is to **change the target, not the emu**: re-capture
each reference kernel at 1024-thread dispatch (16 waves per kernel), then
compare EMU against the wave-modal value per token. Emu's deterministic
predictions line up with HW modal for almost all the currently-failing
tokens. Implementation: modify `run_capture` in rigorous_hw_test.py to
capture 1024 threads, then pick `modal_dt` per token_index across waves
as the reference.

This would convert nearly all wave-variance mismatches into exact matches
(since the modal is what emu effectively predicts), bringing the suite
to ~325-330/340 (~96%) with no emu changes at all. The remaining ~10-15
would be genuine emu-modeling gaps.

### Ranked next steps

1. **Modal re-capture** — largest expected gain (~15 mismatches close).
   Requires sudo for HW capture; 1-2 hours of capture time.
2. **Cndmask chain taper state machine** — 4-6 additional closures if
   modal re-capture doesn't already catch them.
3. **Wave-variance documentation** — set user expectations that 100% is
   unreachable against per-wave references, and document the ceiling.
4. **Live/stochastic emu mode** (long-term) — the only path to per-wave
   matching. Large refactor; low priority until after (1)-(3).
