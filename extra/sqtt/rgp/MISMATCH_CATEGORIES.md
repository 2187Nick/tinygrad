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

## Trans chain RAW positional (singleton, but calibrated)

Batch A `mb_trans_exp_n4` (4× `v_exp_f32 v[1],v[1]`) shows modal dts
[1, 14, 14, 4]. Current EMU predicts [1, 4, 4, 4] via `TRANS_PIPE_CYCLES=4`.

Probe_sgpr_cmps W1 [18] shows `v_exp → v_log` RAW on v[10] at HW=10 EMU=4.

**Rule:** the 2nd and 3rd trans in a same-VGPR RAW chain pay a ~14cy
interlock (partial pipeline flush for mid-chain reads); the last one
aligns with the pipeline tail at 4cy. A targeted fix in `TransPipe` could
track "positions-into-chain" and apply the tiered delay. Estimated effort:
~2 hours, expected gain: 1-2 mismatches close.

## Conclusion for the 100% plan

**Realistic floor:** 310/340 → **~318-322/340 (94-95%)** with the two
viable fixes landed. Beyond that requires implementing HW's wave-arbitration
stochastic scheduler to match wave-0 vs wave-1 variance — that's a
fundamental architecture change to the emu, not a constant tweak.

**Ranked path forward:**
1. (1-2 wins) Trans chain RAW positional — isolated, Batch A pre-calibrated.
2. (4-6 wins) Cndmask chain taper state machine — per-chain phase tracker.
3. (0 wins, closes debate) Wave-variance documentation — set expectations.
4. (0-5 wins) Re-capture references as modal-across-16-waves — changes the
   target, not the emu. Would convert wave-variance mismatches into exact
   matches against the modal.
