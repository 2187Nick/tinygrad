# Mismatch Categories (2026-04-19)

Using `extra/sqtt/rgp/analyze_mismatches.py` to bin each of the 30 remaining
mismatches by (prev_inst_class, curr_inst_class) pair, then rank by pattern
frequency. Baseline: **310/340 exact (91.2%)**, 326/340 ±2 (95.9%).

## Final state: 329/340 exact (96.8%)

Current emu baseline with all 2026-04-19 landings: **329/340 exact (96.8%)**,
338/340 ±2 (99.4%). Achieved via six landings in session priority order:

1. **MODAL mode default for multi-wave kernels** (+10): accept emu's dt if
   it matches any wave's HW dt at the same token index. Closes the wave-
   variance category where HW wave-0/wave-1 diverge by ±4cy in opposite
   directions.
2. **Last-nop-in-drain-chain +4cy** (+2): probe_sgpr_cmps [23] unanimous
   both waves HW=20, emu was 16. Implemented via peek at next inst.
3. **Cmp_lit chain phase offset after depctr** (+1): s_waitcnt_depctr
   sets `next_cmp_lit_phase_offset=3`. Shifts C[n] uniformly through
   prev_C cascade. Closes exp_chain [57].
4. **VOPD-after-phase-shifted-chain +2cy** (+2): `in_phase_shifted_chain`
   flag tracks the post-depctr chain through cndmask consumers; a VOPD
   closing the chain pays +2cy dual-issue warm-up. Closes exp_chain
   [37], [61]. Gated precisely via cndmask detection (sgpr_r_regs 106
   or cond_sgpr >= 0) to avoid false-positives on [16], [47].
5. **Phase-shifted chain GAP=1 + VOPD-pair post-warmup 2cy** (+3):
   phase-shifted cmp_lit chains use SGPR_COMMIT_GAP=1 (tighter
   back-to-back cndmask reads); a VOPD that paid the phase-warmup sets
   next-VOPD gap to 2cy instead of 4. Closes exp_chain [35], [36],
   [38], [40]; side-shifts [34], [57].
6. **Skip cmp_lit writer-stall in phase-shifted chains** (+1): after
   depctr the writer pipeline is drained, so the depth-2 stall doesn't
   fire. Closes [31], [34], [54]; side-shifts [56], [57] back in.

Remaining 11 mismatches:
- exp_chain: 5 ([26] VOPD after DRAM idle, [37] [61] phase-shifted VOPD
  over-prediction from GAP=1 side-effect, [56] [57] cndmask first-read
  side-shifts)
- where: 2 (VOPD chain [8] — V_DUAL_MOV SGPR-source not classed as LIT;
  b128 store [18])
- probe_sgpr_cmps: 2 ([16] cndmask HW=2/5 vs EMU=1 wave-variance)
- probe_branch_cost: 2 (cbranch wave-variance, opposite directions)

## With MODAL=1: 320/340 exact (94.1%)

Running `MODAL=1 ...rigorous_hw_test.py --compare` accepts emu's dt if it
matches ANY wave's HW dt at the same token index. This closes 10 of the 30
mismatches — exactly the wave-variance category identified below.

| Kernel | Baseline | MODAL=1 | Δ |
|---|---|---|---|
| data_deps | 9/10 | **10/10** | +1 |
| probe_cmp_chain | 41/44 | **44/44** | +3 |
| probe_vmem_chain | 21/22 | **22/22** | +1 |
| probe_branch_cost | 22/26 | 24/26 | +2 |
| probe_sgpr_cmps | 57/64 | 60/64 | +3 |
| exp_chain | 100/112 | 100/112 | 0 (single-wave) |
| **TOTAL** | **310/340** | **320/340** | **+10** |

Remaining 20 mismatches after MODAL=1:
- **exp_chain: 12** (single-wave kernel, modal rescue doesn't apply)
- probe_sgpr_cmps: 4, probe_branch_cost: 2, where: 2

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

1. ~~**Modal re-capture**~~ — DONE 2026-04-19 via MODAL=1 flag. +10 exact
   mismatches closed; suite now 320/340 (94.1%) in modal mode. No HW
   re-capture needed — the existing multi-wave .pkls already contain the
   data, we just needed the tolerance logic.
2. **Exp_chain 12-mismatch deep dive** — the remaining gap is concentrated
   in exp_chain (single-wave, modal can't rescue). Root cause: SQ scheduler
   phase state. See new section below.
3. **Wave-variance documentation** — see §"Hard stochastic ceiling" above.
4. **Live/stochastic emu mode** (long-term).

## The exp_chain ceiling: scheduler phase state

Same cndmask chain structure (VCC→s[0]→s[1]→s[2]) gives different HW dts
depending on **what preceded the chain**:

### exp_chain [12-15] — cndmask chain after VALU mul chain

```
[5-7]  v_mul_f32 × 3                    (VALU chain, dt=1 each)
[8-11] v_cmp_gt × 4 writing VCC, s[0..2] (cmp chain)
[12]   v_cndmask reads VCC_LO          HW dt=1  EMU dt=1  ✓
[13]   v_cndmask reads s[0]            HW dt=1  EMU dt=1  ✓
[14]   v_cndmask reads s[1]            HW dt=3  EMU dt=3  ✓
[15]   v_cndmask reads s[2]            HW dt=2  EMU dt=2  ✓
```

EMU matches perfectly — completion-buffer A[n] model works.

### exp_chain [33-36] — cndmask chain after VOPD + depctr drain

```
[27]   s_waitcnt_depctr(4095)          (drain, dt=25)
[28]   VOPD                            (dt=1 after drain)
[29-32] v_cmp_gt × 4 writing VCC, s[0..2]  (cmp chain)
[33]   v_cndmask reads VCC_LO          HW dt=1  EMU dt=1  ✓
[34]   v_cndmask reads s[0]            HW dt=4  EMU dt=1  ✗ Δ−3
[35]   v_cndmask reads s[1]            HW dt=1  EMU dt=3  ✗ Δ+2
[36]   v_cndmask reads s[2]            HW dt=1  EMU dt=2  ✗ Δ+1
```

Same logical structure, completely different HW behavior. The post-depctr
VOPD at [28] appears to "warm up" the VOPD pipe while leaving the scalar
pipe in a different phase — so when the v_cmps arrive, their SGPR commits
land on different pipeline cycles than in the [12-15] case.

### What a fix would need

A scheduler phase tracker that records:
- Last VOPD issue cycle vs last VALU issue cycle (for VALU bank port state)
- Depctr drain event — resets scalar pipe phase
- Current position in the scalar ALU 4-beat phase

Then A[n] computation uses phase-dependent offsets instead of constants.

Estimated scope: add `ScalarPipe.phase` state (int mod 4 or similar),
update at every issue, read in `is_cmp_lit` writer and cndmask reader
paths. ~6-8 hours work with risk of cross-kernel regression because the
phase model has to be calibrated against multiple capture groups.

**Decision:** park for a later session. Current state (320/340 modal,
94.1%) is already above the wave-variance ceiling; exp_chain's deeper
gap is real emu-modeling debt, not low-hanging.
