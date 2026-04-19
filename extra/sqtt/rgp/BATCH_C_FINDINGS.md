# Batch C Findings (2026-04-19)

28 targeted microbenches captured across 16 waves each (4512 token-dts).
Goal: characterize the phase-state and VOPD-chain edge cases driving the
11 remaining reference-suite mismatches.

## Headline numbers

| Suite | Exact | ±2 |
|---|---|---|
| **Reference (340 tokens)** | **330/340 (97.1%)** | **339/340 (99.7%)** |
| **Full microbench (A+B+C)** | **33175/39960 (83.0%)** | **37507/39960 (93.9%)** |
| Batch A+B alone | 12410/14876 (83.4%) | 13781/14876 (92.6%) |
| Batch C alone | **468/564 (83.0%)** | **548/564 (97.2%)** |

## Key HW rule confirmed: VOPD MOV-only chains at 1cy

`mb_vopd_dualmov_sgpr_{pair,chain_n4}` and `mb_vopd_dualmov_{lit_pair,
all_lit_chain_n4}` all show 16/16 waves with **dt=1cy unanimous** for
consecutive VOPD V_DUAL_MOV_B32 (both lanes MOV). Applies regardless of
whether sources are SGPRs, literals, or mixed VGPR/literal.

Mixed variants (`mb_vopd_{mov_add,add_mov,literal_then_vgpr,vgpr_then_literal}_mix`):
also 1cy on transitions.

**Emu was predicting 4-5cy** because:
1. V_DUAL_MOV_B32 doesn't actually read vsrc1/vsrcy1, but the VOPD
   encoding slot is dummy-filled with `v[0]` by the assembler.
2. The decoder reports these dummies in `vgpr_r_regs`, so the bank-
   conflict check (+1cy) fires spuriously.
3. Default `_VOPD_PIPE_CYCLES = 4` is used for non-LIT VOPDs.

**Fix landed (commit `20340a3fb`):** decoder passes `vopd_mov_only=True`
in kwargs when both lanes are V_DUAL_MOV_B32, and the consumer uses
`last_vopd_issue + 1` with bank-conflict skip. Result: `where [8]` closed,
+1 exact on reference suite (329→330/340, 96.8%→97.1%).

## Per-category Batch C results

### C.1 — VOPD no-VGPR-read pairs (8 kernels)

| Kernel | exact % | notes |
|---|---|---|
| mb_vopd_dualmov_sgpr_pair | 59.2% | HW=1cy chain; emu was 5cy (now fixed) |
| mb_vopd_dualmov_sgpr_chain_n4 | 48.4% | same chained 4× |
| mb_vopd_dualmov_lit_pair | 92.6% | VOPD_LIT already correct (1cy) |
| mb_vopd_mov_add_mix | 77.3% | MOV→ADD transition 1cy |
| mb_vopd_add_mov_mix | 88.7% | after fix |
| mb_vopd_dualmov_all_lit_chain_n4 | 93.6% | all LIT (already correct) |
| mb_vopd_literal_then_vgpr | 85.5% | LIT→ADD 1cy |
| mb_vopd_vgpr_then_literal | 85.3% | ADD→LIT 1cy |

### C.2 — Depctr→cmp→cndmask→VOPD phase-state (6 kernels)

| Kernel | exact % | notes |
|---|---|---|
| mb_c2_depctr_cmp2_cnd2_vopd | 83.8% | 2-cmp+2-cndmask chain |
| mb_c2_depctr_cmp3_cnd3_vopd | 85.3% | 3-cmp+3-cndmask chain |
| mb_c2_depctr_cmp4_cnd4_vopd | 91.2% | 4-cmp+4-cndmask (canonical exp_chain pattern) |
| mb_c2_nodepctr_cmp4_cnd4_vopd | 95.1% | **control** — no depctr, best match |
| mb_c2_depctr_cmp_vcc_cnd_vopd | 85.4% | VCC-first like exp_chain [29-37] |
| mb_c2_depctr_vopd_pair_after_cnd | 79.7% | VOPD-pair test for post-warmup gap |

The no-depctr control (95.1%) vs depctr variants (83-91%) confirms the
phase-state rules matter. The depctr variants still have 1-2cy gaps
that the current +3 phase offset + GAP=1 + VOPD+2 rules don't fully
capture — likely precise position-dependent adjustments.

### C.3 — Post-idle VOPD (6 kernels)

| Kernel | exact % | notes |
|---|---|---|
| mb_c3_vmem_wait_then_vopd | 79.7% | vmcnt wait then VOPD |
| mb_c3_snop_long_then_vopd | 92.4% | nop(15) then VOPD |
| mb_c3_snop_short_then_vopd | 93.2% | nop(3) then VOPD |
| mb_c3_cndmask_after_idle_then_vopd | 89.7% | cndmask+wait+VOPD |
| mb_c3_vopd_wait_vopd_chain | 66.1% | VOPD+wait+4×VOPD — chained 1cy unmatched |
| mb_c3_depctr_then_vopd_fresh | 86.4% | depctr+VOPD fresh |

`mb_c3_vopd_wait_vopd_chain` worst performer (66.1%) — HW shows 4×VOPD
at dt=[1,1,1,1] after wait. Emu predicts 4cy for first post-wait VOPD.
This is the same "chained VOPDs = 1cy" pattern but we only fixed MOV-only.
The general VOPD-chain-after-VGPR-read case (like V_DUAL_MUL) is still
off — would need broader VOPD pipe model update (risky, per earlier
reverts in session 2026-04-18).

### C.4 — Cndmask taper in phase-shifted chain (4 kernels)

| Kernel | exact % | notes |
|---|---|---|
| mb_c4_depctr_chain_n1 | 81.0% | single cmp+cndmask |
| mb_c4_depctr_chain_n2 | 82.9% | 2-deep |
| mb_c4_depctr_chain_n3 | 84.6% | 3-deep |
| mb_c4_depctr_chain_n4_vcc_first | 95.3% | 4-deep VCC-first (best!) |

Accuracy improves with chain depth. Short chains (n1/n2) exhibit more
per-position variance that our phase-state heuristics don't yet capture.

### C.5 — b128 VMEM-store variants (4 kernels)

| Kernel | exact % | notes |
|---|---|---|
| mb_c5_b128_store_4mov_seed | 77.8% | 4× v_mov + b128 store |
| mb_c5_b128_store_vopd_seed | 60.6% | 2× VOPD + b128 store (worst) |
| mb_c5_b128_store_interleave | 83.0% | interleaved writes |
| mb_c5_b128_store_after_cndmask | 82.1% | mimics where [8-18] |

`mb_c5_b128_store_vopd_seed` worst at 60.6% — VOPD-written VGPRs feeding
b128 store have different forwarding latency than emu predicts. This is
exactly the `where [18] HW=21 EMU=25 Δ+4` pattern — VOPD→b128 store
over-predicts by 4cy, similar to V_ADDSUB→GSTORE wave-variance.

## Closed this session

- `where [8]` VOPD MOV chain: emu 5cy → 1cy (matches HW).

## Not yet closed — patterns Batch C revealed but would be risky to fix

1. **General VOPD chain at 1cy** (C.3 mb_c3_vopd_wait_vopd_chain, C.2
   mb_c2_depctr_vopd_pair_after_cnd). Requires dropping `_VOPD_PIPE_CYCLES`
   from 4 to 1 for chained VOPDs. Earlier attempts (2026-04-18 session)
   regressed exp_chain [51-52] — needs careful chain-state tracking.

   **Attempted 2026-04-19** (reverted): Added `post_drain_vopd_chain`
   flag on VAluPipe, set after waitcnt/depctr. VOPDs in this state use
   `last_vopd_issue + 1`. Result: closed `mb_c3_vopd_wait_vopd_chain`
   ±2 (83%→100%) but regressed exp_chain [51,52,54] because the
   VOPD_LIT→v_cmp transition after a long DRAM-staggered chain wants
   a 4cy tail that the chain rule prevents. Net: −2 exact (330→328).
   The right fix needs a "VOPD→non-VOPD transition tail" model that
   the current emu doesn't have — out of scope for this pass.

2. **b128 store forwarding from VOPD** (C.5 mb_c5_b128_store_vopd_seed,
   where [18]). VOPD-written VGPRs have different forwarding latency
   than plain VALU-written VGPRs (likely 17cy not 21cy). This is the
   `V_ADDSUB→GSTORE` wave-variance category documented previously —
   reference suite shows it's unfixable by constant change due to HW
   wave-0 vs wave-1 divergence.

3. **Short-chain cndmask taper in phase-shifted chain** (C.4 mb_c4_depctr_chain_n1/n2).
   Precise per-position timing (1st, 2nd, 3rd cndmask) differs from our
   GAP=1 heuristic. Closing these requires the per-chain state machine.

## Path to 100% (geohot's real target)

Current: 330/340 (97.1%) strict / 319/340 without MODAL.

Remaining 10 mismatches:
- **5 wave-variance** (probe_branch_cost W0/W1 opposite ±1, probe_sgpr_cmps [16],
  where [18], potentially one exp_chain case). These require a **stochastic
  wave-arbitration model** to close — not possible with constant
  adjustments. The HW wave-0 and wave-1 literally produce different dts
  for the same instruction; no deterministic emu can match both.
- **5 exp_chain phase-model interactions** ([26], [37], [54], [56], [57]).
  These are closable with more precise phase-model refinement, but each
  fix tends to shift another position — needs holistic re-design of
  the ScalarPipe phase state machine.

**Conclusion:** 97.1% is near the deterministic ceiling. The last 3pp
will need either wave-scheduler stochasticity or a wholesale scalar-pipe
state machine refactor.
