# Microbench Accuracy Sweep (2026-04-19)

Ran `MOCKGPU=1 MICROBENCH=1 rigorous_hw_test.py --compare` against
the full Batch A + Batch B microbench suite (~290 kernels, 32746 tokens,
16 waves each). MODAL mode default.

## Aggregate

- **Total: 26883/32746 exact (82.1%), 30381/32746 ±2 (92.8%)**
- 34 of 100 ranked kernels below 80% exact
- 19 of 100 below 70%

For comparison, the curated 340-token reference suite hits **95.0% exact**.
The microbenches are stress tests designed to probe edge cases, so 82%
here is meaningful signal about emu coverage, not a regression.

## Bottom-25 (hot targets for future improvement)

| Kernel | Exact | ±2 |
|---|---|---|
| mb_valu_add_n2 | 50.0% | 66.7% |
| mb_vcmp_cndmask_k1 | 57.1% | 100.0% |
| mb_salu_smov_n1 | 60.0% | 80.0% |
| mb_trans_exp_n1 | 60.0% | 82.5% |
| mb_trans_log_n1 | 60.0% | 82.5% |
| mb_trans_rcp_n1 | 60.0% | 82.5% |
| mb_trans_rsq_n1 | 60.0% | 62.5% |
| mb_trans_sqrt_n1 | 60.0% | 82.5% |
| mb_vcmp_literal | 60.0% | 80.0% |
| mb_vcmp_vcc_n1 | 60.0% | 60.0% |
| mb_sanity_n3 | 60.6% | 76.0% |
| mb_nop1_solo | 64.6% | 83.3% |
| mb_nop0_solo | 66.7% | 100.0% |
| mb_nop5_solo | 66.7% | 83.3% |
| mb_sanity_n2 | 66.7% | 83.3% |
| mb_vcmp_sgpr_n1 | 66.7% | 100.0% |
| mb_sanity_n4 | 67.0% | 67.0% |
| mb_valu_add_n4 | 67.5% | 81.6% |
| mb_vcmp_cndmask_k2 | 69.4% | 100.0% |
| mb_trans_raw_valu | 71.1% | 71.1% |
| mb_trans_then_salu | 71.4% | 85.7% |
| mb_vopd_mixed_n4 | 74.0% | 87.0% |
| mb_cndmask_vcc_n1 | 75.0% | 87.5% |
| mb_lds_store_b64_n1 | 75.0% | 87.5% |
| mb_salu_smov_n4 | 75.0% | 75.0% |

## Patterns

### Systematic 50-60% on n1 (single-instance) probes

mb_trans_{exp,log,rcp,rsq,sqrt}_n1 all at 60%. These have a simple
prologue → single trans op → global_store structure. A systematic 1cy
offset on the store-after-valu path (see V_ADDSUB→GSTORE wave-variance
analysis) causes half the waves to mismatch. Many of these are ±2 >90%,
so they're close misses.

### Good: chained n4/n8 probes

mb_valu_add_n8 80%, mb_vopd_chain_n4 74-79%, mb_trans_after_trans_8
92.5%. Longer chains amortize per-inst noise and modal rescue more cases.

### Not-actionable: mb_trans_raw_with_depctr (61.4%)

Worst trans-related probe. HW `mb_trans_raw_valu` (71%) pattern suggests
the trans→VALU path has residual state we don't model (see "new direction
for the 28" in STEP7_STATUS.md about trans RAW positional). A phase-state
model would help here too.

## Implications for Batch C design

Most low-performers are already in Batch A/B. New microbenches wouldn't
help unless they probe NEW patterns that we haven't already captured.
The 17 mismatches in the reference suite concentrate on:

1. **exp_chain phase-state** — has no microbench equivalent currently.
   Batch C should include `mb_depctr_vopd_cmp_cndmask_{n2,n4,n8}` to
   calibrate the phase offset precisely (currently using +3 heuristic).
2. **VOPD-adjacent-to-cndmask** — present in `mb_vopd_vcc_producer_then_cndmask`
   (94%) and `mb_vopd_cndmask_*` (84-90%). These could be sharpened.
3. **b128 VMEM forwarding** — `mb_lds_store_b128_n1` 79.9%, `mb_lds_load_b128_n1`
   100% (so read is fine). Store side has gaps.

Batch C is **not** the highest-leverage next step. Extending the scalar-
pipe phase model directly (Task #22) is higher-leverage because it
closes exp_chain mismatches AND would improve `mb_trans_raw_with_depctr`
and `mb_vopd_post_depctr` families in one shot.
