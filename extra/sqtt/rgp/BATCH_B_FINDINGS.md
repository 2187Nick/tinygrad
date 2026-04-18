# Batch B HW Findings (2026-04-18, Batch B2 capture)

192 microbenches × 16 waves each captured from real AMD 7900 XTX after the
v[100]-scratch LDS kernel hung the first B capture run. B.1 LDS remains
blacklisted; B.2/B.3/B.4/B.5 (94 kernels) succeeded.

## CLEAN VOPD findings — chained VOPDs cost 1cy regardless of RAW/bank

Modal dts across 16 waves per kernel, every VOPD pair:

| kernel                              | VOPD[1] dt | VOPD[2] dt | notes |
|-------------------------------------|------------|------------|-------|
| mb_vopd_pair_no_raw                 | 4          | **1**      | unanimous (16/16) |
| mb_vopd_pair_raw_x                  | 5          | **1**      | RAW VOPD[2].srcx0 = VOPD[1].vdstx |
| mb_vopd_pair_raw_y                  | 5          | **1**      | RAW on Y lane |
| mb_vopd_pair_raw_xy                 | 5          | **1**      | RAW on both lanes |
| mb_vopd_chain_n4_no_raw             | 1,1,1,1    | —          | unanimous (16/16) |
| mb_vopd_chain_n4_raw                | 5,1,1,1    | —          | unanimous |
| mb_vopd_bank_conflict_src           | 5,1        | —          | no bank penalty |
| mb_vopd_post_depctr                 | 1          | [depctr=3] 1 | — |

**Rule**: consecutive VOPDs pipe at **1cy**. The 4-5cy on the "first" VOPD is
the v_mov → VOPD transition cost, not the VOPD→VOPD occupancy. RAW dependencies
and bank conflicts DO NOT add cycles to the next VOPD.

**This falsifies the `_VOPD_PIPE_CYCLES = 4` constant** in emu.py:153. The correct
value is 1 for consecutive VOPDs. My earlier experiment flipping it to 1 regressed
exp_chain by 2 tokens — the regression must come from a different mechanism
(possibly VOPD→non-VOPD transition cost, which the emu conflates with VOPD→VOPD).

## Trans pipe RAW findings

| kernel                  | v_exp dt | v_log dt | notes |
|-------------------------|----------|----------|-------|
| mb_trans_raw_exp_log    | 1        | **4**    | RAW on v[1]; NOT 31cy pipeline latency |
| mb_trans_raw_valu       | 1        | (v_add=?)| |

**Rule**: trans→trans RAW on same VGPR costs **4cy**, not 31cy. The 31cy
`_TRANS_PIPELINE_LATENCY_SQRT` only applies when a *non-trans* VALU reads a
trans-written VGPR. Trans→trans uses internal ALU forwarding.

## Implications for exp_chain 12 mismatches

Current emu predicts `_VOPD_PIPE_CYCLES = 4` (or +bank). Corrected per-HW:
- Independent VOPD→VOPD: 1cy
- RAW VOPD→VOPD: 1cy  
- Bank-conflict VOPD→VOPD: 1cy
- First VOPD after non-VOPD: 4-5cy (context-dependent — probably v_mov vs cndmask vs depctr)

The exp_chain mismatches in MISMATCH_ANALYSIS.md §C were characterized as:
- [26]: HW=1 EMU=3 → first VOPD after depctr-drain. Emu over-predicts.
- [37]: HW=3 EMU=1 → VOPD after cndmask. Emu under-predicts.
- [38]: HW=2 EMU=4 → 2nd VOPD in pair. Emu over-predicts.

With the HW-confirmed rule (chain=1, first=context-sensitive), [26] predicts 1 (✓),
[38] predicts 1 (✗, HW=2). So the model still isn't quite right; there's a 2cy
effect on the 2nd VOPD in specific contexts. Might be VCC-implicit dependency
(VOPD writes VCC) or trans pipe state.

## Why the earlier VOPD=1 experiment regressed

When I set `_VOPD_PIPE_CYCLES = 1` uniformly, exp_chain regressed −2 tokens
because:
- [26]: EMU was 3, became 1 (now matches HW=1) ✓
- But [38]: EMU was 4, became 2 (HW=2) ✓
- But [37]: EMU was 1, stayed 1 (HW=3, still mismatches)

Actually looking again that *would* improve [26] and [38], so why was the net −2?

Likely the regression is in OTHER kernels (probe_cmp_chain, data_deps) where
the +4 VOPD spacing currently accidentally matches some HW token. Need
per-kernel diff to confirm.

## Recommended next step

**Land `_VOPD_PIPE_CYCLES = 1` AFTER completing Steps 3-6 of EMU_REWRITE.**
The refactor isolates VOPD timing in `VAluPipe` so a constant change doesn't
cross-perturb other subsystems. Until then, the 4-vs-1 decision keeps ping-pong
regressions.

Alternatively, **in Step 7** add a flag `prev_inst_was_vopd` tracked in
`VAluPipe` — gap is 1 when True, else 4. That handles both "first VOPD of a
chain" (non-VOPD predecessor → 4) and "chained VOPDs" (1).

## Calibrated new constants for TimingConstants

```python
VOPD_CHAIN_PIPE_CYCLES: int = 1   # VOPD→VOPD back-to-back (Batch B confirmed)
VOPD_FIRST_AFTER_VMOV: int = 4    # first VOPD after v_mov seed chain (approximate)
VOPD_FIRST_AFTER_CNDMASK: int = 3 # first VOPD after cndmask (exp_chain [37])
TRANS_RAW_SAME_VGPR: int = 4      # trans→trans RAW on same VGPR (mb_trans_raw_exp_log)
```
