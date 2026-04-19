# Batch D Findings (2026-04-19)

18 surgical microbenches characterizing VOPD→non-VOPD transition tails
and VOPD→VOPD chain-continuation behavior. Captured across 16 waves
each (4166 token-dts).

## Headline rule: non-self-fwd VOPD chains at 1cy

**HW confirmed across all Batch D D.1/D.2/D.3 kernels (16/16 waves
unanimous except D.3 where some show ±2):**

| Pattern | HW gap |
|---|---|
| VOPD → v_cmp_e32 | **1cy** |
| VOPD → v_cmp_e64 | **1cy** |
| VOPD → v_cndmask (VCC) | **1cy** |
| VOPD → v_mov | **1cy** |
| VOPD → v_add | **1cy** |
| VOPD → s_mov (SALU) | **1cy** |
| VOPD → s_nop(3) | **6cy** (standard s_nop post-VALU path) |
| VOPD → s_waitcnt → store | wait + store |
| VOPD_LIT → v_cmp_e32 | **1cy** |
| VOPD_LIT → v_cmp_e64 | **1cy** |
| VOPD_LIT → v_mov | **1cy** |
| VOPD_LIT → v_add | **1cy** |
| 2× VOPD → v_cmp | **1cy** |
| 4× VOPD → v_cmp | **1cy** chained, all at 1cy |
| 2× VOPD_LIT → v_cmp | 2cy (mode 11/16, some 1cy) |
| 4× VOPD_LIT → v_cmp | **1cy** all chained |
| VOPD, VOPD_LIT, VOPD, VOPD_LIT → v_cmp | **1cy** |

## Discovered rule: self-fwd VOPDs break the 1cy chain

Chained VOPDs with **no self-fwd** (read and write **different** VGPRs)
pipeline at 1cy. Chained VOPDs with **self-fwd** (read and write the
**same** VGPR) force the next VOPD to wait 4cy because the register-
file bypass needs to settle.

Evidence:
- `mb_d3_vopd_chain4_then_vcmp` — writes v[20..27], reads v[4..7]
  (disjoint). HW 1cy chained, 16/16 waves. Emu was predicting 5cy.
- `exp_chain [16]→[17]` — both self-fwd on their own VGPRs
  (v[0],v[1] for [16]; v[2],v[3] for [17]). HW 4cy. This matches
  emu's existing `_VOPD_PIPE_CYCLES=4` prediction.

## Fix landed (commit `43f3b094b`)

In `_simulate_sq_timing` VOPD producer update block:
```python
_vopd_selffwd = (isinstance(vgpr_r_regs, (tuple, list)) and
                 isinstance(vgpr_w_regs, (tuple, list)) and
                 bool(set(vgpr_r_regs) & set(vgpr_w_regs)))
if is_vopd_lit:            _pipe_gap = 1
elif _vopd_paid_phase_warmup: _pipe_gap = 2
elif _vopd_selffwd:        _pipe_gap = _VOPD_PIPE_CYCLES  # 4
else:                      _pipe_gap = 1  # non-self-fwd chains at 1cy
```

## Impact

| Suite | Before | After | Δ |
|---|---|---|---|
| Reference (340 tokens) | 330/340 | **330/340** (unchanged) | 0 |
| Reference ±2 | 339/340 | 339/340 | 0 |
| Full microbench | 36885/44126 (83.0%) | **36915/44126 (83.7%)** | +30 exact |
| Full microbench ±2 | 41507/44126 (94.0%) | **42241/44126 (95.7%)** | +734 ±2 |
| Batch C alone ±2 | 548/564 (97.2%) | **564/564 (100.0%)** | +16 |
| Batch D alone | — | 3740/4166 (89.8%) | new |
| Batch D ±2 | — | 4110/4166 (98.7%) | new |

All 8 bounty tests still pass. The reference suite's 10 remaining
mismatches are unchanged — they're wave-variance and exp_chain phase
interactions, not VOPD transitions.

## Per-Batch-D-kernel results

See `/tmp/abcd_after.log` for full breakdown. Summary:
- D.1 (8 kernels): 1604/1787 (89.8%), range 78-93% exact
- D.2 (4 kernels): 815/879 (92.7%), tight range
- D.3 (6 kernels): 1321/1500 (88.1%), range 75-94%

Worst performers:
- `mb_d1_vopd_then_waitcnt` 78.3% — waitcnt emission details
- `mb_d3_vopd_chain4_then_vcmp` 75.1% (99.2% ±2) — deep-chain wave-variance
- `mb_d1_vopd_then_snop` 85.6% — s_nop family edge cases

## What Batch D didn't close

The exp_chain and wave-variance mismatches are unchanged. That's by
design — Batch D targeted VOPD tail behavior, which we now model better.
The remaining bounty gap needs:
- **ScalarPipe state machine** (5 exp_chain phase interactions)
- **Stochastic wave scheduler** (5 wave-variance mismatches)
