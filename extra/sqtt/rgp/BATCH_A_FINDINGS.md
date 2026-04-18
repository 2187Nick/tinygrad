# Batch A HW Findings (2026-04-18)

98 microbenches captured, 16 waves each (1024-thread dispatch). Modal dt across
waves per instruction gives a much sharper picture than the 2-wave probes.

Analysis via `extra/sqtt/rgp/analyze_microbench.py`.

## Critical insight — some "mismatches" are wave-level variance

| kernel | wave-0 dt (old) | modal dt (16 waves) |
|---|---|---|
| `mb_snop_15_after_valu`          | 22  | **18** (14/16) |
| `mb_snop_15_chain_n3` [6] first  | 26  | **18** (15/16) |
| `mb_snop_15_chain_after_vmcnt_n3`| 16  | **16** (16/16) ✓ |
| `mb_snop_15_after_waitcnt_vmcnt` | 20  | **16** (14/16) |

The prior session's "s_nop(15) after vmcnt = 20" from `PROBE_FINDINGS.md` §B
was **wave-level noise** — modal across 16 waves is 16. The 20 that shows up
in probe_sgpr_cmps's 2-wave captures is a tail of the distribution, not a
systematic rule. Fixing the emulator to always predict 20 here would regress
the vast majority of waves.

**Implication for the 28 mismatches:** some fraction (likely 4 of the 4 s_nop
±4 cases) are wave-variance that the emulator *cannot* match deterministically
unless we model the full stochastic scheduler. Best realistic bound for
emulator accuracy against a specific 2-wave capture is ≈325/321 − σ(variance).

## Cleanly verified HW rules

### s_nop(N) cost

| predecessor | position | HW dt modal | formula |
|---|---|---|---|
| v_add (VALU)    | solo           | **18**             | N + 3 |
| v_add (VALU)    | chain-first    | **18** (n3,n5 agree) | N + 3 |
| nop (in chain)  | chain-middle   | **16**             | N + 1 |
| nop (in chain)  | chain-last     | **16** (no +4!)    | N + 1 |
| s_waitcnt_vmcnt (drained 2×) | solo | **16** | N + 1 |
| s_waitcnt_vmcnt (drained)    | chain | all **16** | N + 1 |
| s_waitcnt_depctr             | solo  | **16** | N + 1 |
| s_waitcnt() (empty)          | solo  | **16** | N + 1 |
| s_mov (SALU)   | solo           | **3**              | N + 3 |

**Rule:** `s_nop(N)` stamps at `N + 1` after any drain event, and `N + 3`
after a non-drain predecessor (VALU, SALU). The current EMU uses `N + 2`
for non-drain, so it's off by 1 in the non-drain case. The `last-nop-in-chain +4`
hypothesis from MISMATCH_ANALYSIS.md is **falsified** — last nop costs 16,
same as middle nops.

### s_cbranch_scc1 cost is successor-sensitive too

| predecessor chain | successor | HW dt modal |
|---|---|---|
| tight `s_mov → s_cmp`                    | VALU (v_add)     | **8** (probe_scalar_beat_p0) |
| tight `s_mov → s_cmp`                    | VMEM (store)     | **12** (mb_salu_scmp_tight) |
| nop-broken `s_mov → nop → s_cmp`         | VALU             | **13** (probe_scalar_beat_p1) |
| nop-broken `s_mov → nop(×3) → s_cmp`     | VMEM             | **9** (mb_salu_scmp_spaced_nop0x3) |

So the cost isn't a simple "tight 8 / cold 13" — it's `f(predecessor, successor)`.
The 4 values form a 2×2 grid, and the emulator's constant 9 is the average.

Fixing this cleanly requires modeling the scalar-pipe phase state (what the
`EMU_REWRITE_DESIGN.md` §1.4 `ScalarPipe` subsystem is designed to do). A
constant-only fix would regress in one cell while improving in another.

### VOPD independent chain — dt=1 per VOPD confirmed

`mb_vopd_fmac_mul_n4` and `mb_vopd_mixed_n4` both show 4 consecutive VOPDs
with RAW-independent srcs at dt=1 each. This confirms `VOPD_INDEP_PIPE_CYCLES=1`
from PROBE_FINDINGS.md §C.

**Caveat:** the 4th VOPD + successor shows dt=3 (mode 11/16 for the following
`s_waitcnt()`). So VOPD→scalar transition pays a 3cy tail. Not a VOPD→VOPD cost.

### VALU RAW chain — NOT simply 1cy

`mb_valu_add_n8` (8 RAW `v_add_f32(v[1], 1.0, v[1])`):

| idx | modal dt | count/16 |
|---|---|---|
| [5] | 1 | 3/16 |
| [6] | 5 | 12/16 |
| [7] | 5 | 12/16 |
| [8] | 5 | 12/16 |
| [9] | 5 | 12/16 |
| [10] | 1 | 16/16 |
| [11] | 5 | 12/16 |
| [12] | 5 | 14/16 |

Mixed pattern — most positions show **5cy RAW stall**, but position [5] and
[10] show 1cy. Hypothesis: the v_add that reads a *just-written* VGPR pays
the 5cy VALU pipeline latency, but the first v_add reading the prologue-loaded
v[1] (already 5cy old) pays only 1cy. Position [10] is similarly a "warm"
case — maybe the prior v_add at [9]'s effect has aged past 5cy.

EMU appears to currently handle this correctly via `valu_dep_stall` for
RAW deps — the mismatches don't fall in VALU chains, so no change needed.

### Trans chain — front-loaded

`mb_trans_exp_n4` (4 back-to-back `v_exp_f32 v[1],v[1]`):

| idx | modal dt |
|---|---|
| [5] | 1 (first exp) |
| [6] | **14** (second exp with RAW dep) |
| [7] | **14** |
| [8] | **4** (fourth) |

Not the expected `[1, 4, 4, 4]` — HW shows `[1, 14, 14, 4]`. The 14cy gap
suggests TRANS-on-RAW-VGPR needs to wait for the full 13-cycle trans result
commit before re-reading. Only the last exp drops to 4cy (maybe because by
then the prior result has propagated far enough).

Current EMU's `_TRANS_PIPELINE_LATENCY = 27` applies to trans→non-trans reads,
but trans→trans on same reg uses the `trans_pipe_avail` 4cy occupancy. Our
data suggests trans→trans RAW needs 14cy, not 4cy.

## Calibrated constants (data-driven)

| constant | old value | HW-measured | notes |
|---|---|---|---|
| NOP_POST_NON_DRAIN_EXTRA | +1 | **+2** | s_nop(N) after VALU = N+3, not N+2 |
| NOP_LAST_IN_CHAIN_EXTRA  | (hypothesized +4) | **0** | falsified |
| CBRANCH_COST             | 9 | **context-sensitive 8/9/12/13** | needs phase model |
| VOPD_INDEP_PIPE_CYCLES   | 4 (uniform) | **1** | confirmed for RAW-independent |
| TRANS_RAW_STALL          | 4 | **14** (positions 2–3 of chain) | new constant needed |

## Immediate emu fixes worth trying (post-refactor)

In order of confidence:

1. **s_nop post-non-drain +1 → +2**: line 370 of emu.py. Makes `s_nop(15) after v_add = 18` match HW. Risk: any 2-wave kernel that currently matches at EMU=17 will now predict 18 (HW is probably 18 in modal, so should improve or match).

2. **VOPD RAW-dep detection + dual gap**: only apply `_VOPD_PIPE_CYCLES=4` when there's RAW dep; 1cy otherwise. The fix regressed with simple reg-set intersection earlier because exp_chain's VOPDs have bank-level dep, not reg-level. Need bank-aware detection.

3. **Trans→trans RAW 14cy stall**: currently treated as 4cy occupancy. Would fix trans-chain mispredict, but may regress chains where the VGPR isn't actually read-back.

4. **CBranch phase model**: needs full ScalarPipe rewrite from EMU_REWRITE_DESIGN.md §1.4.

## Next steps

1. **Attempt fix #1 (s_nop +1→+2)** with full baseline regression. Low-risk isolated change.
2. **Run Batch B (resource conflicts)** to get RAW-dep VOPD data for fix #2.
3. **Proceed with EMU_REWRITE Step 1** (constants dataclass extraction) — zero-risk refactor that sets up cleaner fix-landing.
