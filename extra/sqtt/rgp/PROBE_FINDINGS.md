# Probe HW Findings (2026-04-18)

Parametric probe kernels captured on real AMD 7900 XTX, analyzed in
extra/sqtt/rgp/analyze_probes.py. Replaces / extends the
hypotheses in MISMATCH_ANALYSIS.md with measured values.

## Captures obtained

18 / 21 new probes captured (wave 0 only for most — see "retry" at bottom):

| family        | probes captured               |
| ------------- | ----------------------------- |
| cold_start    | n2, n4 (n8 did not land)       |
| nop_chain     | n1, n3, n5                     |
| store         | cold, warm                     |
| trans_pair    | tight (spaced did not land)    |
| scalar_beat   | p0, p1, p2, p3                 |
| vopd          | chain, split (nodep did not land) |

## HW-confirmed patterns

### B: s_nop(15) cost = 20 cycles when preceded by s_waitcnt_vmcnt

```
probe_nop_chain_n1: waitcnt_vmcnt  →  s_nop(15) dt=20  →  v_add dt=1
probe_nop_chain_n3: waitcnt_vmcnt  →  s_nop(15) dt=20×3  → v_add dt=1
probe_nop_chain_n5: waitcnt_vmcnt  →  s_nop(15) dt=20×5  → v_add dt=1
```

EMU currently gives 16 for the drain-predecessor nop chain. HW says 20.
`+4 cycles` per nop, no matter position in chain, when the chain follows a
VMEM drain.

Compare with probe_sgpr_cmps (captured earlier): trans pipe + s_waitcnt()
precedes the nop chain, and there middle nops DO cost 16 (emu matches):

```
probe_sgpr_cmps [20] s_waitcnt() dt=3          (after v_sqrt trans)
                [21] s_nop(15)   dt=16 (w0) or 20 (w1-slip)
                [22] s_nop(15)   dt=16
                [23] s_nop(15)   dt=20   ← last before VALU
                [24] v_cmp       dt=1
```

So the s_nop(15) cost is 20 if the IB had to re-fetch after a VMEM-drain
stall OR the nop is the LAST in a chain before a non-nop. It's 16 only for
middle-of-chain nops in a trans-drained context.

### C: consecutive independent VOPD = 1 cycle

```
probe_vopd_chain: VOPD(v4,v5, r=v6…9) dt=1 × 4   (4 back-to-back VOPDs)
probe_vopd_split: VOPD dt=1, VOPD dt=1, depctr dt=3, VOPD dt=1, VOPD dt=1
```

EMU currently uses `_VOPD_PIPE_CYCLES = 4` (line 153 emu.py), producing 5 cy
spacing between consecutive non-LIT VOPDs. HW shows 1 cy when the next VOPD
has no RAW dependency on the previous one.

**Attempted fix that regressed** (reverted): gate `_VOPD_PIPE_CYCLES` on
source/dest register overlap with the previous VOPD. This drops the emu's
cost to 1cy for non-dependent VOPDs but also drops it for exp_chain's VOPDs,
which are labeled "independent" by reg analysis but HW still spaces them at
3 cycles. exp_chain regressed −2 tokens. Root cause not yet isolated —
exp_chain's VOPDs seem to have a dependency path the simple reg-set check
misses (maybe VCC, maybe a VGPR bank port, maybe the trans pipe state).

### E: scalar pipe — s_cbranch cost is 8 cy (tight) or 13 cy (cold)

```
probe_scalar_beat_p0: s_mov(s4,0) dt=1 → s_cmp dt=1 → s_cbranch_scc1 dt=8
probe_scalar_beat_p1: s_mov dt=1 → s_nop() dt=3 → s_cmp dt=1 → s_cbranch dt=13
probe_scalar_beat_p2: s_mov dt=1 → s_nop()×2 dt=3,1 → s_cmp dt=1 → s_cbranch dt=13
probe_scalar_beat_p3: s_mov dt=1 → s_nop()×3 dt=3,1,1 → s_cmp dt=1 → s_cbranch dt=13
```

EMU uniformly predicts `s_cbranch` dt=9. HW:
- Tight chain (`s_mov → s_cmp → s_cbranch`): 8 cy, a fast-path through the
  scalar issuer.
- Any nop between `s_mov` and `s_cmp`: 13 cy, the slow path.

Secondary finding: `s_mov → s_nop(0)` is 3 cy in HW (EMU predicts 1). The
"scalar issuer warm-up" cost applies consistently across all three p1/p2/p3
probes regardless of how many nops follow.

### Falsified: VMEM store cold/warm bypass

A1/A2 hypothesis predicted `store_cold` at 21 and `store_warm` at 17 (the
elusive "scoreboard warm" bypass). Actual captures:

```
probe_store_cold: v_add dt=1  →  global_store_b32 dt=34
probe_store_warm: v_add dt=1  →  global_store_b32 dt=31   (had prior load)
```

Both are much higher than the 17/21 seen in the baseline probes — because
the test kernel ends right after the store, so the store dt also swallows
whatever IB-drain / endpgm-prepare cost follows. The A1/A2 hypothesis cannot
be tested with this kernel shape; it would need a probe that does real work
*after* the store (e.g., another global_store chain).

### Not reachable: D (wave-1 cold-start slip) and F3 (wave-pair TRANS)

Captures landed only wave 0 for every probe. Wave-1 slip is still the
leading hypothesis for 5 of the 28 mismatches. To validate, re-run capture
with more retries for the probes and/or larger dispatch so at least one
workgroup lands its wave 1 on the traced CU. Alternative: force many
WGs (`Tensor.empty(1024)`) in the _run_probe helper.

## Emu-fix status

Three fixes attempted, all reverted after regression:
- `_VOPD_PIPE_CYCLES = 1` → exp_chain −2
- RAW-dep-aware VOPD gap (1 if no dep else 4) → exp_chain −2
- (none landed) `s_nop(15) +4 for last-in-chain` — would fix 2 mismatches
  but regresses probe_cmp_chain w0[21] (isolated nop, currently matches at
  EMU=22) because the +4 would add to the SGPR drain path, giving EMU=26.

Every remaining mismatch sits on a coupled timing edge: moving the constant
to match one kernel breaks another that's compensating via a different
path. The realistic path forward is a more structured rewrite of the
s_nop / VOPD / scalar-pipe cost functions that separates the overlapping
effects, not more constant-tuning.

## Suggested next steps

1. **Re-capture the 3 missing probes** with more retries / larger dispatch:
   `probe_cold_start_n8`, `probe_trans_pair_spaced`, `probe_vopd_nodep`.
   Bump `max_attempts` from 30 to 100 in rigorous_hw_test.py or dispatch
   `Tensor.empty(1024)` to boost the chance waves land on traced units.
2. **Add wave-1 fixture for the new probes** — the current _run_probe uses
   `Tensor.empty(64)` which gives 2 waves per WG but only 1 WG, and the
   traced CU only sees 1 of the 2 waves most of the time. Try size 512
   (8 WGs) to saturate more traced units.
3. **Restructure emu.py s_nop/VOPD cost** — current code couples the drain
   stamp, SGPR drain, IB-resume penalty, and bank-port timing through
   shared variables. Separating these would let us fit the +4 for "last
   nop before VALU" independently of the +6 for "post-vcmp SGPR drain".
4. **Design a dep-aware VOPD probe** that forces a RAW dep on v[0]/v[1]
   across a chain (what exp_chain actually does) and measures the dt. That
   tells us what the correct non-1 value is, and whether the dep is on
   regs, banks, or something else.
