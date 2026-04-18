# Mismatch Analysis: 28 tokens at 91.3% (293/321)

Captured from a live `--compare` run (2026-04-18) against `extra/sqtt/captures/rigorous/*.pkl` (AMD 7900 XTX, 2-wave WG, WGP-mode, sibling waves on *different* SIMDs per RGP disassembly).

The 28 mismatches are regrouped here relative to the task brief because two of the task-listed indices don't exist in the current pkl files (probe_vmem_chain is only 13 tokens long; the +4 stores are in probe_cmp_chain/probe_branch_cost/data_deps) and the current emulator's mismatch set is slightly different from what the brief listed. Real set, verified against the comparator output:

| kernel            | wave | idx | HW | EMU | diff | inst                                      | group |
| ----------------- | ---- | --- | -- | --- | ---- | ----------------------------------------- | ----- |
| data_deps         | 1    | 6   | 17 | 21  | +4   | `global_store_b32`                        | A     |
| probe_cmp_chain   | 0    | 23  | 17 | 21  | +4   | `global_store_b32`                        | A     |
| probe_branch_cost | 0    | 14  | 17 | 21  | +4   | `global_store_b32`                        | A     |
| probe_cmp_chain   | 1    | 21  | 18 | 22  | +4   | `s_nop(15)`                               | B     |
| probe_sgpr_cmps   | 0    | 23  | 20 | 16  | -4   | `s_nop(15)`                               | B     |
| probe_sgpr_cmps   | 1    | 21  | 20 | 16  | -4   | `s_nop(15)`                               | B     |
| probe_sgpr_cmps   | 1    | 23  | 20 | 16  | -4   | `s_nop(15)`                               | B     |
| exp_chain         | 0    | 26  | 1  | 3   | +2   | `VOPD MUL_MUL v0/v1`                      | C     |
| exp_chain         | 0    | 31  | 3  | 4   | +1   | `v_cmp_gt_f32_e64 s[1]`                   | C     |
| exp_chain         | 0    | 34  | 4  | 1   | -3   | `v_cndmask_b32_e64 v[5]`                  | C     |
| exp_chain         | 0    | 35  | 1  | 3   | +2   | `v_cndmask_b32_e64 v[6]`                  | C     |
| exp_chain         | 0    | 36  | 1  | 2   | +1   | `v_cndmask_b32_e64 v[7]`                  | C     |
| exp_chain         | 0    | 37  | 3  | 1   | -2   | `VOPD MUL_MUL v0/v1`                      | C     |
| exp_chain         | 0    | 38  | 2  | 4   | +2   | `VOPD MUL_MUL v2/v3`                      | C     |
| exp_chain         | 0    | 40  | 2  | 1   | -1   | `v_log_f32_e32 v[0]`                      | C     |
| exp_chain         | 0    | 54  | 1  | 4   | +3   | `v_cmp_gt_f32_e64 s[1]`                   | C     |
| exp_chain         | 0    | 57  | 3  | 1   | -2   | `v_cndmask_b32_e64 v[5]`                  | C     |
| exp_chain         | 0    | 58  | 1  | 3   | +2   | `v_cndmask_b32_e64 v[6]`                  | C     |
| exp_chain         | 0    | 61  | 3  | 1   | -2   | `VOPD MUL_MUL v0/v1`                      | C     |
| probe_vmem_chain  | 1    | 2   | 4  | 1   | -3   | `v_lshlrev_b32_e32(v[0], 2)` after waitcnt| D     |
| probe_branch_cost | 0    | 7   | 8  | 9   | +1   | `s_cbranch_scc1(9)`                       | E     |
| probe_branch_cost | 1    | 5   | 2  | 1   | -1   | `s_mov_b32(s[4], 0)`                      | E     |
| probe_branch_cost | 1    | 7   | 10 | 9   | -1   | `s_cbranch_scc1(9)`                       | E     |
| probe_sgpr_cmps   | 0    | 16  | 2  | 1   | -1   | `v_cndmask_b32_e64(v[9]…)`                | F     |
| probe_sgpr_cmps   | 1    | 8   | 2  | 1   | -1   | `v_mov_b32_e32(v[5], 4.0)`                | F     |
| probe_sgpr_cmps   | 1    | 16  | 5  | 1   | -4   | `v_cndmask_b32_e64(v[9]…)`                | F     |
| probe_sgpr_cmps   | 1    | 18  | 10 | 4   | -6   | `v_log_f32_e32(v[10], …)`                 | F     |
| probe_cmp_chain   | 1    | 6   | 2  | 1   | -1   | `v_mov_b32_e32(v[3], 2.0)`                | F     |

Total: 28 rows. All of F+D (and part of B, E) share one root cause that pops out as soon as you line up sibling waves on a common time axis. See the cross-cutting observation below.

---

## Cross-cutting observation: wave skew accumulates at long-latency ops

In every kernel that has two waves, the two waves start within ~2-15 cycles of each other and run in lock-step through short VALU — but at each **long-latency or resource-shared instruction** the later wave loses a few cycles relative to the earlier one. The gap then never shrinks (both waves proceed in parallel again, but with a bigger skew).

probe_sgpr_cmps is the clearest example — here is `wave1_cycle − wave0_cycle` per token:

```
  idx   Δ   inst                                 comment
  [ 0]  10  s_load_b64                           initial dispatch skew
  [ 1]   2  s_waitcnt_lgkmcnt                    scalar cache done roughly together
  [ 2]   2  v_lshlrev_b32_e32(v[0], 2)           first VALU — lockstep
  [ 3]   2  global_load_b32                      VMEM — lockstep
  [ 4]  18  s_waitcnt_vmcnt(NULL)                DRAM return ordered
  [ 5..7] 18  v_mov_b32_e32 x3                   lockstep resumed
  [ 8]  19  v_mov_b32_e32(v[5], 4.0)             +1 slipped here
  [ 9..15] 19  v_cmp_gt/cndmask chain            lockstep
  [16]  22  v_cndmask_b32_e64(v[9], …, s[6])     +3 slip
  [17]  22  v_exp_f32_e32                        (trans pipe)
  [18]  28  v_log_f32_e32                        +6 slip (TRANS)
  [19..20] 28  v_sqrt_f32_e32, s_waitcnt        stable
  [21]  32  s_nop(15)                            +4 slip (first nop in chain)
  [22..34] 32  remainder of the wave            stable
```

Every "wave 1 slips" event lines up 1-to-1 with a reported emulator mismatch:
- `[8] +1` → F `probe_sgpr_cmps w1 [8] v_mov 4.0 HW=2 EMU=1`
- `[16] +3` → F `probe_sgpr_cmps w1 [16] cndmask(s[6]) HW=5 EMU=1`
- `[18] +6` → F `probe_sgpr_cmps w1 [18] v_log HW=10 EMU=4`
- `[21] +4` → B `probe_sgpr_cmps w1 [21] s_nop(15) HW=20 EMU=16`

And wave 0 reports the **same** instruction with a normal delta (EMU-matching) at each of those indices:

```
  probe_sgpr_cmps w0 [ 8] dt=1 v_mov_b32_e32(v[5], 4.0)
  probe_sgpr_cmps w0 [16] dt=2 v_cndmask_b32_e64(v[9], …, s[6])   (already +1 slip vs wave 1)
  probe_sgpr_cmps w0 [18] dt=4 v_log_f32_e32                     (baseline)
  probe_sgpr_cmps w0 [21] dt=16 s_nop(15)                        (baseline EMU value)
  probe_sgpr_cmps w0 [23] dt=20 s_nop(15)                        (still 20 — see below)
```

So the "wave 1 slips" don't explain wave-0 mismatches. But the wave-0 mismatches (B `[23]=20`) correspond to the *last* nop in an s_nop(15) chain and look like a different bug (post-nop IB resume overhead).

This points at **two separate emulator bugs**:

1. **SIMD-pair resource contention** (D, F, and the wave-1 half of B) — the emulator treats the two waves as fully independent on different SIMDs. HW reality is that specific instruction classes (trans, long cndmask chains, long s_nop stalls) hold a shared resource that the other wave's issuer has to wait for.
2. **VALU→VMEM store bypass modeling** (A) and **s_nop(15)-in-chain cost tail** (B wave-0 half) — non-contention pipeline effects the emulator is just wrong on.

Each pattern below includes the token tables, hypothesis, and concrete probe-kernel designs.

---

## Pattern A — VMEM store bypass (4 mismatches)

### Tokens

Every "solo b32 store with VALU-writes-to-stored-reg preceding it" in the suite:

| kernel            | wave | idx | dt | what preceded                                  |
| ----------------- | ---- | --- | -- | ---------------------------------------------- |
| data_deps         | 0    | 6   | 21 | v_add_f32_e32(v[1], 1.0, v[1]) dt=1            |
| data_deps         | 1    | 6   | 17 | v_add_f32_e32(v[1], 1.0, v[1]) dt=1            |
| probe_cmp_chain   | 0    | 23  | 17 | v_add_f32_e32(v[1], 1.0, v[1]) dt=1            |
| probe_cmp_chain   | 1    | 23  | 21 | v_add_f32_e32(v[1], 1.0, v[1]) dt=1            |
| probe_branch_cost | 0    | 14  | 17 | v_add_f32_e32(v[1], v[10], v[1]) dt=1          |
| probe_branch_cost | 1    | 14  | 21 | v_add_f32_e32(v[1], v[10], v[1]) dt=1          |
| probe_sgpr_cmps   | 0    | 33  | 21 | v_add_f32_e32(v[1], v[6], v[1]) dt=1           |
| probe_sgpr_cmps   | 1    | 33  | 21 | v_add_f32_e32(v[1], v[6], v[1]) dt=1           |

Emulator produces 21 in all 8 cases. HW is **17 or 21**, bimodal, with a bias:
- data_deps, probe_cmp_chain, probe_branch_cost: one wave = 17, the other = 21.
- probe_sgpr_cmps: both = 21.

### Control: `plus`, `cast`, `elementwise` (all pass)

These write `global_store_b128` (four-DW stores) with dt=20, 20, 24 — no 17 available. So 17 is b32-specific.

### Absolute times (which wave is first?)

```
data_deps           w0: store@11007 endpgm@11258 (ep_dt=251)  store_dt=21
                    w1: store@11022 endpgm@11282 (ep_dt=260)  store_dt=17   <-- later wave issues with 17
probe_cmp_chain     w0: store@11949 endpgm@12195 (ep_dt=246)  store_dt=17   <-- earlier wave
                    w1: store@11987 endpgm@12238 (ep_dt=251)  store_dt=21
probe_branch_cost   w0: store@11118 endpgm@11368 (ep_dt=250)  store_dt=17   <-- earlier wave
                    w1: store@11139 endpgm@11393 (ep_dt=254)  store_dt=21
probe_sgpr_cmps     w0: store@12373 endpgm@12624 (ep_dt=251)  store_dt=21   <-- same!
                    w1: store@12405 endpgm@12654 (ep_dt=249)  store_dt=21   <-- same!
```

There's no stable "earlier wave gets 17" rule — data_deps reverses it. But the inter-wave gap in probe_sgpr_cmps is **32 cycles**, large enough that neither wave's store is anywhere near the other's — both get 21.

### Hypothesis A1 (primary)

**The 4-cycle delta is the `s_waitcnt_vmcnt`/VMEM-store scoreboard counter increment latency.** When the scoreboard already has an "in-flight" VMEM op that can merge slots with the new one, HW can issue the store in 17 cy; otherwise 21 cy.

Evidence for scoreboard interpretation:
- The three 17/21 cases each have a **prior VMEM op still in the VMEM pipeline** — specifically the wave-0 `global_load_b32` whose `s_waitcnt_vmcnt(NULL)` only *logically* drained but whose scoreboard counter took extra cycles to decrement. When the store issues while the counter is exactly at the write-side arbiter's "open" window (17) vs outside (21), you get either value.
- probe_sgpr_cmps has **no previous VMEM op** between [3] global_load (drained at [4]) and [33] store — over 2100 cycles elapsed, the scoreboard is fully quiescent for both waves and both pay the full 21.
- `data_deps`, `probe_cmp_chain`, `probe_branch_cost` all have exactly one prior `global_load_b32` whose scoreboard slot is *just* beginning to free when the store wants to issue.

### Hypothesis A2 (alternative)

**VMEM write arbiter contention between the two waves.** The two waves on different SIMDs still share the CU's VMEM write queue. When wave 0's last pre-store VALU wrote its payload register early enough that the store was already launched into the VMEM write queue, wave 1's subsequent store has to wait one slot (4 cy) for the queue to re-open; otherwise it slips in right behind. The "earlier" wave depends on per-run scheduling noise.

Falsifying test (see probe designs below): if A1 is right, a back-to-back (no prior VMEM) sequence should always give 21. If A2 is right, doubling the number of waves should make all stores cost 21.

### Probe-kernel designs

1. **`custom_probe_store_bypass_solo`** — identical prologue to `custom_data_deps` but remove the `global_load` so the only VMEM op in the wave is the store: 1 payload VALU → 1 store. Predict A1: dt = 21 always (no prior scoreboard slot to hit). Predict A2: dt = 17/21 depending on wave pair.

2. **`custom_probe_store_bypass_chain(N)`** — write N consecutive stores to distinct registers (`v[1]`…`v[N]`), each preceded by its own producing VALU one token earlier (interleave `v_add` → `store` pairs). Predict A1: store[0] pays 17/21, store[1..N-1] all chain at dt=1 (already in queue). Predict A2: every store pays individually until the write queue fills.

3. **`custom_probe_store_bypass_spacing(K)`** — insert K `s_nop` padding cycles between the VALU and the store. Sweep K=0..20. Predict: for K ≤ 17 the store dt = 21 − K down to ~4, at K = 17 dt ≈ 4 (cannot go below 4 = VMEM issue cost), at K = 21+ dt = 4. Fits the 17-cycle number as the VMEM store **first-issue latency**.

4. **`custom_probe_store_bypass_mix_load`** — issue a `global_load_b32` first, `s_waitcnt_vmcnt(NULL)`, then one VALU → store. Predict: dt = 17 (scoreboard still warm). Compare to same kernel without the preceding load — predict dt = 21. This directly isolates A1.

---

## Pattern B — `s_nop(15)` context sensitivity (4 mismatches)

### Tokens in context

probe_sgpr_cmps has a chain of three `s_nop(15)` back-to-back. Both waves:

```
  probe_sgpr_cmps w0 [18] dt=4   v_log_f32_e32      (TRANS)
                  [19] dt=4   v_sqrt_f32_e32     (TRANS)
                  [20] dt=3   s_waitcnt()
                  [21] dt=16  s_nop(15)          *first in chain, matches EMU*
                  [22] dt=16  s_nop(15)          *middle, matches*
                  [23] dt=20  s_nop(15)          *last before VALU, MISMATCH*
                  [24] dt=1   v_cmp_gt_f32_e32

  probe_sgpr_cmps w1 [18] dt=10  v_log_f32_e32      (TRANS — +6 slip vs wave 0)
                  [19] dt=4   v_sqrt_f32_e32
                  [20] dt=3   s_waitcnt()
                  [21] dt=20  s_nop(15)          *first in chain, MISMATCH +4*
                  [22] dt=16  s_nop(15)          *middle, matches*
                  [23] dt=20  s_nop(15)          *last before VALU, MISMATCH*
                  [24] dt=1   v_cmp_gt_f32_e32
```

probe_cmp_chain (single `s_nop(15)` after v_cmp chain):

```
  probe_cmp_chain  w0 [18..20] v_cmp_gt_f32_e64 s[9/10/11]  dt=1/1/1
                      [21] dt=22  s_nop(15)          *matches EMU=22*
                      [22] dt=1   v_add_f32_e32
  probe_cmp_chain  w1 [18..20] v_cmp_gt_f32_e64 s[9/10/11]  dt=1/1/1
                      [21] dt=18  s_nop(15)          *MISMATCH: EMU=22*
                      [22] dt=1   v_add_f32_e32
```

### Observations

- `s_nop(15)` costs **16, 18, 20, or 22** depending on surroundings. Emulator gives 16 (first nop), 16 (middle), 16 (last), or 22 (isolated nop after v_cmp chain).
- **Last nop in a 3-nop chain before a VALU always costs 20** (both waves). That's 4 extra cycles the emulator isn't charging.
- **First nop in chain** depends on wave: 16 in wave 0, 20 in wave 1. This tracks the wave-1 trans-chain slip (+6 at v_log), which means wave 1 enters the nop chain while some SGPR/trans-pipe resource is still busy.
- **Middle nop** is always 16 (`nop_cycles + 1` = 16 for N=15, baseline).
- **Isolated nop after v_cmp chain** is 22 (matches EMU) OR 18 (wave 1, −4). Emulator is right by accident for one wave, wrong for the other.

### Hypothesis B1 — "IB resume" adds +4 to the last nop before a VALU

The last nop before a non-nop instruction has to pay a **pipeline restart penalty** to start feeding VALU issue again. Middle nops in a chain don't pay it (the next nop is already queued). HW cost: `nop_cycles + 1 + 4 = 20` for last-before-VALU, `nop_cycles + 1 = 16` for middle-of-chain.

Emulator code (`emu.py:370`) adds `nop_cycles + 1` but doesn't special-case "preceded-by-nop, followed-by-non-nop".

### Hypothesis B2 — "first-in-chain" bonus depends on predecessor TRANS-pipe busy

The first `s_nop(15)` after `s_waitcnt()` costs 16 if the trans-pipe is already drained by the waitcnt, or 20 if the trans-pipe is still draining (observed in wave 1 where v_log took 10 cycles and the trans-pipe hadn't fully cleared when the waitcnt completed). The waitcnt token itself doesn't fully stall — trans completion happens in parallel and the nop picks up the remaining stall.

### Hypothesis B3 — Isolated-nop cost depends on SCC-write predecessor

probe_cmp_chain's isolated nop is preceded by **v_cmp** writing SGPRs. 22 ≈ `15+1 + 6` sgpr-drain. 18 = `15+1 + 2` (wave 1's v_cmps partially overlapped with wave 0's, so less drain). Emulator currently gives 22 (assuming full 6-cycle sgpr drain); wave 1 avoids the full drain because of inter-wave arbitration timing.

### Probe-kernel designs

1. **`custom_probe_nop_chain(N)`** — N consecutive `s_nop(15)` then a `v_add`. Sweep N=1..5. Predict B1: dt_sequence = [16, 16, …, 16, **20**] (last always 20). If B1 wrong, we'll see another pattern.

2. **`custom_probe_nop_after_trans`** — `v_exp_f32 v[0]; v_log_f32 v[0]; s_waitcnt; s_nop(15); v_add`. Vary whether the chain uses 1 vs 4 trans ops. Predict B2: 1 trans → first nop dt=16, 4 trans → first nop dt=20.

3. **`custom_probe_nop_after_scc_write(K)`** — K `v_cmp_gt_f32_e64 s[j], …` writing different SGPRs, then `s_nop(15)`, then `v_add`. Sweep K=1..8. Predict B3: dt_nop = 15 + 1 + min(K, 6) cycles of SGPR drain. This nails down the SGPR_LATENCY constant and its interaction with nop.

4. **`custom_probe_nop_after_sgpr_store`** — `s_mov_b32 s[4], 0; s_nop(15); v_add`. Predict: dt = 16 (no SGPR drain for s_mov? or +1 for scalar-ALU latency). Contrast with SGPR-writing v_cmp.

---

## Pattern C — exp_chain VOPD / cmp / cndmask spacing (12 mismatches)

All 12 mismatches are inside the window `[26..40]` and `[54..61]` of exp_chain wave 0. There are two identical sub-structures each ending in `v_log_f32_e32 v[0]`. Let me tabulate both:

### Chunk 1: indices [25..40]

```
  [25] dt=768 v_cndmask_b32_e64 (v[7])            — huge gap (s_waitcnt_depctr drained here)
  [26] dt=1   VOPD MUL_MUL v0/v1                   EMU=3, diff=+2
  [27] dt=25  s_waitcnt_depctr(4095)               EMU=?, large value (matches — depctr is its own beast)
  [28] dt=1   VOPD MUL_MUL v2/v3                   matches
  [29] dt=1   v_cmp_gt_f32_e32 v[0]                matches
  [30] dt=1   v_cmp_gt_f32_e64 s[0]                matches
  [31] dt=3   v_cmp_gt_f32_e64 s[1]                EMU=4, diff=+1
  [32] dt=1   v_cmp_gt_f32_e64 s[2]                matches
  [33] dt=1   v_cndmask_b32_e64 v[4] (VCC_LO)      matches
  [34] dt=4   v_cndmask_b32_e64 v[5] (s[0])        EMU=1, diff=-3
  [35] dt=1   v_cndmask_b32_e64 v[6] (s[1])        EMU=3, diff=+2
  [36] dt=1   v_cndmask_b32_e64 v[7] (s[2])        EMU=2, diff=+1
  [37] dt=3   VOPD MUL_MUL v0/v1                   EMU=1, diff=-2
  [38] dt=2   VOPD MUL_MUL v2/v3                   EMU=4, diff=+2
  [39] dt=1   v_cndmask_b32_e64 v[4] (VCC_LO)      matches
  [40] dt=2   v_log_f32_e32 v[0]                   EMU=1, diff=-1
```

### Chunk 2: indices [49..61]

```
  [49] dt=1   VOPD SUB_SUB v2/v3
  [50] dt=719 VOPD_LIT MUL_MUL v0/v1               — huge gap (depctr drain)
  [51] dt=1   VOPD_LIT MUL_MUL v2/v3               matches
  [52] dt=4   v_cmp_gt_f32_e32 v[0]                matches
  [53] dt=1   v_cmp_gt_f32_e64 s[0]                matches
  [54] dt=1   v_cmp_gt_f32_e64 s[1]                EMU=4, diff=+3
  [55] dt=1   v_cmp_gt_f32_e64 s[2]                matches
  [56] dt=1   v_cndmask_b32_e64 v[4] (VCC_LO)      matches
  [57] dt=3   v_cndmask_b32_e64 v[5] (s[0])        EMU=1, diff=-2
  [58] dt=1   v_cndmask_b32_e64 v[6] (s[1])        EMU=3, diff=+2
  [59] dt=1   v_cndmask_b32_e64 v[7] (VCC_LO)      matches
  [60] dt=1   v_cndmask_b32_e64 v[11] (s[2])       matches
  [61] dt=3   VOPD MUL_MUL v0/v1                   EMU=1, diff=-2
```

### What's common

The emulator predicts the chunk total correctly within ±1 (110/112 are within ±2) — mismatches are **intra-chunk shuffles of the same total cycle count**:

```
  Chunk 1 indices [26..40]:
    HW dts sum:    1+25+1+1+1+3+1+1+4+1+1+3+2+1+2 = 48
    EMU dts sum:   3+25+1+1+1+4+1+1+1+3+2+1+4+1+1 = 50   (off by +2)
    The two extra emulator cycles come from misattributing work to [26] vs [34].
```

The pattern is: emulator is **over-costing the first VOPD in a new chunk (+2 on [26])** then **under-costing a cndmask in the middle (-3 on [34])** and **under-costing the second VOPD (-2 on [37])** etc. HW is charging those cycles at the v_cmp stage and at later cndmasks; emulator is charging them at the VOPD stage.

### Specifically: VOPD MUL_MUL spacing

| idx | HW dt | EMU dt | predecessor                                  |
| --- | ----- | ------ | -------------------------------------------- |
| [26]| 1     | 3      | cndmask_v[7] dt=768 (just after depctr drain)|
| [28]| 1     | 1      | depctr drain dt=25                           |
| [37]| 3     | 1      | cndmask_v[7] dt=1 (end of cndmask chain)     |
| [38]| 2     | 4      | VOPD v0/v1 dt=3                              |
| [61]| 3     | 1      | cndmask_v[11] dt=1                           |

HW wants 3 cycles between the **first VOPD in a new back-to-back VOPD pair** [37, 38], [61] — emulator gives 1. HW wants 1 cycle between VOPD pairs separated by a depctr drain ([26]). Emulator gives 3.

### Hypothesis C1 — VOPD requires "back-to-back" bypass stamp

A VOPD pair requires 3 cycles of forwarding between adjacent VOPDs when the second VOPD reads the output of the first. After a depctr drain (which clears the bypass network), the next VOPD can issue in 1 cycle because there's no outstanding bypass to satisfy. Emulator has this inverted: costs 3 when it should be 1 (post-drain) and costs 1 when it should be 3 (chained).

### Hypothesis C2 — cndmask SGPR-read latency ceiling

`v_cndmask_b32_e64 vN, a, b, s[k]` reads `s[k]` which was written by a `v_cmp_gt_f32_e64` a few cycles earlier. HW shows the SGPR-read latency is **variable**: usually 1 cycle, but spikes to 4 for the first cndmask that consumes a "cold" SGPR (just-written and not yet in the forwarding network). The spike appears at [34] (first cndmask reading s[0] in chunk 1) and [57] (first cndmask reading s[0] in chunk 2).

Emulator currently models this as a uniform sgpr-read latency and thus spreads the spike evenly across all four cndmasks.

### Hypothesis C3 — v_log-bypass to VOPD carries a +1 penalty

[40] v_log_f32_e32 dt=2 (after cndmask dt=1 at [39]). This is the same +1 the emulator is missing in Pattern F on probe_sgpr_cmps [18]. v_log/v_exp/v_sqrt cost 1 cycle *if* the trans pipe is already warm — else 2. Warming happens on the first TRANS op of a *new* trans sequence.

### Probe-kernel designs

1. **`custom_probe_vopd_spacing(mode)`** — three modes:
   - `chain`: four VOPD MUL_MUL in a row with data dependence v0/v1 → v0/v1 → …
   - `no_dep`: four VOPD writing to disjoint VGPRs
   - `split_depctr`: two VOPDs, `s_waitcnt_depctr(4095)`, two more VOPDs
   
   Predict C1: chain = [1, 3, 3, 3]; no_dep = [1, 1, 1, 1]; split = [1, 25, 1, 3]. Matches HW exp_chain if C1 is right.

2. **`custom_probe_cndmask_sgpr_latency(K)`** — K v_cmp_gt_f32_e64 writing s[4..4+K] then K+1 v_cndmask each reading a distinct s[j]. Sweep K=1..8. Predict C2: `dt[cndmask_0] = 1 + first_sgpr_wait`; subsequent cndmasks = 1 once forwarding is warm. Extract the latency curve vs K.

3. **`custom_probe_trans_warmup`** — measure cost of v_log after varying preceding instruction types: nothing, VALU, VOPD, another trans, nop. Predict C3: warm ≤ 1, cold = 2.

4. **`custom_probe_vopd_after_vcmp`** — put a v_cmp chain between two VOPDs. The v_cmp writes SGPR; does that reset the VOPD-chain cost? Maps to [31/32]→[33..36]→[37/38] path in exp_chain.

---

## Pattern D — `v_lshlrev_b32_e32` after `s_waitcnt_lgkmcnt` (1 mismatch)

### Tokens

All kernels have identical prologue: `s_load_b64 → s_waitcnt_lgkmcnt(NULL) → v_lshlrev_b32_e32(v[0], 2) → …`.

| kernel            | wave | [2] dt | notes                                |
| ----------------- | ---- | ------ | ------------------------------------ |
| data_deps         | 0    | 1      |                                      |
| data_deps         | 1    | 1      |                                      |
| probe_branch_cost | 0    | 1      |                                      |
| probe_branch_cost | 1    | 1      |                                      |
| probe_cmp_chain   | 0    | 1      |                                      |
| probe_cmp_chain   | 1    | 1      |                                      |
| probe_sgpr_cmps   | 0    | 1      |                                      |
| probe_sgpr_cmps   | 1    | 1      |                                      |
| probe_vmem_chain  | 0    | 1      |                                      |
| probe_vmem_chain  | 1    | **4**  | ← anomaly                            |

### Absolute times

```
probe_vmem_chain wave 0: [1] waitcnt@263548  [2] v_lshlrev@263549   dt=1
probe_vmem_chain wave 1: [1] waitcnt@263550  [2] v_lshlrev@263554   dt=4   <-- slipped 3 cy
```

Wave 1's waitcnt is only 2 cycles after wave 0's, but wave 1's first VALU is 5 cycles after wave 0's first VALU. Wave 0 issued 4 back-to-back VALUs [2..5] at cycles 263549–263553. Wave 1's first VALU issues at 263554 — exactly one cycle after wave 0's last back-to-back VALU completes.

### Control — what's different about probe_vmem_chain?

Difference vs the other kernels: probe_vmem_chain has **no global_load before the first VALU**, so the two waves are not separated by an ~600-cycle DRAM wait. They both leave waitcnt within 2 cycles of each other → the first-VALU arbiter sees them both ready simultaneously and has to pick one.

In the other kernels a 600-cycle DRAM wait happens between waitcnt_lgkmcnt and waitcnt_vmcnt, wave separation stabilizes at 10-20 cycles, and by the time the first VALU after vmcnt issues there is enough slack that both waves issue in the same cycle.

### Hypothesis D1 — CU VALU issue port is shared between sibling waves for a "hot start" window

When two waves exit `s_waitcnt_lgkmcnt(NULL)` within 2-3 cycles of each other, the CU's VALU issue arbiter grants priority to wave 0; wave 1 starts its first VALU only when wave 0's VALU burst "breathes" (gap ≥ 1 cycle). In probe_vmem_chain, wave 0 has a 4-VALU back-to-back burst, so wave 1 slips 3 cycles.

Evidence: wave 1's [3..6] dt = 1/1/1/1 (no further slippage). It paid the slip once, at first issue.

This is *the* shared-resource hypothesis, just at the VALU issue port rather than SIMD. A WGP has 4 SIMDs but only ~2 VALU issue slots per cycle per SQ — when both waves want to issue in the same cycle, one loses. Consistent with RGP's "different SIMDs" finding: waves can be on different SIMDs yet share the CU's issuer.

### Probe-kernel designs

1. **`custom_probe_cold_start(N)`** — no global_load, only `s_load_b64 → s_waitcnt → N back-to-back v_mov` on each wave. Sweep N=1..8. Predict D1: wave 1's first VALU dt = N (or min(N, limit)). If N=1, no slip; N ≥ 2 slip grows linearly.

2. **`custom_probe_cold_start_stagger(M)`** — insert M `s_nop(0)` in only one wave (via workgroup-id predicate) between waitcnt and first VALU. Predict: M=0 → wave 1 slip=N; M=N → no slip.

3. **`custom_probe_issue_share_sustained`** — 20 back-to-back v_mov with no global_load. Measure steady-state interleave period; expect 1-cycle rotation between waves (wave 0 issues even cycles, wave 1 odd cycles) if issuer is 1-wide, or full parallel if 2-wide.

4. **`custom_probe_warm_start`** — identical to cold_start but with a global_load before the nop sequence to force wave separation > 20 cycles. Predict: no slip (wave 1 [2] dt=1 like all other kernels).

---

## Pattern E — probe_branch_cost SCC/branch timing (3 mismatches)

### Tokens

```
probe_branch_cost w0                              probe_branch_cost w1
  [ 4] dt=583  s_waitcnt_vmcnt(NULL)                [ 4] dt=595  s_waitcnt_vmcnt(NULL)
  [ 5] dt=1    s_mov_b32(s[4], 0)                   [ 5] dt=2    s_mov_b32(s[4], 0)       ← +1
  [ 6] dt=1    s_cmp_eq_i32(s[4], 1)                [ 6] dt=1    s_cmp_eq_i32(s[4], 1)
  [ 7] dt=8    s_cbranch_scc1(9)                    [ 7] dt=10   s_cbranch_scc1(9)        ← +2
  [ 8] dt=3    v_mov_b32_e32(v[10], 1.0)            [ 8] dt=3    v_mov_b32_e32(v[10])
  [ 9] dt=1    v_mov_b32_e32(v[11], 2.0)            [ 9] dt=1
  [10] dt=1    s_cmp_eq_i32(s[4], 1)                [10] dt=1
  [11] dt=9    s_cbranch_scc1(5)                    [11] dt=9    s_cbranch_scc1(5)        same
  [12..14]     identical                            [12..14]     identical
```

Emulator always produces: s_mov=1, s_cmp=1, s_cbranch=9, v_mov=3.

Wave 0 matches on s_mov and s_cmp, but s_cbranch[7]=8 (not 9) — EMU over-costs by 1.
Wave 1's s_mov costs 2 (EMU=1) and s_cbranch[7]=10 (EMU=9) — +1 and +1.

Wave 0's second s_cbranch[11] is 9 (matches). Wave 1's second s_cbranch[11] is 9 (matches). The first s_cbranch is flakier.

### Observations

Wave 1 waitcnt takes 595 cy vs wave 0's 583 cy — that's +12 cy of DRAM-return stagger. The stagger propagates into wave 1's early scalar instructions.

Absolute times:
```
w0 [5] s_mov@11102   [7] s_cbranch@11104+8=11111
w1 [5] s_mov@11126   [7] s_cbranch@11129+10=11139
```
Wave1−wave0 at [5] = 24; at [7] = 28. Wave 1 is drifting +4 through the sequence.

### Hypothesis E1 — Scalar instruction latency under DRAM-return contention

The scalar pipeline (s_mov, s_cmp, s_cbranch) has a variable per-instruction latency in the 1-2 cycle range when the DRAM return is recent. Wave 1's extra 12-cycle DRAM wait on vmcnt leaves some scalar forwarding path still settling, costing +1 at s_mov and +2 at the first s_cbranch.

### Hypothesis E2 — `s_cbranch_scc1` cost depends on whether SCC-writer was recent

- Wave 0 [7] s_cbranch = 8. SCC written at [6] 1 cy ago, fresh.
- Wave 1 [7] s_cbranch = 10. SCC written at [6] 1 cy ago, but s_mov was slower.
- Both [11] s_cbranch = 9. SCC written at [10] 1 cy ago.

Emulator's constant 9 is a reasonable average; HW varies in the range 8-10 depending on scalar pipe phase. This is probably the **scalar issue cycle aligning vs misaligning with the 4-cycle scalar engine beat**.

### Probe-kernel designs

1. **`custom_probe_scalar_beat(phase)`** — insert 0..3 `s_nop(0)` before `s_cmp_eq_i32 → s_cbranch`. Sweep phase. Predict E2: s_cbranch dt oscillates 8/9/10/9 with period 4 as phase varies.

2. **`custom_probe_scalar_after_vmem(K)`** — K `s_nop` between `s_waitcnt_vmcnt` and `s_mov_b32`. Sweep K=0..20. Predict E1: at K=0 s_mov dt = 1 or 2 depending on how much DRAM stagger leaked; at K ≥ 4 s_mov dt = 1 always.

3. **`custom_probe_branch_fanout(M)`** — M copies of `s_cmp_eq / s_cbranch_scc1` in a row. Predict: first = 8-10, remaining stabilize at 9.

4. **`custom_probe_scc_write_pathway`** — compare `s_cmp_eq_i32 → s_cbranch`, `v_cmp_eq_u32 → s_cbranch`, `s_bitcmp1_b32 → s_cbranch`. Predict: each SCC-writer has its own latency; v_cmp→s_cbranch may cost more because SCC comes from vector pipe.

---

## Pattern F — Wave-slip on cndmask / v_log / v_mov chains (5 mismatches)

### Tokens

All five F mismatches are in the **trailing wave** (wave 1 mostly, w0 [16] on the cndmask chain). Reprise of cross-cutting observation:

| kernel          | w | idx | inst                    | HW | EMU | slip at this point |
| --------------- | - | --- | ----------------------- | -- | --- | ------------------ |
| probe_cmp_chain | 1 | 6   | v_mov_b32(v[3], 2.0)    | 2  | 1   | +1                 |
| probe_sgpr_cmps | 1 | 8   | v_mov_b32(v[5], 4.0)    | 2  | 1   | +1                 |
| probe_sgpr_cmps | 0 | 16  | v_cndmask_b32_e64(v[9]) | 2  | 1   | +1                 |
| probe_sgpr_cmps | 1 | 16  | v_cndmask_b32_e64(v[9]) | 5  | 1   | +4 (cumulative)    |
| probe_sgpr_cmps | 1 | 18  | v_log_f32_e32(v[10])    | 10 | 4   | +6                 |

These are the specific tokens where wave 1 (or w0 at [16]) picks up an extra cycle relative to the other wave. The **absolute cycle counts** from the probe_sgpr_cmps cross-section above show:

```
  wave1−wave0 skew grows from 19 to 22 to 28 to 32 through indices [15→16→18→21]
```

The places where the skew grows are exactly the mismatched F/B tokens.

### Why these specific instructions?

- **v_mov_b32 to v[5]** and **v_cndmask_b32_e64 to v[9]** — these are the last VGPR allocations of the chain (first use of a fresh high-numbered VGPR). Hypothesis: **VGPR allocation stalls when the SIMD's VGPR file write-port is saturated**. Fresh allocations are more expensive than re-writes to already-touched VGPRs.
- **v_log_f32_e32 (trans)** — the TRANS pipe is a 4-wide shared resource with some wave-serialization. When wave 0 just issued v_exp, wave 1 has to wait 4 cy for its own v_exp and *another* 4 for v_log (pipeline interlock).

Note: Wave 0 [16] `v_cndmask_b32_e64(v[9])` also has HW=2 (not 1). So even the *leading* wave pays 1 extra cycle on the last cndmask-to-new-VGPR. This matches the "VGPR fresh-allocation" hypothesis.

### Hypothesis F1 — Fresh-VGPR allocation cost

HW charges +1 cycle when a VALU writes a VGPR that hasn't been touched in the current wavefront's recent history. Emulator treats all VGPR writes as uniform cost.

Controls: probe_sgpr_cmps w0 [16] is a cndmask to v[9], brand new. HW=2. [13/14/15] cndmask to v[6]/v[7]/v[8] — all dt=1 (fresh too!) — hmm, that contradicts. Let me refine.

Actually [13]=1, [14]=1, [15]=1, [16]=2. The difference for [16] is that its SGPR source `s[6]` was written by the last v_cmp [12]. SGPR forwarding for the **last** cndmask in a sequence crosses a forwarding boundary — maybe the SGPR write cache is being retired.

### Hypothesis F2 — SGPR-read-window closes at end of cndmask chain

After N consecutive cndmasks reading N different SGPRs written by N v_cmps, the (N+1)th cndmask (or the first v_cmp after) pays 1 extra cycle because the SGPR forwarding network refills on a 4-wide schedule. Same cost model as C2 but viewed from the other side.

### Hypothesis F3 — Wave-pair TRANS pipe serialization

The TRANS pipe (v_exp/v_log/v_sqrt/v_rsq…) serializes wave-pair access. When wave 0 issues v_exp at cycle T, wave 1's v_exp can issue at T+1 (one cycle skew in the trans pipe), but wave 1's v_log has to wait for wave 0's v_log to finish, causing a 4-cycle stall on wave 1's v_log. The 4→10 jump on probe_sgpr_cmps w1 [18] is exactly this.

Emulator doesn't model wave-pair TRANS interlock at all.

### Probe-kernel designs

1. **`custom_probe_vgpr_fresh_alloc(N)`** — N consecutive v_mov to fresh VGPRs v[10..10+N], sweep N=1..32. Predict F1: linear cost. If F1 wrong, steady 1 cy/VALU.

2. **`custom_probe_cndmask_tail(K)`** — K v_cmp writing s[4..4+K] then K v_cndmask reading them, then 1 more v_cndmask reading **s[4]** (already used). Predict F2: the extra cndmask dt = 2 (fresh SGPR port cycle) if K ≥ 3, else 1.

3. **`custom_probe_trans_wave_pair`** — identical trans chain on both waves (workgroup size 64×2). Measure wave 1's v_log cost as a function of position. Predict F3: wave 1 v_log dt ≥ wave 0 v_log dt + 4 when the trans sequence is back-to-back.

4. **`custom_probe_trans_spaced(K)`** — K v_add between v_exp and v_log. Predict: at K=0 the wave-pair interlock is visible; at K ≥ 4 both waves pay baseline because the wave-0 v_log finishes before wave-1 v_exp blocks.

---

## Priorities for emulator fixes

Ranked by mismatches covered and model simplicity:

1. **Wave-pair shared-issuer slip model** (covers D, F, half of B) — ~10-12 mismatches. Add a model where wave 1's ready time is advanced by max(0, wave0_issue_cycle − wave1_ready + 1) when both are trying to issue in the same cycle AND the instruction class is on a shared pipe (TRANS, long cndmask chain, back-to-back VALU burst).

2. **Last-nop-before-VALU +4 penalty** (covers half of B) — 2-3 mismatches. Trivial patch: when ending an s_nop chain with a non-nop follow-up, add 4 cycles to the last nop's issue stamp.

3. **VMEM store first-issue latency 17 vs 21** (covers A) — 3-4 mismatches. Encode the VMEM write arbiter open/close state and charge 17 when there's a recent VMEM op keeping the arbiter warm, 21 when cold.

4. **VOPD / cndmask sgpr-read phasing** (covers C) — 12 mismatches. Largest but most subtle. Probe-kernels 1, 2, 3 in section C isolate the three sub-hypotheses; fix in that order.

5. **Scalar beat phase** (covers E) — 3 mismatches. Probe-kernel E1 gives a 4-phase oscillation if true; then add a `scalar_issue_phase` counter.

## Probe-kernel rollout plan

Start with **D1 (cold_start) → F3 (trans_wave_pair) → C1 (vopd_spacing) → B1 (nop_chain) → A4 (store_bypass_mix_load)** in that order. D and F share infrastructure (two-wave WG with controlled skew), so batching them costs nothing. C and B need single-wave precision, so they can run on WG=1 to eliminate inter-wave noise.

Each probe should be a ≤ 20-line custom kernel in `test/amd/test_custom_kernel.py`, parametrized by a single knob that varies the suspected trigger. Harness the existing `rigorous_hw_test.py --capture` path — the pkl files land in `extra/sqtt/captures/probe_*.pkl` alongside the existing ones — then compare deltas by hand.
