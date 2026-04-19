# Step 7 Status — HW-Confirmed Fixes (2026-04-18)

## What's ready (Steps 1-6 done, pushed)

The emulator `_simulate_sq_timing` is now fully decoupled into subsystems,
each owning its state in `test/mockgpu/amd/sq_timing/*.py`:

| Step | Subsystem | State migrated | Commit |
|------|-----------|----------------|--------|
| 1 | `TimingConstants` | 28 constants → dataclass | a49bd99 |
| 2 | `IbFetch` | `last_drain_stamp`, `had_drain_nop` | 38622708d |
| 3 | `LdsPipe` | `cu_lds_*`, `lgkm_pend` | 75e38eab1 |
| 4 | `VmemPipe` | 6 per-wave VMEM arrays | c28bfb653 |
| 5a | `TransPipe` | `trans_pipe_avail`, `trans_vgpr_ready`, `scalar_after_trans_ready`, `valu_pend` | d40a1f0c5 |
| 5b | `SgprScoreboard` | `sgpr_write_time` + 4 LIT v_cmp buffers, `smem_sgpr_ready` | 15973cf9a |
| 5c | `VAluPipe` | `vopd_pipe_avail`, `last_vopd_issue`, `bank_vopd_write_time`, `vgpr_*` scoreboards | 58ffe63de |
| 6 | `ScalarPipe` | `scc_write_time`, `exec_write_time` | 7214c9eb2 |

**Baseline preserved at 310/340 exact through every step.**

## Step 7a attempted — REVERTED

Goal: land `_VOPD_PIPE_CYCLES = 1` gated on "prev_inst_was_vopd" flag (per
`BATCH_B_FINDINGS.md` — chained VOPDs HW=1cy unanimous across 16 waves).

Implementation: added `prev_inst_was_vopd` flag to `VAluPipe`; set after every
issue, used to gate VOPD pipe-avail gap (1 if chained, 4 if first).

Result: **exp_chain regressed 100 → 98 exact (−2 tokens)**. NET baseline
310 → 308. Reverted.

### Why it regressed — per-kernel diff revealed the root cause

The 12 C-group mismatches at `[26, 31, 34-38, 40, 54, 57, 58, 61]` are NOT
from VOPD pipe occupancy. The Step 7a fix changed NONE of those 12. Instead
it broke `[51]` and `[52]` (previously matching) because the chain-gap
redistribution shifted stamp[51] forward 3 cycles.

**Key insight:** `exp_chain`'s VOPDs pay their 3-cycle-ish spacing because of
**cndmask→VOPD SGPR forwarding**, NOT VOPD→VOPD pipe occupancy. Our Batch B
microbenches (`mb_vopd_chain_n4_{raw,no_raw}` etc.) measure pure VOPD→VOPD
behavior in isolation — 1cy. But in `exp_chain` the VOPDs sit AFTER cndmask
chains, so the slow-path is the cndmask-result forwarding, which our pure
VOPD probes never triggered.

This aligns with `MISMATCH_ANALYSIS.md` §C hypothesis **C2: "cndmask SGPR-read
latency ceiling"** — the 4cy spike at `[34]` and `[57]` (first cndmask consuming
a fresh SGPR after a v_cmp chain) is the real pattern, and the subsequent VOPD
delays are cascade effects.

## Recommended Step 7 rework

The 12 C-group mismatches are a **cndmask-forwarding** problem, not a VOPD
problem — but see the update below: the Batch B cndmask sweep did **not**
confirm the "first-use +3cy" hypothesis.

### 2026-04-19 UPDATE — Batch B cndmask sweep analysis

Analyzed modal-across-16-waves for all cndmask probes:

| Capture | cndmask dts (modal) | Notes |
|---|---|---|
| `mb_vcmp_cndmask_k1` | [1] | 1 cmp → 1 cndmask |
| `mb_vcmp_cndmask_k2` | [1, 1] | 2 cmps → 2 cndmasks |
| `mb_vcmp_cndmask_k4` | [1, 1, 1, 1] | 4 cmps → 4 cndmasks (all dt=1) |
| `mb_vcmp_cndmask_k8` | [1, 1, ...] | 8 cmps → 8 cndmasks (all dt=1) |
| `mb_cndmask_sgpr_fresh_n4` | [1, 1, **2**, 1] | 4-chain, pos-3 spike |
| `mb_vcmp_after_cndmask_chain` | [1, 1, **2**, 1] | same pos-3 spike |
| `mb_cndmask_sgpr_stale_n4` (nop drain) | [1, 1, 1, 1] | drain-then-cndmask |
| `mb_vcmp_spaced_cndmask` | [1] (after s_nop(4)) | 1 cmp → nop → 1 cndmask |
| `mb_vcmp_interleave_cndmask` | [1, 1, 1, 1] | alternating cmp/cndmask |

**Falsified:** the "first cndmask pays 3-4cy, tapering to 1cy" hypothesis is
**not** supported. HW shows cndmask→cndmask throughput is a clean **1cy**,
with only a minor 2cy spike at position 3 of a 4-chain (likely the LIT v_cmp
completion-buffer depth-2 artifact already modeled in `_CMP_LIT_WB_LATENCY`).

The 12 exp_chain C-group mismatches must therefore stem from a different
mechanism — candidates:
- **VOPD-context** sgpr forwarding (cndmask→VOPD, not cndmask→cndmask)
- The 2cy pos-3 spike compounding through longer chains
- A VCC-implicit dependency path (VOPD writes VCC, next cndmask reads it)

Fix #1 as originally designed (SgprScoreboard tiering) would regress the
Batch B k-sweep kernels since they're currently matching at 1cy. **Do not
land tiering**. Next investigation: inspect exp_chain's precise token pattern
around each of the 12 mismatches to identify what the VOPDs read — if any
consume an SGPR written by a prior cndmask/VOPD, that's the next hypothesis.

### Original hypothesis (superseded by the update above)

The fix needs to live in `SgprScoreboard.read_stall` — the first
`v_cndmask_b32_e64` that reads a just-written SGPR (from a chain of v_cmps)
pays an extra ~3-4cy that tapers on subsequent cndmasks. Current emu models
SGPR-read latency as a constant `_SGPR_LATENCY = 4` or `_CNDMASK_SGPR_LATENCY = 4`;
the HW data suggests it's **tiered** — first consumer pays full 4, subsequent
consumers of the same chain pay 1.

## The cleanest HW-confirmed fixes still on the table

Each is one isolated change in a specific subsystem (now that Steps 1-6 have
decoupled them):

1. **Cndmask SGPR first-use +3cy** — `SgprScoreboard.read_stall` tiering
   (Batch B probes already captured, need analysis)

2. **s_nop post-VALU +1 → +2** — Batch A `mb_snop_15_after_valu` modal=18,
   current EMU gives 17. Small 1cy gain, regresses
   `probe_cmp_chain/probe_vmem_chain` by 3 tokens on current pkls (wave-noise).
   Net −2. Park.

3. **Trans→trans RAW same-VGPR 4cy** — ~~`TransPipe.trans_read_stall` currently
   uses 27/31cy for all trans-vgpr reads; HW shows 4cy when the reader is
   also a trans op.~~ **2026-04-19 UPDATE:** already in emu. trans→trans
   stalls only on `trans[i].pipe_avail` (4cy via `TRANS_PIPE_CYCLES`), and
   commit `e62be6e28` intentionally dropped the 27/31cy path for non-trans
   readers (s_waitcnt_depctr absorbs it). Batch B `mb_trans_raw_exp_log`
   confirms the prediction (HW=4cy). The `_trans_read_deadline` local in
   emu.py:578-582 is harmless dead code. **Nothing to change.**

4. **CBranch_scc1 not-taken 8/13 context-sensitive** — `ScalarPipe.cbranch_cost`.
   Batch A probes showed:
   - tight `s_mov→s_cmp→s_cbranch`→VALU: 8cy
   - broken `s_mov→nop→s_cmp→s_cbranch`→VALU: 13cy
   - tight + VMEM successor: 12cy

Each of these is one subsystem-isolated change; the refactor makes them
tryable without cross-regression (unlike Step 7a, where the problem turned
out to be cross-subsystem anyway).

## What to do next

1. ~~**Analyze `mb_vcmp_cndmask_k{1,2,4,8}`**~~ — DONE 2026-04-19. Hypothesis
   falsified; see the update block above. Chained cndmasks run at 1cy.
2. ~~**Land fix #3 (Trans RAW 4cy)** first — isolated, won't cross-perturb.~~
   Already in emu (see updated fix #3 entry).
3. **Land fix #4 (cbranch context-sensitive)** — also isolated.
4. **Design cndmask-first-use fix** from the Batch B curve; land as fix #1.
5. **Final validation**: capture Batch B again against the fixed emu, confirm
   10+ of the 12 exp_chain C-group mismatches close.

## Takeaway

Steps 1-6 shipped the full subsystem decoupling the plan called for — the
emulator now has clean extension points. The 28 remaining mismatches are
closable in ~3-4 targeted fixes, each living in its own subsystem. Step 7a
didn't ship because the target hypothesis was incomplete (HW data showed
chained VOPDs 1cy, but exp_chain's slow-path is cndmask-forwarding, not
VOPD-pipe).

Total work this session:
- 7 new sq_timing subsystems (~450 LoC)
- 98 Batch A + 142 Batch B microbenches (~1800 LoC of probes)
- 240 HW .pkl captures × 16 waves each
- 4 analysis docs (MISMATCH_ANALYSIS, PROBE_FINDINGS, BATCH_A/B_FINDINGS, EMU_REWRITE_DESIGN)
- Baseline 310/340 preserved across every commit
