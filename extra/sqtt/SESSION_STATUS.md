# RDNA3 Cycle-Accurate Emulator — Session Status

## Bounty Goal
$1,000 bounty: Make tinygrad's software GPU emulator produce cycle-accurate instruction timing matching real AMD 7900 XTX hardware, validated via SQTT (Shader Queue Thread Trace).

## Current Accuracy: 293/321 exact (91.3%), 309/321 ±2 (96.3%)

All SQTT tests pass (encoder, map, examples, timing, E2E).
Validated against 9 HW-captured kernels (17 total captures, 11 comparable + 6 new probes).

### Per-Kernel Breakdown (2026-04-18)

| Kernel | Exact | Total | Rate | ±2 Rate | Notes |
|------------------|-------|-------|--------|---------|-------|
| elementwise | 16 | 16 | 100.0% | 100.0% | ✅ Perfect |
| cast | 14 | 14 | 100.0% | 100.0% | ✅ Perfect |
| plus | 13 | 13 | 100.0% | 100.0% | ✅ Perfect |
| data_deps | 9 | 10 | 90.0% | 90.0% | w1 [2] v_lshlrev variance |
| probe_vmem_chain | 21 | 22 | 95.5% | 95.5% | w1 startup stagger (±1cy) |
| probe_sgpr_cmps | 57 | 64 | 89.1% | 92.2% | s_nop(15) context variance |
| probe_cmp_chain | 41 | 44 | 93.2% | 95.5% | W0/W1 store bypass split |
| probe_branch_cost| 22 | 26 | 84.6% | 96.2% | branch SCC stamp variance |
| exp_chain | 100 | 112 | 89.3% | 98.2% | VOPD inter-pattern spacing |

Progress: 74.8% → 91.3% exact across sessions. ±2 accuracy is 96.3%.

### Session 2026-04-18 Changes
1. **LIT v_cmp commit-buffer model** (292 → 293): C[n] = max(W[n], C[n-1]+2) for consecutive LIT v_cmp chains.
2. **VMEM bypass restricted to done-wave drain** (probe_cmp_chain w1 + probe_branch_cost w1 fixed).
3. **Trans VGPR read stall deferred to ready[i]** (exp_chain VOPD+depctr distribution fixed).
4. **VOPD_PIPE_CYCLES = 4 for cold starts, 1 for VOPD_LIT→VOPD_LIT** (exp_chain dual-issue).
5. **VGPR bank/cache infrastructure added** (Seb-V model): `bank_vopd_write_time`, `last_vopd_issue`, `consecutive_single_valu` tracking. Inter-VOPD bank-conflict check in place but doesn't fire on current test set (compiler enforces VOPD bank validity). Kept as groundwork for future work.

### Session 2026-04-17 Changes
1. **PC offset normalization** in rigorous_hw_test.py: compare PC offsets relative to first instruction (HW/EMU load bases differ). Unblocked exp_chain/cast/elementwise/plus from being skipped — recovered 277/321 → baseline.
2. **Defer trans VGPR read stall to ready[i]** in emu.py _simulate_sq_timing: non-trans VALU reading trans-written VGPR now stamps at issue_cycle (matches HW SQTT stamp-at-dispatch), with trans wait absorbed by subsequent s_waitcnt_depctr. Fixed exp_chain VOPD+depctr distribution: +3 exact, +4 ±2.
3. Parked probe_vmem_chain w1 [2]: ±1cy artifact of SMEM backend stagger model (HW=2cy, EMU=1cy). Fixing would regress other tokens.
4. **Restrict VMEM bypass to done-wave drain**: changed `_vmem_wr_bypass_active` so that a running wave's vmem_drain_deadline no longer qualifies (only wave_done wave's does). Fixed probe_cmp_chain w1 + probe_branch_cost w1: +2 exact, +2 ±2. data_deps w1 regresses −1 (waves in this tiny kernel are out-of-sync enough that the later wave legitimately bypasses off the earlier running wave's drain).

## What We Built

### 1. Timing Model Improvements (test/mockgpu/amd/emu.py)
- **Time-based delay_alu stalls**: VALU_DEP computed from actual issue times (5-cycle pipeline). **+15 exact matches.**
- **Parallel trans ALU**: v_exp/v_log run in parallel with VALU (issue_cost=1, pipeline occupancy enforced).
- **VMEM drain tracking**: s_nop before s_endpgm waits ~15cy for VMEM pipeline acceptance.
- **Width-dependent VMEM forwarding**: global_store_b128 adds +3cy for wider VGPR reads.
- **Split DS/VMEM forwarding**: LDS write=26, LDS read=22, VMEM write=21, VMEM read=22.
- **Trans pipeline**: 27cy latency (v_log/v_rcp), 31cy (v_exp/v_sqrt/v_rsq), 4cy pipeline occupancy.
- **s_nop(N) stall modeling**: N+1 cycles, late-stamp for N>0, caps VMEM forwarding deadlines.
- **Branch NOT-TAKEN timing**: stamp=issue+7, cost=10 (HW validated: probe_branch_cost).
- **SGPR write-to-read stalls**: 4cy HW-enforced without delay_alu; 6cy for v_cndmask non-VCC.
- **v_cndmask SGPR drain**: pending non-VCC write-backs stall condition SGPR reads.
- **VOP3_SDST src2 skip**: HW doesn't read phantom src2 encoding.
- **Inter-wave VMEM bypass**: SQ pipelines forwarding when other waves are ready (21→17cy).
- **VOPD dual-issue occupancy**: 2cy spacing for consecutive VOPDs.
- **EXEC write latency**: v_cmpx→s_cbranch_execz needs 24cy propagation.
- **LDS b128 stagger**: serialized b128 loads have 17cy upper-VGPR stagger.
- **Slow-fresh VGPR tracking**: first-use VGPR from b128 stagger has 9cy latency.
- **VALU burst scheduling**: consecutive VALU from same wave gets priority.
- **Wave-count-dependent stalls**: single-wave kernels need +1 stall vs multi-wave.

### 2. SQTT Hardware Capture (tinygrad/runtime/ops_amd.py)
- **wgp_sel auto-detect**: `(cu_per_simd_array - 1) // 2` = 3 (wgp_sel=0 DISABLES tracing!)
- **Token mask overflow fix**: Prevented token_exclude from corrupting ttrace_exec bit.
- **profile_standard enforcement**: Required for clock gating suppression.
- **Auto-detect traced CU**: map_insts finds traced CU from first WAVESTART.
- **SQTT_ITRACE_SE_MASK**: All 6 SEs enabled (0x3f).
- **s_sendmsg skip**: Added S_SENDMSG/S_SENDMSGHALT to _SOPP_SKIP.

### 3. Test Infrastructure
- **test/amd/test_emulator_e2e.py**: 12 E2E tests, 45+ subtests across 10 kernel types.
- **extra/sqtt/rigorous_hw_test.py**: Multi-kernel HW capture + comparison (17 kernels, --capture/--compare).
- **extra/sqtt/capture_discover_ops.py**: Broad instruction category capture tool.
- **extra/sqtt/SQTT_DEEP_DIVE.md**: Comprehensive SQTT documentation.

### 4. HW Reference Captures (extra/sqtt/captures/rigorous/)
17 kernels captured from real 7900 XTX with profile_standard power mode:
data_deps, exp_chain, elementwise, plus, cast, softmax, layernorm, reduce256, reduce_large,
matmul_medium, probe_sgpr_cmps, probe_cmp_chain, probe_branch_cost, probe_vmem_chain,
plus discover_ops (751 ALU + 59 memory instructions).

## Key Architecture Discoveries

1. **RDNA3 trans ALU runs in PARALLEL with VALU** — After v_exp issues, VALU can issue 1 cycle later. The 4-cycle trans pipeline only blocks subsequent TRANS instructions.
2. **VMEM drain = 15 cycles** — After global_store/load issues, s_nop waits ~15 cycles for VMEM pipeline acceptance.
3. **Time-based delay_alu is the correct model** — Fixed stalls incorrectly add cycles even after long waits. Time-based: `max(0, dep_issue + 5 - current_cycle)` gives 0 stall when enough time elapsed.
4. **wgp_sel=0 DISABLES instruction tracing** on GFX11 — Must use wgp_sel≥1.
5. **profile_standard power mode REQUIRED** — Dynamic clock gating suppresses SQ instruction tokens without it.
6. **Wave placement is non-deterministic** — 1-wave kernels only land on traced CU ~15% of the time. Must retry.
7. **S_SENDMSG is async fire-and-forget** — No SQTT packet emitted.
8. **Branch NOT-TAKEN has late-stamp** — SQTT token stamped 7cy after issue, 10cy total cost.
9. **SGPR writes have 4cy HW-enforced latency** — No delay_alu needed; HW interlocks.
10. **v_cndmask non-VCC condition SGPR has 6cy latency** + drain stall for pending writes.
11. **Inter-wave VMEM forwarding bypass** — SQ overlaps forwarding when other waves schedulable 4cy early.
12. **s_nop counts toward forwarding windows** — Caps VALU→VMEM deadlines after drain.

## Remaining 43 Mismatches — Analysis & Plan

### Analyzed root causes (ready to implement):

1. **Post-trans SALU stall** (probe_sgpr_cmps, ~4 matches):
   After trans VALU (v_exp/v_log/v_sqrt), next scalar/immediate op is delayed +2cy.
   HW evidence: v_sqrt→s_waitcnt gap=3 (not 1). Use consume-on-first-use pattern.

2. **s_nop formula depends on predecessor type** (probe_sgpr_cmps, probe_vmem_chain, ~6 matches):
   After VALU: `nop_stamp = nop_start + nop_cycles + 1` (validated: gap=13 for nop(10)).
   After scalar: `nop_stamp = nop_start + nop_cycles` (validated: gap=16 for nop(15)).

3. **s_nop SGPR write drain** (probe_cmp_chain, ~2 matches):
   s_nop waits for pending non-VCC SGPR writes to complete before starting.
   Formula: `nop_start = max(ready, max(sgpr_write_time[r] + SGPR_LATENCY for r if r!=VCC) + 1)`.

### Harder problems (need more analysis):

4. **exp_chain SGPR write-back contention** (13 mismatches):
   Three identical SGPR blocks have different stall distributions. Block 1 [10] has 4cy stall,
   Block 3 [54] has 0cy stall for identical instructions. Likely SGPR write-back queue pressure
   cleared by depctr flush. Needs queue depth tracking.

5. **probe_branch_cost SCC latency** (5 mismatches):
   2nd branch in sequence shows different SCC propagation timing. Possibly HW caches SCC state.

6. **v_cndmask drain model refinement** (probe_sgpr_cmps):
   Drain threshold `gap<3` is too aggressive in some contexts but correct in others.

7. **s_nop inter-wave interference** (probe_sgpr_cmps):
   Alternating s_nop gaps (16,16,20 vs 20,16,20) suggest SQ serializes s_nop across waves.

## Timing Constants (Calibrated from HW)

```python
_LDS_RD_LATENCY = 31           # LDS read result latency
_LDS_WR_LATENCY = 33           # LDS write completion latency
_SMEM_LATENCY = 200            # Scalar memory latency
_VMEM_LATENCY = 300            # Vector memory (DRAM) latency
_BARRIER_FROM_LAST = 6         # Barrier overhead from last wave arrival
_LDS_SERVICE_COST = 6          # Per-wave LDS serialization cost
_VALU_DS_WR_FORWARD = 26       # VALU→DS write forwarding stall
_VALU_DS_RD_FORWARD = 22       # VALU→DS read forwarding stall
_VALU_VMEM_WR_FORWARD = 21     # VALU→VMEM write forwarding (HW-validated)
_VALU_VMEM_WR_BYPASS = 4       # Inter-wave VMEM overlap (21→17cy)
_VALU_VMEM_RD_FORWARD = 22     # VALU→VMEM read forwarding
_VMEM_DRAIN_CYCLES = 15        # VMEM pipeline acceptance time
_VMEM_EXEC_MIN = 8             # Minimum VMEM execution after forwarding overlap
_TRANS_PIPE_CYCLES = 4         # Trans ALU pipeline occupancy
_TRANS_PIPELINE_LATENCY = 27   # Trans result latency (v_log, v_rcp)
_TRANS_PIPELINE_LATENCY_SQRT = 31 # Trans result latency (v_exp, v_sqrt, v_rsq)
_SGPR_LATENCY = 4              # VALU SGPR write-to-read latency
_CNDMASK_SGPR_LATENCY = 6     # v_cndmask non-VCC condition SGPR latency
_WAVESTART_GAP = 1             # Gap between wave starts
_FIRST_INST_GAP = 2            # Gap from wavestart to first instruction
_VOPD_PIPE_CYCLES = 2         # VOPD dual-issue occupancy
_EXEC_WRITE_LATENCY = 24      # v_cmpx → s_cbranch_execz propagation
_LDS_B128_EXTRA = 5           # Extra LDS latency for b128 loads
_LDS_B128_VGPR_STAGGER = 17   # Upper VGPR stagger for serialized b128
_LDS_B128_RD_SERVICE = 19     # b128 read LDS serialization
VALU_PIPELINE_LATENCY = 5     # VALU result available after 5 cycles
```

## Next Steps to Reach 100%

### Phase 1: Implement 3 analyzed fixes (~+12 exact → ~290/321)
1. **Post-trans SALU stall**: consume-on-first-use `post_trans_scalar_ready` after trans VALU
2. **s_nop formula variant**: predecessor-dependent formula (VALU→+1, scalar→+0)
3. **s_nop SGPR drain**: non-VCC SGPR writes delay s_nop start

### Phase 2: exp_chain SGPR model (~+8 exact → ~298/321)
1. Track SGPR write-back queue depth/pressure
2. Model depctr flush clearing queue state
3. Fix [10] vs [54] paradox (same instruction, different stall)

### Phase 3: Branch & probe refinement (~+5 exact → ~303/321)
1. SCC propagation for 2nd branch in sequence
2. v_cndmask drain threshold context-sensitivity
3. probe_vmem_chain w1 startup stagger

### Phase 4: Remaining edge cases → 321/321
- s_nop inter-wave interference modeling
- exp_chain VOPD timing after depctr
- data_deps VMEM bypass edge case

## Test Commands
```bash
# All SQTT tests (emulator only, no GPU needed)
DEV=AMD MOCKGPU=1 PYTHON_REMU=1 PROFILE=1 SQTT=1 PYTHONPATH=. .venv/bin/python -m pytest \
  test/amd/test_sqtt_encoder.py test/amd/test_sqttmap.py \
  test/amd/test_sqtt_examples.py test/amd/test_emulator_timing.py \
  test/amd/test_emulator_e2e.py -x -q

# HW accuracy comparison (the main benchmark)
DEV=AMD MOCKGPU=1 PYTHON_REMU=1 PROFILE=1 SQTT=1 PYTHONPATH=. \
  .venv/bin/python extra/sqtt/rigorous_hw_test.py --compare

# HW capture (needs sudo + AMD 7900 XTX)
sudo DEV=AMD AM_RESET=1 VIZ=-2 PROFILE=1 SQTT=1 DEBUG=0 \
  PYTHONPATH=. .venv/bin/python extra/sqtt/rigorous_hw_test.py --capture
```

## Key Files
- `test/mockgpu/amd/emu.py` — Core emulator + SQTT timing model (~850 lines)
- `extra/sqtt/rigorous_hw_test.py` — HW capture + comparison tool (17 kernels)
- `extra/sqtt/captures/rigorous/` — HW reference captures (17 .pkl files)
- `extra/sqtt/capture_discover_ops.py` — Broad instruction capture tool
- `extra/sqtt/answer.md` — ISA team analysis of SGPR timing
- `extra/sqtt/answer2.md` — ISA team analysis of s_nop/VMEM forwarding
- `extra/sqtt/ISA_TIMING_REFERENCE.md` — RDNA3 ISA timing reference
- `extra/sqtt/SESSION_STATUS.md` — This file

## Important Notes
- **NEVER** reference PRs to main tinygrad repo in commits without approval
- Git remote: `https://github.com/2187Nick/tinygrad.git`
- Venv: `.venv/bin/python`
- GPU: AMD 7900 XTX (gfx1100), 96 CUs


## 2026-04-18 — Handoff to 7900 XTX Server

### Current state: 293/321 (91.3%) exact, 309/321 (96.3%) ±2

### What we want
Close the remaining 28 exact mismatches to reach 100% cycle-accurate SQTT match between emulator and real AMD 7900 XTX HW. Preserve existing ±2 accuracy (no regressions on already-matching tokens).

### Analysis of the remaining 28 mismatches

**Wave-dependent SQ arbitration (hard to model deterministically):**
- `probe_cmp_chain` / `probe_branch_cost`: W0 gets store bypass (17cy), W1 doesn't (21cy) — same instruction stream, different arbitration state.
- `probe_cmp_chain` W1: s_nop(15) = 18cy (vs 22cy in W0), store = 21cy (vs 17cy W0). Totals balance to 40cy either way; HW distributes the stall differently per wave.
- `probe_vmem_chain` W1 [2]: post-DRAM v_lshlrev dt=4 in HW vs 1 in EMU — SQ busy issuing W0's stream when W1's waitcnt returns.

**Context-dependent s_nop(15) (16cy vs 20cy):** `probe_sgpr_cmps` 3rd s_nop = 20cy when preceded by 2 others, but 16cy individually. Not trans pipeline (v_sqrt completed 24cy prior). Likely SQ serializes s_nop decode across waves.

**exp_chain VOPD inter-pattern spacing (~12 mismatches):** HW varies VOPD→VOPD spacing 2-4cy based on context that doesn't map cleanly to bank or trans state.

### Tried & reverted in 2026-04-18 session
- Aggressive `vmem_wr_bypass` (-2), set-time narrow bypass, writer-side stall=4, waveend-based bypass — each fixed one mismatch but regressed others.
- Shared-SIMD serialization (`n==2` → 1 VALU/cy): regressed 293 → 214 because `data_deps` shows 2 waves on *different* SIMDs (no serialization). Global rule can't distinguish; needs per-wave SIMD-residency signal.
- VOPD dual-issue ramp (+2cy after 2+ single-issue VALUs): broke exp_chain [26] because we don't model the 768cy SMEM wait that resets HW's ramp state.

### Radeon GPU Profiler (RGP) — is it the unblocker?

**Short answer: probably yes, for the wave-level mismatches.**

RGP uses SQTT internally — so the instruction timestamps themselves won't be different from what we already capture. But RGP exposes **additional metadata** that raw SQTT parsing doesn't give us:

1. **Wave-to-SIMD mapping** — tells us which SIMD each wave is on. This is the missing signal that would let us correctly model shared-SIMD VALU serialization (the big win we couldn't land because of `data_deps` vs `probe_vmem_chain` divergence).
2. **WGP / CU placement per wave** — confirms whether 2-wave WGs share SIMD or not.
3. **Performance counters alongside SQTT** — occupancy, SALU/VALU busy, wavefront launch rate.
4. **Sync/barrier events** — explicit signals for waves waiting on each other.
5. **UI pipeline view** — nice for eyeballing WG/WGP/SE layout at a glance.

**What RGP will NOT solve:**
- The raw instruction timing is still SQTT packets — same data we have.
- If an effect is below the SQTT sample resolution (sub-cycle arbitration state), RGP won't expose it either.

**Recommended capture plan on the server:**
1. Re-capture our 9 probe kernels with RGP (`rgp_capture`) alongside our existing SQTT pickles, so we can cross-reference.
2. For each of the 9 kernels, dump the RGP-exposed wave→SIMD mapping.
3. If the mapping confirms co-residence hypothesis (probe_vmem_chain same-SIMD, data_deps different-SIMD), implement shared-SIMD model gated on that signal.
4. Investigate RGP's SALU/VALU busy counters — they may reveal the s_nop context effect.

### Key files for new session

- `test/mockgpu/amd/emu.py` — emulator + SQTT timing model (~3090 lines). `_simulate_sq_timing()` is the per-wave scheduler.
- `extra/sqtt/rigorous_hw_test.py` — HW capture + comparison harness. `--capture` captures, `--compare` runs emulator and diffs.
- `extra/sqtt/captures/rigorous/*.pkl` — 17 HW reference captures.
- `extra/sqtt/SESSION_STATUS.md` — this file.
- `extra/sqtt/answer.md`, `answer2.md` — prior ISA team analyses of SGPR/s_nop/VMEM timing.
- `extra/sqtt/ISA_TIMING_REFERENCE.md` — RDNA3 timing reference.

### Run commands (on the new 7900 XTX server)

```bash
# Emulator-only validation (no GPU needed):
DEV=AMD MOCKGPU=1 PYTHON_REMU=1 PROFILE=1 SQTT=1 PYTHONPATH=. \
  .venv/bin/python extra/sqtt/rigorous_hw_test.py --compare

# SQTT HW capture (needs root + AMD 7900 XTX):
sudo DEV=AMD AM_RESET=1 VIZ=-2 PROFILE=1 SQTT=1 DEBUG=0 \
  PYTHONPATH=. .venv/bin/python extra/sqtt/rigorous_hw_test.py --capture

# RGP capture (to try): use AMD's rgp CLI with -a attach to the above process,
# or integrate RGP SDK into the capture harness for synchronized dump.
```

### Working tree state at handoff
- Branch: `master` at commit `0145c31b0` (pushed after this session's commits).
- Uncommitted: VGPR bank infrastructure in `test/mockgpu/amd/emu.py` (behavior-neutral, ready to commit).
- Both 2026-04-18 session revert paths have been applied; tree is stable at 293/321.
