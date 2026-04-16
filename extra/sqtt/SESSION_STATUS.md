# RDNA3 Cycle-Accurate Emulator — Session Status

## Bounty Goal
$1,000 bounty: Make tinygrad's software GPU emulator produce cycle-accurate instruction timing matching real AMD 7900 XTX hardware, validated via SQTT (Shader Queue Thread Trace).

## Current Accuracy: 203/233 exact (87.1%), 228/233 ±2 (97.9%)

All 63 SQTT tests pass (encoder, map, examples, timing, E2E).

### Per-Kernel Breakdown

| Kernel | Exact | Total | Rate | ±2 Rate |
|-------------|-------|-------|--------|---------|
| elementwise | 16 | 16 | 100.0% | 100.0% |
| cast | 14 | 14 | 100.0% | 100.0% |
| plus | 12 | 13 | 92.3% | 100.0% |
| data_deps | 9 | 10 | 90.0% | 90.0% |
| exp_chain | 99 | 112 | 88.4% | 99.1% |
| layernorm | 34 | 42 | 81.0% | 100.0% |
| lds_sync | 19 | 26 | 73.1% | 88.5% |

Progress: Started at 74.8% → now 87.1% exact (+16 points).

## What We Built

### 1. Timing Model Improvements (test/mockgpu/amd/emu.py)
- **Time-based delay_alu stalls**: VALU_DEP computed from actual issue times (5-cycle pipeline) instead of fixed additive values. Correctly produces 0 stall after long waits (VMEM/SMEM). **+15 exact matches.**
- **Parallel trans ALU**: v_exp/v_log run in parallel with VALU (issue_cost=1, pipeline occupancy enforced separately via trans_pipe_avail).
- **VMEM drain tracking**: s_nop before s_endpgm waits ~14 cycles for VMEM pipeline acceptance.
- **Width-dependent VMEM forwarding**: global_store_b128 adds +3 cycles for wider VGPR reads.
- **Trans pipeline latency**: Calibrated to 27 cycles (validated from HW v_log→depctr).
- **VMEM forwarding calibration**: _VALU_VMEM_WR_FORWARD=21 (HW-validated).
- **s_waitcnt_depctr tracking**: Transcendental pipeline modeling for depctr stalls.
- **s_nop(N) stall modeling**: N+1 cycles of stall.
- **Wave-count-dependent VALU stalls**: Single-wave kernels need +1 stall vs 2-wave.
- **Split DS/VMEM forwarding**: Separate constants for LDS write (26), LDS read (22), VMEM write (21), VMEM read (22).

### 2. SQTT Hardware Capture (tinygrad/runtime/ops_amd.py)
- **wgp_sel auto-detect**: `(cu_per_simd_array - 1) // 2` = 3 (wgp_sel=0 DISABLES tracing!)
- **Token mask overflow fix**: Prevented token_exclude from corrupting ttrace_exec bit.
- **profile_standard enforcement**: Required for clock gating suppression.
- **Auto-detect traced CU**: map_insts finds traced CU from first WAVESTART.
- **SQTT_ITRACE_SE_MASK**: All 6 SEs enabled (0x3f).
- **Hiwater**: Changed from 1 to 5 (Mesa match).
- **s_sendmsg skip**: Added S_SENDMSG/S_SENDMSGHALT to _SOPP_SKIP (async fire-and-forget, no SQTT packet).

### 3. Test Infrastructure
- **test/amd/test_emulator_e2e.py**: 12 E2E tests, 45+ subtests across 10 kernel types.
- **extra/sqtt/rigorous_hw_test.py**: Multi-kernel HW capture + emulator comparison (13 kernels, --capture and --compare modes).
- **extra/sqtt/hw_validation_suite.py**: Earlier HW capture tool.
- **extra/sqtt/beam_search_poc.py**: SQTT-guided beam search POC.
- **extra/sqtt/SQTT_DEEP_DIVE.md**: Comprehensive SQTT documentation.

### 4. HW Reference Captures (extra/sqtt/captures/rigorous/)
Captured from real 7900 XTX with profile_standard power mode:
lds_sync, data_deps, softmax, layernorm, exp_chain, reduce256, reduce_large, matmul_medium, elementwise, plus, cast.

## Key Architecture Discoveries

1. **RDNA3 trans ALU runs in PARALLEL with VALU** — After v_exp issues, VALU can issue 1 cycle later. The 4-cycle trans pipeline only blocks subsequent TRANS instructions.
2. **VMEM drain = 14 cycles** — After global_store/load issues, s_nop waits ~14 cycles for VMEM pipeline acceptance.
3. **Time-based delay_alu is the correct model** — Fixed stalls incorrectly add cycles even after long waits. Time-based: `max(0, dep_issue + 5 - current_cycle)` gives 0 stall when enough time elapsed.
4. **wgp_sel=0 DISABLES instruction tracing** on GFX11 — Must use wgp_sel≥1.
5. **profile_standard power mode REQUIRED** — Dynamic clock gating suppresses SQ instruction tokens without it.
6. **Wave placement is non-deterministic** — 1-wave kernels only land on traced CU ~15% of the time. Must retry.
7. **S_SENDMSG is async fire-and-forget** — No SQTT packet emitted; rocprof returns phantom PC=0 entry.

## Timing Constants (Calibrated from HW)

```python
_LDS_RD_LATENCY = 31          # LDS read result latency
_LDS_WR_LATENCY = 33          # LDS write completion latency
_SMEM_LATENCY = 200            # Scalar memory latency
_VMEM_LATENCY = 300            # Vector memory (DRAM) latency
_BARRIER_FROM_LAST = 6         # Barrier overhead from last wave arrival
_LDS_SERVICE_COST = 6          # Per-wave LDS serialization cost
_VALU_DS_WR_FORWARD = 26       # VALU→DS write forwarding stall
_VALU_DS_RD_FORWARD = 22       # VALU→DS read forwarding stall
_VALU_VMEM_WR_FORWARD = 21     # VALU→VMEM write forwarding (HW-validated)
_VALU_VMEM_RD_FORWARD = 22     # VALU→VMEM read forwarding
_VMEM_DRAIN_CYCLES = 14        # VMEM pipeline acceptance time
_TRANS_PIPE_CYCLES = 4         # Trans ALU pipeline occupancy
_TRANS_PIPELINE_LATENCY = 27   # Transcendental result latency for depctr
_WAVESTART_GAP = 1             # Gap between wave starts
_FIRST_INST_GAP = 2            # Gap from wavestart to first instruction
VALU_PIPELINE_LATENCY = 5      # VALU result available after 5 cycles
```

## Remaining 30 Mismatches — Detailed Analysis

### exp_chain (13 mismatches) — Biggest opportunity
Three repeating SGPR blocks with identical instruction patterns but different stall
distributions — the hardest unsolved problem:
- **Block 1 [10]**: HW=4 EMU=1 — SGPR write buffer contention after VOPC
- **Block 2 [31]**: HW=3 — VGPR readiness (v[2] from VOPD, lat=5)
- **Block 3 [54]**: HW=1 EMU=2 — No SGPR stall (different context)
- **[26],[51]**: HW=1 EMU=3 — VOPD after depctr, unknown stall source
- Fixing Block 1 root cause would cascade-fix 7+ downstream mismatches
- Debug instrumentation (TIMING_DEBUG=1) is ready to trace exact stall sources

### layernorm (8 mismatches) — All within ±2
Agent analysis suggests 3 constant adjustments:
- `_VALU_DS_WR_FORWARD: 26→25` (fix [10] ds_store_b32)
- `_LDS_B128_RD_SERVICE: 19→18` (fix [30] s_waitcnt)
- `_VALU_VMEM_WR_FORWARD: 21→22` (fix [41] global_store_b32)
- Risk: may regress other kernels using same constants — test carefully

### lds_sync (7 mismatches) — Multi-wave scheduling
- Wave 1 barrier/waitcnt split issues
- Some ±4 errors (larger than other kernels)

### data_deps (1) + plus (1) — Low priority

## What We Built

### Timing Model Improvements (test/mockgpu/amd/emu.py)
- **Time-based delay_alu stalls** (+15 exact)
- **Parallel trans ALU** — v_exp/v_log run in parallel with VALU
- **VMEM drain tracking** — 14cy pipeline acceptance
- **Width-dependent VMEM forwarding** — b128 adds +3cy
- **Split DS/VMEM forwarding** — separate LDS write/read, VMEM write/read constants
- **b128 stagger model** — serialized LDS b128 detection + 17cy stagger
- **Slow-fresh VGPR tracking** — first-use VGPR latency from register file
- **VMEM drain overlap** — proper drain timing for VMEM stores
- **v_cndmask non-VCC SGPR stall** — 6cy latency + pending write drain
- **VOP3_SDST src2 skip** — HW doesn't read phantom src2
- **Wave-count-dependent VALU stalls** — single-wave vs multi-wave
- **Trans pipeline latency** — 27cy calibrated from HW

### Tools & Infrastructure
- `rigorous_hw_test.py` — multi-kernel HW capture + emulator comparison
- `discover_ops.py` — instruction category discovery across tinygrad kernels
- `ISA_TIMING_REFERENCE.md` — extracted timing data from RDNA3 ISA manual
- HW captures for 7 kernels in `captures/rigorous/`
- Debug trace (TIMING_DEBUG=1 env var) for per-instruction stall breakdown

## Next Steps to Reach 100%

### Phase 1: Low-hanging fruit (+4-8 exact → ~210/233)
1. Implement layernorm constant adjustments (test for regressions)
2. Debug exp_chain [26],[51] VOPD stall source via TIMING_DEBUG
3. Trace exp_chain Block 1 [10] root cause

### Phase 2: SGPR timing model (+3-7 exact → ~215/233)
1. The 3 identical blocks having different stalls suggests a stateful mechanism
2. Implement dynamic SGPR write buffer tracking
3. Cascade fixes for [13]-[17],[19]

### Phase 3: lds_sync multi-wave (+2-3 exact → ~218/233)
1. Investigate Wave 1 barrier scheduling
2. Fix waitcnt timing in multi-wave context

### Phase 4: Expand kernel coverage
- Add softmax, matmul_medium, reduce_large, attention kernels
- Capture HW traces, compare, fix new mismatch categories

## Test Commands
```bash
# All 63 SQTT tests (emulator only, no GPU needed)
DEV=AMD MOCKGPU=1 PYTHON_REMU=1 PROFILE=1 SQTT=1 .venv/bin/python -m pytest \
  test/amd/test_sqtt_encoder.py test/amd/test_sqttmap.py \
  test/amd/test_sqtt_examples.py test/amd/test_emulator_timing.py \
  test/amd/test_emulator_e2e.py -x -q

# HW accuracy comparison
DEV=AMD MOCKGPU=1 PYTHON_REMU=1 PROFILE=1 SQTT=1 PYTHONPATH=. \
  .venv/bin/python extra/sqtt/rigorous_hw_test.py --compare

# HW capture (needs sudo + AMD 7900 XTX)
echo '<password>' | sudo -S bash -c \
  'echo profile_standard > /sys/class/drm/card1/device/power_dpm_force_performance_level'
echo '<password>' | sudo -S DEV=AMD AM_RESET=1 VIZ=-2 PROFILE=1 SQTT=1 DEBUG=0 \
  PYTHONPATH=. .venv/bin/python extra/sqtt/rigorous_hw_test.py --capture

# Debug trace for stall analysis
TIMING_DEBUG=1 DEV=AMD MOCKGPU=1 PYTHON_REMU=1 PROFILE=1 SQTT=1 PYTHONPATH=. \
  .venv/bin/python /tmp/trace_stalls2.py
```

## Key Files
- `test/mockgpu/amd/emu.py` — Core emulator + SQTT timing model
- `extra/sqtt/rigorous_hw_test.py` — HW capture + comparison tool
- `extra/sqtt/captures/rigorous/` — HW reference captures (7 kernels)
- `extra/sqtt/ISA_TIMING_REFERENCE.md` — RDNA3 ISA timing reference
- `extra/sqtt/answer.md` — ISA team analysis of SGPR timing
- `extra/sqtt/examples/discover_ops.py` — Instruction category scanner

## Important Notes
- **NEVER** reference PRs to main tinygrad repo in commits without approval
- Git remote: `https://github.com/2187Nick/tinygrad.git`
- Venv: `.venv/bin/python`
- GPU: AMD 7900 XTX (gfx1100), 96 CUs
