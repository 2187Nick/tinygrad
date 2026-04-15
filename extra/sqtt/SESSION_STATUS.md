# RDNA3 Cycle-Accurate Emulator — Session Status

## Bounty Goal
$1,000 bounty: Make tinygrad's software GPU emulator produce cycle-accurate instruction timing matching real AMD 7900 XTX hardware, validated via SQTT (Shader Queue Thread Trace).

## Current Accuracy: 187/250 exact (74.8%), 221/250 ±2 (88.4%)

All 63 SQTT tests pass (encoder, map, examples, timing, E2E).

### Per-Kernel Breakdown

| Kernel | Exact | ±2 | Notes |
|---|---|---|---|
| lds_sync | 18/26 (69.2%) | 22/26 (84.6%) | 2-wave LDS+barrier+store |
| data_deps | 9/10 (90.0%) | 9/10 (90.0%) | 2-wave VALU chain |
| exp_chain | 89/112 (79.5%) | 106/112 (94.6%) | Transcendental chain (1 wave) |
| matmul_medium | 34/59 (57.6%) | 43/59 (72.9%) | HW=6 waves, EMU=1 wave |
| elementwise | 14/16 (87.5%) | 15/16 (93.8%) | b128 store (1 wave) |
| plus | 11/13 (84.6%) | 13/13 (100%) | Simple add (1 wave) |
| cast | 12/14 (85.7%) | 13/14 (92.9%) | Type conversion (1 wave) |

4 kernels skipped due to PC mismatch (MOCKGPU compiles different binary): softmax, layernorm, reduce256, reduce_large.

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

## Known Remaining Mismatches

### Fixable (Tomorrow)
1. **PC mismatch on 4 kernels** — softmax, layernorm, reduce256, reduce_large compile to different binary offsets under MOCKGPU. Fix: add instruction-name-based comparison fallback (PCs differ by constant offset, instruction sequences are identical). This would add ~1000+ new comparison points.
2. **matmul_medium wave count** — HW=6 waves, EMU=1 wave. MOCKGPU launches different wave count. Need to investigate MOCKGPU CU configuration.
3. **SMEM variable latency** — matmul_medium [2] HW=9, EMU=1. S_load after s_load has cache-hit latency. Quick win: detect SMEM→SMEM sequences and use shorter latency.

### Harder (Need Register Tracking)
4. **SGPR dependency stalls** — exp_chain [10]/[34]/[52]: v_cmp chains reading s[0] written by previous v_cmp. Would need SGPR write-to-read dependency tracking.
5. **False delay_alu stalls** — exp_chain [51]: VOPD v[2:3] has VALU_DEP on VOPD v[0:1] — no register overlap but EMU applies stall. Needs register-level tracking.
6. **Multi-wave VMEM contention** — lds_sync [13]: HW=27/31, EMU=21. Post-barrier VMEM port serialization across waves.
7. **global_store_b128 width variance** — HW=20-25 depending on kernel context, EMU=24 fixed.

## Test Commands
```bash
# All 63 SQTT tests (emulator only, no GPU needed)
DEV=AMD MOCKGPU=1 PYTHON_REMU=1 PROFILE=1 SQTT=1 .venv/bin/python -m pytest test/amd/test_sqtt_encoder.py test/amd/test_sqttmap.py test/amd/test_sqtt_examples.py test/amd/test_emulator_timing.py test/amd/test_emulator_e2e.py -v

# HW capture (needs real GPU + sudo + profile_standard)
echo '<password>' | sudo -S bash -c 'echo profile_standard > /sys/class/drm/card1/device/power_dpm_force_performance_level'
sudo DEV=AMD AM_RESET=1 VIZ=-2 PROFILE=1 SQTT=1 DEBUG=0 PYTHONPATH=. .venv/bin/python extra/sqtt/rigorous_hw_test.py --capture

# EMU compare against HW captures
DEV=AMD MOCKGPU=1 PYTHON_REMU=1 PROFILE=1 SQTT=1 PYTHONPATH=. .venv/bin/python extra/sqtt/rigorous_hw_test.py --compare
```

## Tomorrow's Plan
1. **Fix PC mismatch comparison** — Add instruction-name fallback to unlock softmax, layernorm, reduce256, reduce_large (hundreds of new comparison points).
2. **SMEM cache-hit latency** — Quick win for matmul_medium.
3. **VMEM port contention model** — Shared VMEM port across waves for lds_sync post-barrier stores.
4. **Capture more kernels** — wave_sync, where, conv2d, attention patterns.
5. **Target 85%+ exact accuracy** across all kernels.

## Commits (on fork: https://github.com/2187Nick/tinygrad.git)
1. `956c54812` — SQTT deep dive docs + E2E tests
2. `e274ff784` — Wave-count VALU stalls + beam search POC + HW validation
3. `841831190` — Split DS/VMEM forwarding + rigorous HW test tool
4. `ce7671623` — VMEM forwarding, transcendental pipeline, s_nop timing
5. `9949bd1e6` — Time-based delay_alu, parallel trans, VMEM drain, width-dep forwarding
