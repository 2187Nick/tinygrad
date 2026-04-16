# RDNA3 SQTT Emulator — Accuracy Progress

## Current State
- **Rigorous suite (HW vs EMU):** 294/347 exact (84.7%), 333/347 within ±2 (96.0%)
- **Unit tests:** 40/40 passing
- **Target:** 100% within ±2 cycles on non-DRAM deltas

## Session Breakthroughs

### 1. Wave Independence (major, this session)
**Problem:** The emulator's `clock` variable was incremented on every VALU issue across all waves, causing later waves to stagger artificially behind earlier ones. On multi-wave captures the delta accumulated ~16cy per wave, distorting timing.

**RDNA3 reality:** Each wave runs on its own SIMD; VALU issue is per-SIMD, not globally serialized.

**Fix:** In `test/mockgpu/amd/emu.py::_simulate_sq_timing()`, compute per-wave issue as `issue_cycle = ready[i]` (removed the `max(clock, ready[i])` gate that modeled a shared issue port).

**Impact:** +6 exact matches on rigorous suite; fixed root-cause of probe_cmp_chain/branch_cost misalignment.

### 2. Post-Barrier Pipeline Refill (this session)
**Problem:** Post-barrier `v_add_nc_u32` at PC=0x138 showed HW=3cy but EMU=1cy after the wave-independence fix (previously matched via cross-wave clock contamination).

**Fix:** `ready[i] = release_cycle + idx * 2 + 2` (added 2cy post-barrier pipeline refill constant).

**Impact:** lds_sync 19/26 → 21/26 exact.

### 3. s_nop Drains VALU→VMEM Forwarding Deadlines (answer2 Bonus)
Per ISA team, `s_nop N` caps pending VALU→VMEM forwarding deadlines at `nop_stamp + 5`. Applied at `emu.py:287-310`.

## Per-Kernel Breakdown (rigorous suite)

| Kernel              | Exact        | ±2           | Notes |
|---------------------|--------------|--------------|-------|
| elementwise         | 16/16 100%   | 16/16 100%   | ✅    |
| plus                | 13/13 100%   | 13/13 100%   | ✅    |
| cast                | 14/14 100%   | 14/14 100%   | ✅    |
| probe_vmem_chain    | 21/22 95.5%  | 21/22 95.5%  | 1 wave-1 outlier |
| data_deps           |  9/10 90.0%  |  9/10 90.0%  | 1 HW-variance pc |
| exp_chain           | 99/112 88.4% | 111/112 99.1%| transcendental dispatch |
| probe_cmp_chain     | 39/44 88.6%  | 42/44 95.5%  | SGPR commit edge-case |
| lds_sync            | 21/26 80.8%  | 23/26 88.5%  | LDS multi-wave staging |
| probe_sgpr_cmps     | 46/64 71.9%  | 60/64 93.8%  | ← biggest gap, SGPR model |
| probe_branch_cost   | 16/26 61.5%  | 24/26 92.3%  | inherent HW variance |

## Remaining Opportunities (prioritized)

### A. SGPR `sgpr_readable_time` refactor (biggest gain)
`probe_sgpr_cmps` 46/64 → target 60+. Current model over/under-stalls dependent v_cndmask based on which SGPR is read. ISA team notes src2 on VOPC is unused, and v_cndmask reading non-VCC SGPR has 6cy latency + drain stall.

### B. Capture noise reduction
Probes show ~1cy of HW variance across runs. Capture 3-5x per probe, take median per PC, rebuild HW ground truth.

### C. Per-wave-group barrier timing
`lds_sync` wave-1 barrier entry shows HW=6 EMU=2. Needs wave-position-aware release timing.

### D. probe_branch_cost HW variance (likely inherent)
HW shows ±1cy jitter on s_cbranch_scc1 not-taken stamp latency. Probably near the noise floor.

## Key Files
- `test/mockgpu/amd/emu.py` — emulator core
- `extra/sqtt/rigorous_hw_test.py` — HW-vs-EMU harness
- `extra/sqtt/captures/rigorous/*.pkl` — HW ground-truth traces (now checked in)
- `test/amd/test_emulator_timing.py` / `test_emulator_e2e.py` — unit tests

## Test Commands
```bash
# Rigorous comparison
DEV=AMD MOCKGPU=1 PYTHON_REMU=1 PROFILE=1 SQTT=1 PYTHONPATH=. .venv/bin/python extra/sqtt/rigorous_hw_test.py --compare

# Unit tests
DEV=AMD MOCKGPU=1 PYTHON_REMU=1 PROFILE=1 SQTT=1 PYTHONPATH=. .venv/bin/python -m pytest test/amd/test_emulator_timing.py test/amd/test_emulator_e2e.py -q

# HW capture (requires 7900 XTX + sudo)
sudo DEV=AMD AM_RESET=1 VIZ=-2 PROFILE=1 SQTT=1 DEBUG=0 PYTHONPATH=. .venv/bin/python extra/sqtt/rigorous_hw_test.py --capture
```
