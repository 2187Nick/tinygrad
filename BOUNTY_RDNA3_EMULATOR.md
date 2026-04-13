# $1,000 Bounty: Cycle-Accurate RDNA3 Emulator

> "Cycle accurate RDNA3 emulator (add SQTT support to emulator and have it match non DRAM kernels on real hardware perfectly)"  
> — geohot, April 10 2026

**Fork:** https://github.com/2187Nick/tinygrad  
**CI Status:** ✅ Both timing tests PASSING (as of 2026-04-11)

---

## What Is This Bounty? (Plain English)

AMD GPUs run programs called "shaders" or "kernels." When a kernel runs on real hardware, the GPU records a super-detailed log of **exactly which instructions ran and at exactly which clock cycle** — this log format is called **SQTT** (Shader Queue Thread Trace).

Tinygrad already has a **software emulator** of AMD's RDNA3 GPU (the chip inside cards like the RX 7900 XTX). The emulator can run the same GPU kernels on a regular CPU, useful for testing and debugging without needing real hardware.

**The bounty goal:** Make the emulator's SQTT output **exactly match what real RDNA3 hardware produces**, specifically for kernels that only use registers and local memory (no DRAM/global memory).

---

## Key Concepts

### SQTT — The GPU's Flight Recorder

SQTT is AMD's hardware performance tracing mechanism. It records a compressed binary log of every instruction executed by every "wave" (group of 32–64 GPU threads). Each entry contains:

- **What type of instruction ran** (VALU, DS, barrier, etc.)
- **How many clock cycles** elapsed since the previous instruction (the `delta` field)

By summing all the deltas, you get an absolute cycle timestamp for every instruction.

Real hardware captures are stored as `.pkl` files in `extra/sqtt/examples/gfx1100/` (gfx1100 = RDNA3, e.g. RX 7900 XTX).

### Non-DRAM Kernels — Why They're Testable

Global memory (DRAM/VRAM) has unpredictable latency (100–400+ cycles, cache-dependent). **LDS (Local Data Share)** and registers are on-chip with fixed, deterministic latencies. The bounty focuses on these — same result every run, so you can compare against pre-captured hardware traces.

---

## Current Progress

### ✅ FULLY IMPLEMENTED

| Feature | Status | Notes |
|---------|--------|-------|
| SQTT blob format | ✅ | C rocprof decoder accepts emulator blobs |
| Per-instruction cycle costs | ✅ | Via `InstOp` suffix (VALU=4, TRANS=10, etc.) |
| S_DELAY_ALU stall hints | ✅ | Full INSTID0+INSTID1+INSTSKIP decoding |
| s_waitcnt (SOPP + SOPK) | ✅ | LGKM and VM threshold tracking |
| WAVESTART tokens | ✅ | 1-cycle gap between waves confirmed |
| Round-robin wave scheduler | ✅ | Multi-wave kernels scheduled correctly |
| LDS latency | ✅ | `_LDS_LATENCY = 32` cycles (matches GFX1100) |
| Barrier overhead | ✅ | `_BARRIER_FROM_LAST = 18` cycles from last arrival (GFX1100) |
| VALU→DS/VMEM forwarding stall | ✅ | `_VALU_DS_WR_FORWARD = 26` / `_VALU_DS_RD_FORWARD = 22` cycles (GFX1100) |
| WAVEALLOC/DATA_LOST handling | ✅ | Blob structure bugs fixed |
| test_plus_timing_consistent | ✅ PASSING | Plus kernel self-consistency |
| test_sync_timing_consistent | ✅ PASSING | LDS+barrier kernel self-consistency |

### 🔄 IN PROGRESS / REMAINING

| Work Item | Priority | Notes |
|-----------|----------|-------|
| Real hardware comparison test | HIGH | Compare emulator timestamps directly to pkl data |
| VALU→DS accuracy for 2-VALU-before-DS pattern | MEDIUM | ~5 cycle error when 2 VALUs precede a DS |
| SMEM latency model | LOW | Non-DRAM kernels rarely use SMEM; 200-cycle placeholder OK |

---

## What the Tests Check (Two-Level Validation)

### Level 1: Internal Consistency (Currently PASSING ✅)

```python
# test/amd/test_emulator_timing.py
# For every instruction in the emulator's SQTT blob:
assert pkt._time == rocprof_inst.time + rocprof_inst.stall
```

This checks: the SQTT blob the emulator generates is **self-consistent** — the Python delta-encoding matches what the C rocprof decoder reads back. Both decoders look at the same blob and agree on every timestamp.

This test PASSES even if the absolute cycle numbers differ from real hardware.

### Level 2: Real Hardware Match (NEEDED FOR BOUNTY 🎯)

The next test (not yet written) would:
1. Load `extra/sqtt/examples/gfx1100/profile_sync_run_0.pkl` (real GFX1100 trace)
2. Run the same kernel through the emulator
3. Assert per-instruction timestamps match real hardware within tight tolerance

---

## Timing Model: Key Constants

All constants are in `test/mockgpu/amd/emu.py`, calibrated against real GFX1100 SQTT traces:

```python
_LDS_LATENCY       = 32   # LDS read/write: data ready 32 cycles after issue
_SMEM_LATENCY      = 200  # Scalar memory: variable (non-deterministic, not bounty target)
_VMEM_LATENCY      = 300  # Vector memory: variable (non-deterministic, not bounty target)
_BARRIER_FROM_LAST = 18   # Cycles from last wave's barrier issue to release (with 2-cycle RR stagger)
_LDS_SERVICE_COST  = 5    # Cycles the LDS unit is busy servicing one DS write (contention window)
_VALU_DS_WR_FORWARD = 26  # Min cycles from VALU to DS_WR/VMEM_WR issue (address + data forwarding)
_VALU_DS_RD_FORWARD = 22  # Min cycles from VALU to DS_RD/VMEM_RD issue (address-only, shorter pipeline)
_WAVESTART_GAP     = 1    # Cycles between consecutive WAVESTART tokens
_FIRST_INST_GAP    = 2    # Cycles from WAVESTART to first instruction
```

### Real Hardware Evidence (GFX1100, `profile_sync_run_0.pkl`, wave 0)

```
+0     s_load_b64        ← SMEM read (DRAM, non-deterministic)
+895   s_waitcnt         ← SMEM stall (895 cycles: cache miss to DRAM)
+896   v_lshlrev_b32     ← VALU (address computation)
+922   ds_store_b32      ← 26-cycle gap after VALU = _VALU_DS_WR_FORWARD ✓
+955   s_waitcnt         ← LDS stall (33 cycles ≈ _LDS_LATENCY=32 ✓)
+956   s_barrier
+981   v_add_nc_u32      ← 25-cycle barrier overhead (18 from last arrival + 7 inter-wave gap) ✓
+982   v_cmp_gt_u32
+1003  ds_load_b32       ← 22 cycles from v_add (dep VALU) = _VALU_DS_RD_FORWARD ✓, 21 from v_cmp (VOPC, skipped)
+1035  s_waitcnt         ← LDS stall (32 cycles = _LDS_LATENCY ✓)
+1036  v_mov_b32
+1037  v_cndmask_b32
+1038  v_lshlrev_b32
+1065  global_store_b32  ← 27 cycles from last VALU ≈ _VALU_DS_WR_FORWARD ✓
```

---

## Architecture: How the Emulator Works

```
Kernel binary (.bin / ELF)
        │
        ▼
  emit() in emu.py
  ├─ Decodes each RDNA3 instruction
  ├─ Classifies: valu / ds_rd / ds_wr / vmem_rd / vmem_wr / smem / barrier / ...
  ├─ Appends (pkt_cls, kwargs, cat, extra) to per-wave event list
  └─ Zero-cost events: delay_alu, waitcnt, immediate
        │
        ▼
  _simulate_sq_timing()
  ├─ Phase 1: Emit WAVESTART tokens (1 cycle apart per wave)
  ├─ Phase 2: Round-robin scheduler
  │   ├─ Drain zero-cost events (delay_alu → stall hints, waitcnt → LGKM/VM stall)
  │   ├─ Select next wave: min(effective_ready) where:
  │   │   effective_ready = max(ready[i], valu_forward_deadline[i]) for DS/VMEM
  │   ├─ Issue instruction: compute issue_cycle, apply forwarding stall
  │   ├─ Track LGKM/VM pending ops for waitcnt
  │   └─ Barrier: stall waves until all arrive, release with +25 cycle overhead
  └─ Returns: [(cycle, wave_id, pkt_cls, kwargs), ...]
        │
        ▼
  finalize() → SQTT blob (delta-encoded binary)
        │
        ▼
  rocprof-trace-decoder (C library) → per-instruction timestamps
        │
        ▼
  test assertion: pkt._time == rocprof_inst.time + rocprof_inst.stall
```

### VALU→DS Forwarding: Why It Matters

On RDNA3, the result of a VALU instruction (e.g. computing an LDS address) cannot be used by a DS instruction for ~26 clock cycles. The hardware's scoreboard enforces this — it's the register forwarding path latency.

**Without this model:** emulator would schedule DS ops ~4 cycles after VALU (just the round-robin slot). Real hardware: 26 cycles.

**With this model:** the stalled wave is *skipped* during selection, other waves fill the gap naturally, and the DS issues at the correct cycle. No other waves are blocked.

---

## How to Run / Test (CI, No GPU Needed)

### GitHub Actions CI

The test suite runs automatically on every push to `2187Nick/tinygrad`. The key job is `rdna3-emu`:

```yaml
# .github/workflows/test.yml
rdna3-emu:
  - Installs rocprof-trace-decoder (AMD CPU-only library)
  - Runs: DEV=AMD MOCKGPU=1 PYTHON_REMU=1 PROFILE=1 SQTT=1 python -m pytest test/amd/
```

### Key Test Files

```
test/amd/test_emulator_timing.py   # ← Our bounty tests (both PASSING ✅)
  test_plus_timing_consistent      # elementwise add kernel
  test_sync_timing_consistent      # LDS + barrier kernel

test/amd/test_sqttmap.py           # rocprof_inst_traces_match() helper
test/amd/test_sqtt_examples.py     # Real pkl file packet parsing
```

### Running Locally (Linux with ROCm only)

```bash
git clone https://github.com/2187Nick/tinygrad && cd tinygrad
pip install -e ".[testing]"
sudo PYTHONPATH="." ./extra/sqtt/install_rocprof_decoder.py

# Run timing tests
DEV=AMD MOCKGPU=1 PYTHON_REMU=1 PROFILE=1 SQTT=1 \
  python -m pytest test/amd/test_emulator_timing.py -v
```

---

## Next Steps Toward Bounty Completion

### 1. Real Hardware Comparison Test (High Priority)

Add `test_hw_timing_match` to `test/amd/test_emulator_timing.py`:
- Load `extra/sqtt/examples/gfx1100/profile_sync_run_0.pkl`
- Extract wave 0 relative timestamps (first instruction = 0)
- Run same kernel through emulator (MOCKGPU=1)
- Assert each emulated timestamp matches real HW within ±2 cycles

This is the definitive bounty validation — it proves the emulator matches hardware, not just itself.

### 2. Refine VALU→DS for Sequential VALU Patterns

Current model uses `last_valu + 26`. The real hardware uses the specific DEP VALU + a skip-count reduction:
- 0 instructions between dep VALU and DS: 26 cycles (✓ matches)
- 1 instruction between (v_cmp which doesn't write the dep register): 22 cycles (model gives ~26, ~4 off)

The fix requires tracking the "anchor VALU" (first VALU after last DS/VMEM), not the "last VALU", plus counting VALUs issued since the anchor.

### 3. Submit to Tinygrad (When Ready)

Once the real hardware comparison test passes:
1. Review final diff with user
2. Open PR to `tinygrad/tinygrad` (NOT done yet — only pushing to fork)
3. Claim bounty 🏆

---

## File Map

| File | Purpose |
|------|---------|
| `test/mockgpu/amd/emu.py` | Core emulator: `emit()`, `_simulate_sq_timing()`, timing constants |
| `test/amd/test_emulator_timing.py` | Bounty tests: timing consistency checks |
| `test/amd/test_sqttmap.py` | `rocprof_inst_traces_match()` helper |
| `test/amd/test_sqtt_examples.py` | Real pkl packet parsing tests |
| `extra/sqtt/examples/gfx1100/` | Real GFX1100 hardware captures |
| `tinygrad/renderer/amd/sqtt.py` | SQTT packet format definitions |
| `tinygrad/runtime/autogen/amd/rdna3/enum.py` | `SOPPOp`, `SOPKOp` enums (RDNA3-specific numbering) |

---

## Resources

- **Bounty spreadsheet** — https://docs.google.com/spreadsheets/d/1WKHbT-7KOgjEawq5h5Ic1qUWzpfAzuD_J06N1JwOCGs/
- **AMD RDNA3 ISA** — https://gpuopen.com/rdna3-isa/ (also at `amd-rdna3-isa.pdf`)
- **S_DELAY_ALU spec** — https://releases.llvm.org/19.1.0/docs/AMDGPU/gfx11_delay.html
- **ROCProfiler** — https://github.com/ROCm/rocprofiler

---

*Working fork: https://github.com/2187Nick/tinygrad — never push to upstream tinygrad until bounty is ready to claim.*
