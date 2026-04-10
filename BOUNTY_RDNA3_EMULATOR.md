# $1,000 Bounty: Cycle-Accurate RDNA3 Emulator

> "Cycle accurate RDNA3 emulator (add SQTT support to emulator and have it match non DRAM kernels on real hardware perfectly)"  
> — geohot, April 10 2026

---

## What Is This Bounty? (Plain English)

AMD GPUs run programs called "shaders" or "kernels." When a kernel runs on real hardware, the GPU records a super-detailed log of **exactly which instructions ran and at exactly which clock cycle** — this log format is called **SQTT** (Shader Queue Thread Trace).

Tinygrad already has a **software emulator** of AMD's RDNA3 GPU (the chip inside cards like the RX 7900 XTX). The emulator can run the same GPU kernels on a regular CPU, which is incredibly useful for testing and debugging without needing real hardware.

**The problem:** The emulator runs the right *instructions* in the right *order*, but it doesn't know the right *timing*. Every instruction is stamped with `delta=1` (1 cycle apart), while real hardware has varying cycle counts per instruction — some take 1 cycle, others 4, 16, or more.

**The bounty goal:** Fix the emulator's timing so the SQTT output it generates **exactly matches what real RDNA3 hardware produces**, specifically for kernels that only use registers and local memory (no DRAM/global memory).

---

## Key Concepts

### SQTT — The GPU's Flight Recorder

SQTT is AMD's hardware performance tracing mechanism. It records a compressed binary log of every instruction executed by every "wave" (group of 32 GPU threads). Each entry contains:

- **What type of instruction ran** (arithmetic, memory, branch, barrier, etc.)
- **How many clock cycles** elapsed since the previous instruction (the `delta` field)

By summing all the deltas, you get an absolute cycle timestamp for every instruction. This lets you see exactly when pipeline stalls happened, how long memory operations took, etc.

Real hardware captures are stored as `.pkl` files in `extra/sqtt/examples/gfx1100/` (gfx1100 = RDNA3).

### Cycle Accuracy

"Cycle accurate" means the emulator's simulated clock ticks match the real hardware clock ticks, instruction by instruction.

| Instruction Type | Real RDNA3 Cycles | Current Emulator |
|-----------------|-------------------|------------------|
| SALU (scalar ALU) | ~1 cycle | delta=1 ✗ |
| VALU standard | ~4 cycles | delta=1 ✗ |
| VALU transcendental (exp, log, sqrt) | ~4 cycles | delta=1 ✗ |
| 64-bit shifts | ~2 cycles | delta=1 ✗ |
| 64-bit MAD | ~4 cycles | delta=1 ✗ |
| 64-bit FP ops | ~16 cycles | delta=1 ✗ |
| LDS read (DS_READ) | ~2 cycles | delta=1 ✗ |
| LDS write (DS_WRITE) | ~2-6 cycles | delta=1 ✗ |

### Non-DRAM Kernels — The "Easy" Case

Global memory (DRAM/VRAM) has unpredictable latency: it can take anywhere from 100 to 400+ cycles depending on cache state, memory pressure, etc. This makes cycle-level matching against real hardware extremely difficult.

**LDS (Local Data Share)** and **registers**, however, are on-chip and have deterministic, fixed latencies. Kernels that only use these are:
- ✅ Perfectly deterministic — same cycle count every run
- ✅ Testable without DRAM: matrix multiply with data in LDS, register-only compute
- ✅ What this bounty focuses on — match these *perfectly*

---

## Current State of the Codebase

### What Already Exists

The emulator infrastructure is already quite mature:

```
test/mockgpu/amd/emu.py          # Core RDNA3 Python emulator
tinygrad/renderer/amd/sqtt.py    # SQTT packet encoder/decoder
test/amd/test_sqtt_examples.py   # Tests vs real hardware captures
test/amd/test_sqttmap.py         # Validates emulator SQTT vs rocprof
extra/sqtt/examples/gfx1100/     # Real RDNA3 hardware captures (.pkl files)
extra/sqtt/                      # SQTT header and tools
```

The emulator:
1. Decodes RDNA3 binary opcodes
2. Runs each instruction via tinygrad's CPU backend  
3. **Already emits SQTT packets** — but with wrong cycle timing (`delta=1` everywhere)

### What Needs to Change

**File:** `test/mockgpu/amd/emu.py`, function `_init_sqtt_encoder()`

Currently:
```python
_emit_nibbles(nibbles, IMMEDIATE, delta=1, wave=w)     # WRONG - always 1
_emit_nibbles(nibbles, VALUINST, delta=1, wave=w)      # WRONG - always 1
_emit_nibbles(nibbles, INST, delta=1, wave=w, op=...)  # WRONG - always 1
```

Needs:
```python
_emit_nibbles(nibbles, IMMEDIATE, delta=actual_cycles, wave=w)   # cycle-accurate
_emit_nibbles(nibbles, VALUINST, delta=actual_cycles, wave=w)    # cycle-accurate
_emit_nibbles(nibbles, INST, delta=actual_cycles, wave=w, op=...) # cycle-accurate
```

The `actual_cycles` value must account for:
1. **Instruction base latency** (1, 2, 4, or 16 cycles per type)
2. **Pipeline stalls** from S_DELAY_ALU hints (the compiler explicitly encodes how many cycles to wait)
3. **Dependency hazards** (result of one instruction used immediately by the next)
4. **S_WAITCNT** effects (waiting for outstanding memory operations)

---

## How to Test Without Real AMD Hardware

### The Key Insight

Real hardware captures (`.pkl` files) are **already committed to the repo**. These contain binary SQTT blobs recorded from a real gfx1100 GPU. The testing approach:

1. Run a kernel through the emulator → get emulator's SQTT output
2. Compare against the reference `.pkl` captured from real hardware
3. Assert every instruction's `time + stall` matches exactly

### Test Command (Runs in GitHub Actions CI)

```bash
# Install the ROCm profiler decoder (AMD provides this as a library)
sudo PYTHONPATH="." ./extra/sqtt/install_rocprof_decoder.py

# Run the full AMD test suite including SQTT cycle accuracy tests
python -m pytest test/amd/ --durations 20

# Run just the cycle-accuracy comparison test
python -m pytest test/amd/test_sqttmap.py::TestSQTTMapRDNA3 -v
```

### What the Tests Check

**`test_sqttmap.py::TestSQTTMapRDNA3::test_rocprof_inst_traces_match`**

This is the critical test. For each instruction in each kernel:
```python
# This must pass for every instruction:
assert pkt._time == rocprof_inst.time + rocprof_inst.stall
```

Where:
- `pkt._time` = cumulative cycle count from the emulator's SQTT output
- `rocprof_inst.time` = when instruction was issued on real hardware  
- `rocprof_inst.stall` = how many cycles it was stalled

**GitHub CI workflow:** `.github/workflows/test.yml` runs the `rdna3-emu` job which executes `python -m pytest -n=auto test/amd/` — this includes all SQTT tests.

### CI Architecture (No GPU Required)

```
GitHub Actions Runner (Ubuntu, no GPU)
    │
    ├─ Installs rocprof-trace-decoder (AMD library, CPU-only decoder)
    ├─ Runs: python -m pytest test/amd/
    │
    ├─ test_sqtt_examples.py  ──→ Loads real .pkl files, verifies packet parsing
    ├─ test_sqttmap.py        ──→ Runs emulator, compares SQTT vs real hardware
    └─ test_sqtt_encoder.py   ──→ Unit tests for packet encoding
```

The comparison happens **entirely in software** — the real GPU traces are pre-captured and stored in git.

---

## Technical Deep Dive: How the Timing Works

### The SQTT Delta/Time Model

Every SQTT packet has a `_time` field = cumulative sum of all previous `delta` values.

The test asserts: `pkt._time == rocprof_inst.time + rocprof_inst.stall`

Where:
- `rocprof_inst.time` = the cycle when the instruction was dispatched/issued
- `rocprof_inst.stall` = stall cycles incurred (dependency wait + pipeline stall)
- Together: `time + stall` = the "effective" completion/stamp cycle

**The emitter needs to produce deltas such that `_time` matches `time + stall` from real hardware.**

### Cycle Counts Are Encoded in InstOp Names

Commit `#15473` added cycle counts to the `InstOp` enum names — the suffix `_N` is the cycle count:

| InstOp | Value | Meaning | Cycles |
|--------|-------|---------|--------|
| `VALUT_4` | 0xb | transcendental (exp/log/sqrt/rcp/sin/cos) | 4 |
| `VALUB_2` | 0xd | 64-bit shifts | 2 |
| `VALUB_4` | 0xe | 64-bit multiply-add | 4 |
| `VALUB_16` | 0xf | 64-bit FP ops | 16 |
| `FLAT_RD_2` | 0x1c | flat load | 2 |
| `FLAT_WR_3..6` | 0x1d-0x20 | flat stores (by size) | 3-6 |
| `SGMEM_RD_1` | 0x21 | global load (saddr=SGPR) | 1 |
| `SGMEM_RD_2` | 0x22 | global load (saddr=NULL) | 2 |
| `LDS_RD` | 0x29 | LDS read (no cycle in name) | ? |
| `LDS_WR_2..5` | 0x2b-0x2e | LDS writes (by size) | 2-5 |

Standard VALU (`VALUINST`, no suffix) = **4 cycles** (RDNA3 VALU throughput)  
SALU = **1 cycle** base  
IMMEDIATE (s_nop, s_waitcnt) = **1 cycle**

### S_DELAY_ALU: The Compiler's Explicit Stall Hint

`S_DELAY_ALU` is the RDNA3 instruction the compiler inserts to encode exactly how many stall cycles are needed between dependent instructions. **It does NOT produce a SQTT packet** (skipped on hardware and in emulator), but its stall info must be added to the NEXT instruction's delta.

**`simm16` encoding** (from [LLVM docs](https://releases.llvm.org/19.1.0/docs/AMDGPU/gfx11_delay.html)):

```
bits [3:0]  = INSTID0  — dependency for next instruction (0-11)
bits [6:4]  = INSTSKIP — which instruction INSTID1 applies to (0=SAME, 1=NEXT, 2=SKIP_1 ... 5=SKIP_4)
bits [10:7] = INSTID1  — dependency for instruction at INSTSKIP offset (0-11)
```

**INSTID values:**
| ID | Name | Meaning |
|----|------|---------|
| 0 | NO_DEP | No stall needed |
| 1-4 | VALU_DEP_1..4 | Depends on VALU instruction N opcodes back |
| 5-7 | TRANS32_DEP_1..3 | Depends on transcendental N opcodes back |
| 8 | FMA_ACCUM_CYCLE_1 | 1 extra cycle for FMA accumulation |
| 9 | SALU_CYCLE_1 | 1 cycle penalty after SALU |
| 10 | SALU_CYCLE_2 | 2 cycle penalty after SALU |
| 11 | SALU_CYCLE_3 | 3 cycle penalty after SALU |

The `VALU_DEP_N` values tell the GPU the previous VALU result is N opcodes back — the hardware uses its internal latency tables to compute the actual stall. For the emulator, we need to translate these dependency IDs into actual cycle counts based on RDNA3 pipeline latencies.

### The Core Implementation Gap

**File:** `test/mockgpu/amd/emu.py`, function `_init_sqtt_encoder()`

The encoder needs to be extended to:
1. Track a `current_cycle` counter
2. Parse `S_DELAY_ALU` `simm16` values when encountered
3. Apply the stall cycles to the **next instruction's** delta (INSTID0)
4. Apply INSTID1 stall cycles to the instruction at INSTSKIP offset
5. Use the `_N` suffix cycle counts from `InstOp` names as base deltas

```python
# CURRENT (wrong):
_emit_nibbles(nibbles, VALUINST, delta=1, wave=w)

# NEEDED (cycle accurate):
cycles = _get_base_cycles(inst_type, op_name)  # from InstOp suffix or VALU=4
cycles += pending_stall                         # from previous S_DELAY_ALU
pending_stall = 0                               # consumed
current_cycle += cycles
_emit_nibbles(nibbles, VALUINST, delta=cycles, wave=w)
```

When `S_DELAY_ALU` is encountered (before emitting the next packet):
```python
simm16 = inst.simm16
instid0 = simm16 & 0xF            # bits 3:0
instskip = (simm16 >> 4) & 0x7    # bits 6:4
instid1 = (simm16 >> 7) & 0xF     # bits 10:7
stall_next = _instid_to_cycles(instid0)
stall_skip = _instid_to_cycles(instid1)  # applies to instruction at offset instskip+1
```

---

## Implementation Plan

### Phase 1: Add Cycle Tracking State to sqtt_encoder

In `_init_sqtt_encoder()`:
```python
current_cycle = [0]   # mutable for closure
pending_delay = [0]   # stall cycles from S_DELAY_ALU
skip_delays: dict[int, int] = {}  # instruction_offset → stall_cycles
```

### Phase 2: Parse S_DELAY_ALU Before Emitting

Move `S_DELAY_ALU` out of `_SOPP_SKIP` into a new handler that saves stall info instead of emitting a packet:

```python
elif inst_op == SOPPOp3.S_DELAY_ALU.value:
    simm16 = inst.simm16
    instid0 = simm16 & 0xF
    instskip = (simm16 >> 4) & 0x7
    instid1 = (simm16 >> 7) & 0xF
    pending_delay[0] = _instid_to_cycles(instid0)
    if instid1 != 0:
        skip_delays[instskip + 1] = _instid_to_cycles(instid1)
    return  # no packet emitted
```

### Phase 3: Map InstOp → Base Cycles

```python
def _base_cycles(op_name: str) -> int:
    # Extract suffix _N from InstOp name
    import re
    m = re.search(r'_(\d+)$', op_name)
    return int(m.group(1)) if m else 4  # default VALU = 4 cycles
```

### Phase 4: Compute Delta Correctly

```python
def _emit(pkt_cls, wave_id, **kwargs):
    cycles = _base_cycles(...) + pending_delay[0]
    pending_delay[0] = 0
    current_cycle[0] += cycles
    _emit_nibbles(nibbles, pkt_cls, delta=cycles, wave=wave_id, **kwargs)
```

### Phase 5: Validate Against Real Hardware Captures

Run `test_sqttmap.py::TestSQTTMapRDNA3` repeatedly, fix discrepancies:
- Start with `profile_plus` (simple, minimal stalls)
- Move to `profile_gemm` (LDS-heavy)
- Iterate until all tests pass

---

## What We Need (Potential Blockers)

| Resource | Status | Where to Get |
|----------|--------|--------------|
| RDNA3 ISA Manual | Public | [GPUOpen RDNA3 ISA](https://gpuopen.com/rdna3-isa/) |
| Real gfx1100 captures | ✅ In repo | `extra/sqtt/examples/gfx1100/` |
| rocprof-trace-decoder | ✅ Auto-install | `extra/sqtt/install_rocprof_decoder.py` |
| RDNA3 instruction latency table | Needs research | AMD ISA guide, GPUOpen perf guide |
| S_DELAY_ALU encoding spec | Needs research | RDNA3 ISA manual section on SOPP |

---

## Why This Is "AI-Proof" (According to geohot)

The challenge isn't generating code — it's **empirically matching hardware behavior**. The emulator must produce SQTT byte streams that are bitwise identical (in timing) to what a real RDNA3 chip produces. This requires:

1. Deep understanding of the RDNA3 pipeline model
2. Careful study of how S_DELAY_ALU encodes stall information
3. Iterative comparison against real hardware captures
4. Getting every edge case right (branch mispredictions, LDS banking, etc.)

There's no shortcut — you must match the numbers, not just write code that "looks right."

---

## Quick Start

```bash
# Clone and set up
git clone https://github.com/2187Nick/tinygrad
cd tinygrad
pip install -e ".[testing]"

# Install the AMD rocprof decoder (needed for validation)
sudo PYTHONPATH="." ./extra/sqtt/install_rocprof_decoder.py

# See what the current emulator produces (will fail cycle checks)
python -m pytest test/amd/test_sqttmap.py::TestSQTTMapRDNA3 -v

# Run just the packet structure tests (these pass already)
python -m pytest test/amd/test_sqtt_examples.py::TestSQTTExamplesRDNA3 -v

# Run with debug output to see the SQTT packets
DEBUG=2 python -m pytest test/amd/test_sqttmap.py::TestSQTTMapRDNA3 -v -k test_rocprof_inst_traces_match
```

---

## Resources

- **Tinygrad Discord** — `#bounties` channel for questions
- **Bounty spreadsheet** — https://docs.google.com/spreadsheets/d/1WKHbT-7KOgjEawq5h5Ic1qUWzpfAzuD_J06N1JwOCGs/
- **AMD RDNA3 ISA** — https://gpuopen.com/rdna3-isa/
- **AMD GPU Performance Guide** — https://gpuopen.com/learn/rdna-performance-guide/
- **ROCProfiler** — https://github.com/ROCm/rocprofiler
- **Radeon GPU Profiler** — https://github.com/GPUOpen-Tools/radeon_gpu_profiler

---

*This README was generated as part of the bounty research effort. The goal is to win the $1,000 bounty by making the tinygrad RDNA3 software emulator produce SQTT traces that match real gfx1100 hardware cycle-for-cycle on non-DRAM kernels.*
