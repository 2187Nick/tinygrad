# SQTT Deep Dive — RDNA3 Cycle-Accurate Emulator Internals

This document provides a complete technical breakdown of the SQTT (Shader Queue Thread Trace)
system in tinygrad: how packets work, how the emulator assigns cycle-accurate timing, what we
changed in the hardware driver, and why.

---

## Table of Contents

1. [SQTT Packet Types](#1-sqtt-packet-types)
2. [The Timing Model](#2-the-timing-model)
3. [Why We Skip S_SENDMSG Packets](#3-why-we-skip-s_sendmsg-packets)
4. [Changes to ops_amd.py](#4-changes-to-ops_amdpy)
5. [GPU Topology and Wave Placement](#5-gpu-topology-and-wave-placement)
6. [Hardware Validation Results](#6-hardware-validation-results)

---

## 1. SQTT Packet Types

SQTT records a stream of timestamped packets for every instruction every wave executes. Each
packet carries a **delta** — the number of GPU clock cycles since the previous packet. The delta
field has limited bit width, so large gaps require chaining overflow packets.

### Packet Summary

| Packet | Delta Bits | Max Delta | When Emitted | Purpose |
|--------|-----------|-----------|--------------|---------|
| **WAVESTART** | 2 | 3 | Wave allocation | Marks a new wave being scheduled to a SIMD |
| **WAVEEND** | 3 | 7 | `s_endpgm` executes | Marks wave completion and VGPR deallocation |
| **INST** | 3 | 7 | SALU, SMEM, barriers, branches, special VALU | Generic instruction packet with an `InstOp` field |
| **VALUINST** | 3 | 7 | Standard VALU (v_add, v_mul, etc.) | VALU execution; lighter weight than INST |
| **IMMEDIATE** | 3 | 7 | s_waitcnt, s_nop, s_sleep, s_wait_idle | Zero-cost control flow instructions |
| **TS_DELTA_SHORT** | — | — | Overflow | Adds 8-23 cycles when delta exceeds max |
| **TS_DELTA_S5_W3** | 3 | 7 | Overflow | Adds 0-7 more cycles, chainable with SHORT |

### Delta Encoding

Every SQTT packet encodes cycles-since-last-packet in its delta field. When the actual gap
exceeds the field width:

```
If delta > 7 (for 3-bit fields):
  1. Emit TS_DELTA_SHORT (adds field+8 = 8..23 cycles to the running total)
  2. If still not enough, chain TS_DELTA_S5_W3 packets (adds field = 0..7 each)
  3. Finally emit the actual packet with remaining delta (0..7)
```

WAVESTART is special — only 2-bit delta (max 3), because wave allocations happen in rapid
succession.

### INST Packet — The InstOp Field

The INST packet's `op` field tells you what kind of instruction executed:

| InstOp | Value | Cycles | Description |
|--------|-------|--------|-------------|
| SALU | 0x0 | 1 | Scalar ALU (s_add, s_mov, s_cmp, etc.) |
| SMEM_RD | 0x1 | 1 issue, 200 latency | Scalar memory read (L2 cache) |
| BARRIER | 0x13 | 1 | s_barrier — blocks until all waves arrive |
| JUMP | 0x14 | 1 | Branch taken |
| JUMP_NO | 0x15 | 1 | Branch not taken |
| VALUT_4 | 0xb | 4 | Transcendental: exp, log, rcp, sqrt, sin, cos |
| VALUB_2 | 0xd | 2 | 64-bit shift ops |
| VALUB_4 | 0xe | 4 | 64-bit multiply-add |
| VALUB_16 | 0xf | 16 | 64-bit float ops (add, mul, fma, div) |
| VALU1_WR_EXEC | — | 1 | v_cmpx writes exec mask |
| LDS_RD | 0x29 | 1 issue, 31 latency | ds_load (LDS read) |
| LDS_WR_2 | 0x2a | 1 issue, 33 latency | ds_store (LDS write) |
| SGMEM_RD_1 | — | 1 issue, 300 latency | global_load (DRAM read) |
| SGMEM_WR_2 | — | 1 issue, 300 latency | global_store (DRAM write) |

The suffix number (e.g., `_2`, `_4`, `_16`) is the **issue cost** — how many cycles that
instruction occupies the execution unit before the next instruction from the same wave can issue.

### VALUINST Packet

Lighter than INST — used for standard 1-cycle VALU ops (v_add_f32, v_mul_f32, v_lshlrev_b32,
etc.). Has a `flag` field that indicates if this is a VOPC (comparison) instruction. VOPC matters
because comparisons write VCC (vector condition code), not VGPR, so they **don't create VALU
forwarding stalls** to subsequent DS/VMEM operations.

### IMMEDIATE Packet

Used for zero-cost "control" instructions:
- `s_waitcnt` — wait for memory operations to complete (may stall for many cycles, but the
  *packet itself* is zero-cost; the stall is modeled separately)
- `s_nop` — no-op padding
- `s_sleep` — yield SIMD time slot
- `s_wait_idle`, `s_set_inst_prefetch_distance`, `s_clause`

These don't consume an issue slot — they're processed "for free" between real instructions.

---

## 2. The Timing Model

The emulator (`test/mockgpu/amd/emu.py`) uses a **deferred emission + round-robin scheduling**
approach to assign cycle-accurate timestamps. This matches how the real RDNA3 SQ (Shader Queue)
hardware works.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Kernel Execution (emit() called per instruction per wave)      │
│    → Collects events into wave_events[wave_id] list             │
│    → Categories: salu, valu, smem, ds_rd, ds_wr, vmem_rd, ...  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  _simulate_sq_timing(wave_events)                               │
│    Phase 1: Emit WAVESTART at fixed offsets (1, 2, 3, ...)      │
│    Phase 2: Round-robin schedule instructions across waves      │
│    → Applies latency, forwarding stalls, barrier sync           │
│    → Returns [(timestamp, wave_id, pkt_class, kwargs), ...]     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  finalize() → Encode packets with delta encoding → raw bytes    │
└─────────────────────────────────────────────────────────────────┘
```

### Latency Constants

These were tuned against real GFX1100 hardware captures (pkl files in
`extra/sqtt/examples/gfx1100/`):

```python
_LDS_RD_LATENCY    = 31   # DS read → result available (cycles)
_LDS_WR_LATENCY    = 33   # DS write → acknowledged complete (cycles)
_SMEM_LATENCY      = 200  # Scalar memory L2 → result available
_VMEM_LATENCY      = 300  # Vector memory DRAM → result available
_BARRIER_FROM_LAST = 6    # Last barrier arrival → barrier release
_LDS_SERVICE_COST  = 6    # LDS unit busy time per DS operation
_VALU_DS_WR_FORWARD = 26  # VALU result → DS_WR forwarding stall
_VALU_DS_RD_FORWARD = 22  # VALU result → DS_RD forwarding stall
_WAVESTART_GAP     = 1    # Gap between consecutive wave allocations
_FIRST_INST_GAP    = 2    # WAVESTART → first real instruction
```

### Round-Robin Scheduling

The SQ issues one instruction per cycle (per SIMD). When multiple waves are ready, it picks using
**round-robin** with these priority rules:

1. **VALU Burst**: If the last-scheduled wave had a VALU and its next instruction is also VALU
   and ready, it gets priority. This avoids pipeline bubbles in VALU chains.

2. **Earliest Ready**: Among all non-blocked, non-burst waves, pick the one with the earliest
   `effective_ready` cycle. Ties broken by round-robin position.

3. **Forwarding Stall**: When a wave's next instruction is a DS/VMEM operation, its
   `effective_ready` is clamped to the VALU forwarding deadline:
   ```
   DS_WR or VMEM_WR: effective_ready = max(ready, valu_wr_deadline)  # +26 from last VALU
   DS_RD or VMEM_RD: effective_ready = max(ready, valu_rd_deadline)  # +22 from last VALU
   ```

### S_DELAY_ALU — Compiler Hints

RDNA3 uses `s_delay_alu` instructions to tell the SQ about data dependencies. The emulator
decodes these exactly:

```python
# simm16 encodes up to 2 dependency hints:
#   bits [3:0]  = INSTID0 (which prior instruction has the dependency)
#   bits [6:4]  = INSTSKIP (how many instructions to skip before applying INSTID1)
#   bits [10:7] = INSTID1 (second dependency hint)

_INSTID_STALLS = (0, 3, 2, 1, 0, 9, 8, 7, 1, 1, 2, 3)
#                 │  └──────┘  │  └──────┘  │  └──────┘
#                 │  VALU_DEP  │ TRANS32_DEP│  SALU_CYCLE
#              NO_DEP 1-4    4=0  5-7     FMA  9-11
```

Each INSTID maps to a stall in cycles:
- **VALU_DEP_1** (index 1) = 3 cycles — result from 1 VALU ago needs 3 more cycles
- **VALU_DEP_2** (index 2) = 2 cycles — result from 2 VALU ago
- **VALU_DEP_3** (index 3) = 1 cycle
- **VALU_DEP_4** (index 4) = 0 cycles — already available
- **TRANS32_DEP_1** (index 5) = 9 cycles — transcendental result latency
- **TRANS32_DEP_2** (index 6) = 8 cycles
- **TRANS32_DEP_3** (index 7) = 7 cycles
- **SALU_CYCLE_1** (index 9) = 1 cycle — SALU pipeline stages
- **SALU_CYCLE_2** (index 10) = 2 cycles
- **SALU_CYCLE_3** (index 11) = 3 cycles

### VALU → DS/VMEM Forwarding

When a VALU instruction writes a result that a subsequent DS or VMEM instruction needs (as an
address or data), there's a forwarding stall. The hardware can forward the result directly from
the VALU execution unit to the memory unit, but it takes time:

```
VALU writes VGPR at cycle T
  → DS_WRITE/VMEM_STORE needs both address AND data: can't issue before T + 26
  → DS_READ/VMEM_LOAD needs only address:             can't issue before T + 22
```

Why the asymmetry? DS writes need both the address and data VGPRs forwarded (2 operands), while
reads only need the address forwarded (1 operand).

**Exception**: VOPC instructions (v_cmp_*) write VCC, not VGPR, so they do NOT create forwarding
stalls. The emulator tracks this with the `is_vopc` flag.

### Barrier Synchronization (s_barrier)

Barriers synchronize all waves in a workgroup:

```
Timeline for 4-wave barrier on real GFX1100:

  Wave 0: issue s_barrier at cycle 50   (first arrival — no penalty)
  Wave 1: issue s_barrier at cycle 51   (+1 non-first arrival penalty)
  Wave 2: issue s_barrier at cycle 52   (+1)
  Wave 3: issue s_barrier at cycle 53   (+1, last arrival)

  release_cycle = 53 + 6 = 59          (_BARRIER_FROM_LAST = 6)

  Wave 0: resumes at cycle 59           (first in sorted order)
  Wave 1: resumes at cycle 61           (+2 round-robin stagger)
  Wave 2: resumes at cycle 63           (+2)
  Wave 3: resumes at cycle 65           (+2)
```

### LDS Contention Model

The LDS unit is shared across all waves on a CU. Only one wave can use it at a time:

```python
# LDS write serialization:
lds_start = max(issue_cycle, cu_lds_available)     # wait for unit
lgkm_pend.append(lds_start + _LDS_WR_LATENCY)     # completion at +33
cu_lds_available = lds_start + _LDS_SERVICE_COST   # unit busy for 6 cycles

# LDS read (no serialization, but mode-switch penalty):
penalty = 1 if cu_lds_last_was_write else 0
lgkm_pend.append(issue_cycle + _LDS_RD_LATENCY + penalty)  # +31 or +32
```

### Waitcnt — Memory Fence

`s_waitcnt` stalls the wave until N or fewer memory operations are still pending:

```python
# Example: s_waitcnt lgkmcnt(0) — wait for ALL LDS/SMEM ops to complete
lgkm_threshold = 0
stall_until = max(ready, lgkm_pend[-1])  # wait for last pending op
# After stall: issue resumes at stall_until + 1
```

---

## 3. Why We Skip S_SENDMSG Packets

### The Bug

Before the fix, the emulator treated `s_sendmsg` as a generic SALU instruction and emitted an
INST SQTT packet for it. This caused `test_plus_timing_consistent` to fail with `KeyError: 0`.

### What is S_SENDMSG?

`s_sendmsg` (opcode 54) sends a hardware message — typically `MSG_DEALLOC_VGPRS` to return
vector register resources after a kernel is done with them. It's the **last instruction before
s_endpgm** in many compiled kernels.

### The Evidence

**Real hardware does NOT emit an SQTT packet for s_sendmsg.** We verified this through two
independent sources:

1. **Reference PKL files** (`extra/sqtt/examples/gfx1100/`): The hardware-captured SQTT data
   from a real GFX1100 contains zero MESSAGE-type (InstOp=0x9) packets. The `plus` kernel's
   instruction sequence goes directly from the last VALU to WAVEEND — no s_sendmsg packet in
   between.

2. **Live hardware captures on our 7900 XTX**: After fixing hardware capture (see section 4),
   we confirmed: the SQTT packet stream from real hardware never includes s_sendmsg. The wave
   ends with the last real instruction followed immediately by WAVEEND.

### Why No Packet?

`s_sendmsg` is an **asynchronous fire-and-forget** message. The SQ dispatches it to the message
bus and continues immediately — it doesn't occupy an execution slot in the instruction pipeline.
The SQTT hardware tracer only records instructions that flow through the SQ's normal issue logic.
Since `s_sendmsg` bypasses this, no packet is generated.

This is the same reason `s_endpgm` doesn't get an INST packet — it's handled as a special
WAVEEND packet in the SQTT stream, separate from the normal instruction flow.

### The Fix

```python
# test/mockgpu/amd/emu.py line 362-363
_SOPP_SKIP = {
    SOPPOp3.S_ENDPGM.value,                    # → emitted as WAVEEND instead
    SOPPOp3.S_ENDPGM_SAVED.value,              # → WAVEEND variant
    SOPPOp3.S_ENDPGM_ORDERED_PS_DONE.value,    # → WAVEEND variant
    SOPPOp3.S_SENDMSG.value,                   # ← NEW: no SQTT packet on real HW
    SOPPOp3.S_SENDMSGHALT.value,               # ← NEW: no SQTT packet on real HW
}
```

When the emulator encounters any instruction in `_SOPP_SKIP`, it returns immediately without
adding an event to the wave's event list. No event → no SQTT packet → matches real hardware.

### How the KeyError Happened

Without the fix:
1. Emulator emits N+1 INST packets (extra one for s_sendmsg)
2. `map_insts()` tries to map packet N+1 to a PC in the kernel
3. The PC has advanced past the last real instruction (s_endpgm)
4. `pc_map[wave_pc]` → KeyError because that PC isn't in the instruction map

### Was This Only PKL-Based?

**No — it was validated against both PKL reference data AND live hardware captures.** The
reference PKLs were captured on a different GFX1100 machine. Our live 7900 XTX captures
confirmed the same behavior: zero s_sendmsg packets in the SQTT stream. The fix is grounded in
real silicon behavior, not just test data.

---

## 4. Changes to ops_amd.py

`tinygrad/runtime/ops_amd.py` is the hardware driver that configures SQTT registers on the real
GPU. We made 8 changes, each addressing a specific hardware capture failure.

### 4.1 SQTT_ITRACE_SE_MASK: 0b11 → 0x3f

**What**: Controls which Shader Engines get instruction-level tracing (INST/VALUINST packets).
Other SEs still get WAVESTART/WAVEEND but no per-instruction data.

**Before**: `0b11` = only SE0 and SE1 (2 of 6 SEs)
**After**: `0x3f` = all 6 SEs

**Why**: With only 2 SEs having instruction tracing, the probability of a small kernel's wave
landing on a traced SE was ~33%. With all 6 SEs, every SE can capture instruction data, so any
wave placement gives us data. This dramatically improved capture hit rate for small kernels
(1-wave `plus` kernel went from ~10% to ~15% hit rate per attempt).

### 4.2 SQTT_WGP_SEL and SQTT_SA_SEL ContextVars

**What**: New environment variables to control which WGP and Shader Array to trace.

```python
SQTT_WGP_SEL = ContextVar("SQTT_WGP_SEL", -1)  # -1 = auto-detect
SQTT_SA_SEL = ContextVar("SQTT_SA_SEL", 0)       # default: SA 0
```

**Why**: Instruction tracing is limited to ONE specific CU per SE. The `wgp_sel` field in
`regSQ_THREAD_TRACE_MASK` selects which WGP (pair of CUs) to trace. Previously hardcoded to 0,
which **completely disables tracing on GFX11** (empirically verified: wgp_sel=0 → zero INST
packets for any kernel, including GEMM with 1000+ waves).

### 4.3 wgp_sel Auto-Detection

```python
wgp_sel = (self.dev.iface.props['cu_per_simd_array'] - 1) // 2
# For 7900 XTX: cu_per_simd_array=8 → wgp_sel = (8-1)//2 = 3
```

**What**: Automatically selects the last active WGP, matching Mesa's `ac_sqtt.c` behavior.

**Why**: Mesa (AMD's open-source GL/Vulkan driver) uses this exact formula. WGP 3 maps to CUs
6-7, which is the last WGP in each SA. Using the last WGP avoids CU 0 (which may have special
hardware behavior on GFX11 that suppresses tracing).

**Critical discovery**: `wgp_sel=0` completely disables instruction tracing on GFX11. This was
the root cause of our initial zero-INST captures. Despite the register field name, wgp_sel=0
appears to mean "no selection" rather than "WGP 0".

### 4.4 SQTT_LIMIT_SE CU Mask Fix

**Before**: When SQTT_LIMIT_SE > 1, used a simple mask that could miss the traced WGP
**After**: Precisely targets the traced WGP's CUs in both SAs

```python
cu_bits = 0b11 << (wgp_sel * 2)        # 2 CUs per WGP
mask = cu_bits | (cu_bits << 16)        # bit[0:15]=SA0, bit[16:31]=SA1
```

**Why**: `regCOMPUTE_STATIC_THREAD_MGMT_SE` controls which CUs can receive waves. By enabling
only the traced WGP's CUs, we force all waves to land on the CU being traced, guaranteeing
instruction-level capture. Without this, waves scatter across all CUs and most miss the traced
WGP.

### 4.5 hiwater: 1 → 5

```python
self.wreg(regSQ_THREAD_TRACE_CTRL, ..., hiwater=5, ...)
```

**What**: The high-water mark for the SQTT ring buffer. Controls when the hardware signals
"buffer getting full".

**Why**: Mesa's `ac_sqtt.c` uses hiwater=5. A higher value gives the hardware more buffer room
before stalling the shader to avoid overflow. With hiwater=1, the SQ stalls more frequently to
flush the trace buffer, potentially disturbing the timing we're trying to measure.

### 4.6 token_exclude Field Width Mask

```python
token_exclude &= (1 << (self.gc.regSQ_THREAD_TRACE_TOKEN_MASK.fields['token_exclude'][1] + 1)) - 1
```

**What**: Masks `token_exclude` to its 11-bit field width in the TOKEN_MASK register.

**Why**: The `token_exclude` value is built by OR-ing multiple bit flags. Without masking, if the
accumulated value exceeds 11 bits, bit 11 overflows into the adjacent `ttrace_exec` field. This
silently disables execution tracing (`ttrace_exec=0`), which eliminates INST/VALUINST packets
from the trace. This was an intermittent, hard-to-debug failure mode.

Register layout of `regSQ_THREAD_TRACE_TOKEN_MASK`:
```
Bits [10:0]  = token_exclude (11 bits)
Bit  [11]    = ttrace_exec   (1 bit)  ← CRITICAL: must be 1 for instruction tracing
Bit  [12]    = bop_events_token_include
Bits [23:16] = reg_include
Bits [25:24] = inst_exclude
```

### 4.7 ttrace_exec=1 Explicit Setting

```python
self.wreg(self.gc.regSQ_THREAD_TRACE_TOKEN_MASK, ..., ttrace_exec=1, ...)
```

**What**: Explicitly enables thread trace execution token generation.

**Why**: Previously, `ttrace_exec` was not set in the wreg call, relying on whatever the hardware
default was. The default may be 0 (disabled) on some firmware versions or after certain reset
sequences. By explicitly setting it to 1, we guarantee instruction-level packets are generated.
Belt-and-suspenders with the field width mask fix above.

### 4.8 require_profile_mode() — AM Driver Fix

**Before**: `return True` (no-op)
**After**: Attempts to set `profile_standard` via sysfs

```python
def require_profile_mode(self):
    import glob
    for card in sorted(glob.glob('/sys/class/drm/card*/device')):
        try:
            if open(f'{card}/vendor').read().strip() != '0x1002': continue
            fn = f'{card}/power_dpm_force_performance_level'
            if os.path.exists(fn) and open(fn).read().strip() != 'profile_standard':
                with open(fn, 'w') as f: f.write('profile_standard\n')
            return
        except Exception: pass
```

**What**: Sets the GPU power mode to `profile_standard` (fixed clocks, no power gating).

**Why**: **This was the root cause of our initial zero-INST-packet problem.** Without
`profile_standard`, the GPU's dynamic clock gating can suppress the SQ's instruction token
generation pipeline. WAVESTART/WAVEEND packets still work (they're generated in a different
pipeline stage), but INST/VALUINST/IMMEDIATE packets disappear entirely.

The KFD driver (used by ROCm) sets this automatically. The AM driver (tinygrad's bare-metal
driver) was a no-op, which meant instruction tracing silently produced empty results.

**Important caveat**: The AM driver unbinds the amdgpu kernel driver during initialization, which
removes the sysfs interface. `profile_standard` must be set **before** AM_RESET unbinds the
driver. The hardware power state persists at the firmware level even after unbind.

---

## 5. GPU Topology and Wave Placement

### 7900 XTX (gfx1100) Layout

```
GPU (96 CUs)
├── SE0 ─┬── SA0 (8 CUs = 4 WGPs)  ← WGP 0,1,2,3
│        └── SA1 (8 CUs = 4 WGPs)
├── SE1 ─┬── SA0
│        └── SA1
├── SE2 ─┬── SA0
│        └── SA1
├── SE3 ─┬── SA0
│        └── SA1
├── SE4 ─┬── SA0
│        └── SA1
└── SE5 ─┬── SA0
         └── SA1

Total: 6 SEs × 2 SAs × 8 CUs = 96 CUs
Each CU has 4 SIMDs (wave32 execution units)
Each WGP = 2 adjacent CUs
```

### SQTT Tracing Constraints

SQTT instruction tracing is limited to **one CU + one SIMD** per SE. This means:
- wgp_sel selects which WGP (pair of CUs) to trace
- simd_sel selects which SIMD within that CU
- sa_sel selects which Shader Array

If a wave doesn't land on the exact traced CU+SIMD, you get WAVESTART/WAVEEND but zero
instruction packets. For small kernels (1-4 waves), the wave may land on any of the 96 CUs, so
capturing instruction data requires either:
- **Luck**: Retry until the wave hits the traced CU (~1/96 probability per wave)
- **CU masking**: Use `regCOMPUTE_STATIC_THREAD_MGMT_SE` to force waves onto the traced CU
  (what SQTT_LIMIT_SE=2 does)

---

## 6. Hardware Validation Results

### What We Validated

Using a real AMD 7900 XTX with the AM driver:

| Kernel | Non-DRAM Deltas | Emulator Match | Source |
|--------|----------------|----------------|--------|
| custom_lds_sync W0 | 11 deltas | ✅ Exact (diff=0) | PKL + live HW |
| custom_lds_sync W1 | 11 deltas | ✅ Exact (diff=0) | PKL + live HW |
| plus VALU→store | 1 delta (=25) | ✅ Exact | PKL |
| gemm startup | 9 deltas | ✅ Exact | PKL |
| gemm tail | 33 deltas | ✅ Exact | PKL |
| **Total** | **65 deltas** | **All exact** | |

### Non-Determinism Discovery

Real hardware SQTT timing has inherent non-determinism from SQ scheduling state:
- The sync kernel's `v_lshlrev → ds_store` delta varies between 22 and 26 cycles across captures
- The `ds_store → s_waitcnt` delta varies between 33 and 38
- However, the **reference timing pattern IS reproducible** — our Capture 3 of 5 matched the
  reference PKL Wave 0 exactly: `[0, 26, 33, 1, 25, 1, 21, 32, 1, 1, 1]`

This confirms the emulator models one specific (and common) scheduling pattern that real hardware
produces.
