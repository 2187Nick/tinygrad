# RDNA3 ISA Timing Reference for Cycle-Accurate Emulator

> Sources:
> - **XML**: `amdgpu_isa_rdna3.xml` — AMD ISA Spec Manager machine-readable RDNA3 specification
> - **PDF**: "RDNA3" Instruction Set Architecture (Feb 2023, 609 pages)
> - **EMP**: Empirical measurements from our emulator tuned against GFX1100 SQTT hardware traces
> - **ISA_SM**: [GPUOpen-Tools/isa_spec_manager](https://github.com/GPUOpen-Tools/isa_spec_manager) — C++ API for decoding

---

## 1. Architecture Overview (PDF §1.2, pp.15–16)

| Component | Detail |
|---|---|
| WGP (Work-group Processor) | Basic compute unit; contains 2 CUs |
| CU (Compute Unit) | Half a WGP; 2 SIMD32s sharing one memory path |
| SIMD32 | Vector ALU processing 32 lanes per cycle |
| Wave32 / Wave64 | 32 or 64 work-items; wave64 takes 2 passes through SIMD32 |
| LDS per WGP | 128 KB, 64 banks × 512×32 (2-port: 1R/1W per clock) |
| GDS | 4 KB global, 128 bytes/cycle access bandwidth |
| SGPR file | 32-bit registers shared by all work-items in a wave |
| VGPR file | 32-bit registers private per work-item (4 VGPR banks, 3 read ports each) |

### SQ Round-Robin Scheduling
The Sequencer (SQ) issues one instruction per cycle per SIMD from the highest-priority ready wave, using round-robin arbitration among waves on the same SIMD. Multi-wave scheduling hides ALU latency — with N waves, (N-1) cycles of latency are hidden.

---

## 2. Instruction Encodings (XML + PDF §15)

The XML defines **1,148 instructions** across **25 base encodings**:

| Encoding | Bits | FG | Count | Description |
|---|---|---|---|---|
| ENC_SOP1 | 32 | SALU | 65 | Scalar ALU, 1 src |
| ENC_SOP2 | 32 | SALU | 52 | Scalar ALU, 2 src |
| ENC_SOPC | 32 | SALU | 18 | Scalar compare |
| ENC_SOPK | 32 | SALU/WAVE_CONTROL | 24 | Scalar w/ 16-bit immediate |
| ENC_SOPP | 32 | BRANCH/SALU/MSG/TRAP/WAVE | 39 | Scalar control flow |
| ENC_SMEM | 64 | SMEM | 12 | Scalar memory (through K-cache) |
| ENC_VOP1 | 32 | VALU | 86 | Vector ALU, 1 src |
| ENC_VOP2 | 32 | VALU | 42 | Vector ALU, 2 src |
| ENC_VOP3 | 64 | VALU | 310 | Vector ALU, 3 src + modifiers |
| ENC_VOP3P | 64 | VALU | 34 | Packed math (2×16-bit ops) |
| ENC_VOPC | 32 | VALU | — | Vector compare → VCC/EXEC |
| VOPDXY | 64 | VALU | 17 | Dual-issue VALU (wave32 only) |
| ENC_VINTERP | 64 | VALU | 6 | Interpolation |
| ENC_LDSDIR | 32 | VALU | 2 | LDS direct load/param load |
| ENC_DS | 64 | VMEM | 126 | LDS/GDS indexed + atomics |
| ENC_MUBUF | 64 | VMEM | 73 | Untyped buffer operations |
| ENC_MTBUF | 64 | VMEM | 16 | Typed buffer operations |
| ENC_MIMG | 64 | VMEM | 84 | Texture/image operations |
| ENC_FLAT | 64 | VMEM | 52 | Flat addressing |
| ENC_FLAT_GLOBAL | 64 | VMEM | 55 | Global addressing |
| ENC_FLAT_SCRATCH | 64 | VMEM | 22 | Scratch addressing |
| ENC_EXP | 64 | EXPORT | 1 | Graphics export |

Variant encodings (DPP8, DPP16, INST_LITERAL, VOP3_SDST_ENC) extend base encodings with additional fields.

---

## 3. VALU — Vector ALU (VOP1/VOP2/VOP3/VOP3P/VOPC)

### 3.1 Pipeline Depth
**VALU pipeline = 5 cycles** (EMP, validated against SQTT traces)

- `S_DELAY_ALU` INSTID codes 1–4 correspond to VALU_DEP_1 through VALU_DEP_4
- These express dependencies on VALU instructions 1–4 back in the stream
- With N waves round-robin: effective stall = max(0, 5 - N) cycles
- Single-wave stalls: DEP_1=4cy, DEP_2=3cy, DEP_3=2cy, DEP_4=1cy (EMP)

**Source**: PDF §5.7 (pp.53–54): *"The hardware then determines the number of cycles of delay to add"*
**Source**: XML `OPR_DELAY` operand: `INSTID_VALU_DEP_1` through `INSTID_VALU_DEP_4`

### 3.2 DPP Instructions
**DPP instructions incur an extra cycle of delay** (PDF §7.7, p.78):
> *"DPP instructions incur an extra cycle of delay to execute."*

### 3.3 Dual-Issue VALU (VOPD)
PDF §7.6 (pp.77–78): Encodes two VALU operations in one 64-bit instruction, executed in parallel.

- **Wave32 only** — must not be used by wave64
- 17 supported opcodes (XML VOPDXY encoding): ADD_F32, MUL_F32, MOV_B32, FMAC_F32, etc.
- Bank restrictions: src0X and src0Y must use different VGPR banks; vsrc1X and vsrc1Y must differ
- Destination restriction: one VDST must be even, the other odd
- No DPP support
- VOPD occupancy: consecutive VOPDs need 2 extra cycles (EMP: `_VOPD_PIPE_CYCLES = 2`)

### 3.4 WMMA (Wave Matrix Multiply-Accumulate)
PDF §7.9 (p.82): 16×16×16 matrix multiply operations using VOP3P encoding.

- Multi-cycle execution internally using DOT instructions
- Back-to-back dependent WMMA requires 1 V_NOP between them if D overlaps next A/B
- Types: F32←F16, F32←BF16, F16←F16, BF16←BF16, I32←IU8, I32←IU4

### 3.5 FMA Accumulation
XML `OPR_DELAY`: `INSTID_FMA_ACCUM_CYCLE_1` (code 8) = *"Single cycle penalty for FMA accumulation (reserved)"*

### 3.6 EXEC Write Latency
When `V_CMPX` writes EXEC, branches reading EXEC (e.g., `S_CBRANCH_EXECZ`) must wait:
- `_EXEC_WRITE_LATENCY = 24` cycles (EMP, validated: layernorm kernel)

---

## 4. SALU — Scalar ALU (SOP1/SOP2/SOPC/SOPK/SOPP)

### 4.1 Pipeline Depth
**SALU ≈ 1 cycle** with up to 3 cycles dependency penalty.

From XML `OPR_DELAY`:
- `INSTID_SALU_CYCLE_1` (code 9): 1 cycle penalty for prior SALU
- `INSTID_SALU_CYCLE_2` (code 10): 2 cycle penalty (reserved)
- `INSTID_SALU_CYCLE_3` (code 11): 3 cycle penalty (reserved)

PDF §5.7 Table 19 (p.54): *"Codes 9-11: SALU ops typically complete in a single cycle, so waiting for 1 cycle is roughly equivalent to waiting for 1 SALU op to execute before continuing."*

### 4.2 SALU → VALU SGPR Forwarding
**`_SGPR_LATENCY = 4` cycles** (EMP): VALU reading an SGPR written by a recent SALU stalls 4 cycles. HW enforces this without explicit `S_DELAY_ALU` hints.

### 4.3 S_GETREG SHADER_CYCLES
PDF §3.4.10 (p.36): Reading the cycle counter via SALU has *"a typical latency of around 8 cycles"*.

### 4.4 Control Flow Instructions

| Instruction | Description | Notes |
|---|---|---|
| S_BRANCH | Unconditional: PC = PC + SIMM16*4 + 4 | |
| S_CBRANCH_\<test\> | Conditional on SCC/VCC/EXEC | |
| S_NOP N | Delay next issue by N+1 clocks | 0x0=1 clock, 0xf=16 clocks (PDF p.242) |
| S_SLEEP N | Sleep wave for 64*(N-1)..64*N clocks | Approximate; max ~8000 clocks (PDF p.242) |
| S_BARRIER | Sync waves in work-group | `_BARRIER_FROM_LAST = 6` cycles (EMP) |
| S_WAIT_IDLE | Wait for all wave activity complete | All counters at zero |

---

## 5. LDS — Local Data Share (DS Instructions)

### 5.1 Architecture (PDF §12.1, pp.127–128)
- 128 KB per WGP, split into 64 banks (32 banks per CU pair)
- Each bank: 512×32 two-port RAM (1R/1W per clock cycle)
- DWORDs placed serially across banks
- 32 simultaneous load/store operations (32-bit each)
- Extended instructions (load_2addr/store_2addr): 64-bit each

### 5.2 Latency
| Parameter | Value | Source |
|---|---|---|
| LDS read latency (DS_READ_B32) | **31 cycles** | EMP: `_LDS_RD_LATENCY = 31` |
| LDS write latency (DS_WRITE_B32) | **33 cycles** | EMP: `_LDS_WR_LATENCY = 33` |
| LDS b128 read extra latency | **+5 cycles** | EMP: `_LDS_B128_EXTRA = 5` |
| LDS b128 VGPR stagger | **4 cycles** | EMP: upper 2 VGPRs arrive 4cy after lower 2 |
| LDS b128 read service cost | **19 cycles** | EMP: consecutive b128 reads serialized |
| LDS service cost per DS op | **6 cycles** | EMP: `_LDS_SERVICE_COST = 6` |

### 5.3 Bank Conflicts (PDF §12.5, p.134)
> *"Operations can complete in as little as one cycle (for wave32, or 2 cycles for wave64), or take as many 64 cycles, depending upon the number of bank conflicts."*

- 64 banks total, 32 per CU half
- Same-bank accesses are serialized
- Atomics: performed in LDS hardware, not ALU; latency incurred

### 5.4 VALU→LDS Forwarding Stalls (EMP)
| Path | Stall | Description |
|---|---|---|
| VALU → DS_WRITE | 26 cycles | `_VALU_DS_WR_FORWARD = 26` (reference; HW shows 22) |
| VALU → DS_READ | 22 cycles | `_VALU_DS_RD_FORWARD = 22` |

### 5.5 Dependency Tracking
LDS indexed/atomic instructions use **LGKMcnt** counter. LDS operations stay in-order with other LDS ops from the same wave.

---

## 6. VMEM — Vector Memory (MUBUF/MTBUF/MIMG/FLAT/GLOBAL/SCRATCH)

### 6.1 Latency
| Parameter | Value | Source |
|---|---|---|
| VMEM load latency | **300 cycles** | EMP: `_VMEM_LATENCY = 300` |
| VMEM pipeline drain | **15 cycles** | EMP: `_VMEM_DRAIN_CYCLES = 15` |

### 6.2 VALU→VMEM Forwarding Stalls (EMP)
| Path | Stall | Description |
|---|---|---|
| VALU → VMEM store data | 21 cycles | `_VALU_VMEM_WR_FORWARD = 21` |
| VALU → VMEM address VGPR | 27 cycles | `_VALU_VMEM_ADDR_FORWARD = 27` |
| VALU → VMEM read data | 22 cycles | `_VALU_VMEM_RD_FORWARD = 22` |

### 6.3 Dependency Tracking
- **VMcnt**: Texture SAMPLE, Buffer/Global/Scratch/Flat Loads, atomic-with-return
- **VScnt**: Buffer/Global/Scratch/Flat Stores, atomic-without-return

Memory operations of different types can complete out of order.

### 6.4 FLAT Instructions
FLAT instructions use *both* LGKMcnt and VMcnt/VScnt (PDF §5.6, p.53).

---

## 7. SMEM — Scalar Memory

### 7.1 Latency
| Parameter | Value | Source |
|---|---|---|
| SMEM load latency | **200 cycles** | EMP: `_SMEM_LATENCY = 200` |

PDF §8 (pp.85–87): SMEM loads go through the Constant Cache (K-cache). Loads 1–16 DWORDs into SGPRs.

### 7.2 Dependency Tracking
- Uses **LGKMcnt** counter
- Incremented by 1 for single-DWORD fetch, 2 for multi-DWORD
- Loads can return **out of order** — the only safe pattern is `S_WAITCNT LGKMcnt 0`
- Cache invalidates (`S_DCACHE_INV`, `S_GL1_INV`) also tracked by LGKMcnt

### 7.3 Clauses
SMEM clauses lock the instruction arbiter onto a wave. Groups end at non-SMEM instructions. `S_DCACHE_INV` must be in a group by itself (PDF §8.3, p.87).

---

## 8. Transcendental ALU (TRANS32)

### 8.1 Supported Operations (XML VOP1 encoding)
All encoded as VOP1 (promotable to VOP3):
- `V_RCP_F32`, `V_RCP_F16`, `V_RCP_F64`, `V_RCP_IFLAG_F32`
- `V_RSQ_F32`, `V_RSQ_F16`, `V_RSQ_F64`
- `V_SQRT_F32`, `V_SQRT_F16`, `V_SQRT_F64`
- `V_EXP_F32`, `V_EXP_F16`
- `V_LOG_F32`, `V_LOG_F16`
- `V_SIN_F32`, `V_SIN_F16`
- `V_COS_F32`, `V_COS_F16`

### 8.2 Pipeline Timing

| Parameter | Value | Source |
|---|---|---|
| Trans pipeline occupancy | **4 cycles** | EMP: `_TRANS_PIPE_CYCLES = 4` |
| Trans result latency | **27 cycles** | EMP: `_TRANS_PIPELINE_LATENCY = 27` |
| Trans result latency (sqrt/rsq) | **31 cycles** | EMP: `_TRANS_PIPELINE_LATENCY_SQRT = 31` |

**Key insight**: Trans ALU runs *in parallel* with VALU — `S_DELAY_ALU` TRANS32_DEP codes (5–7) have base stall = 0. The 4-cycle pipeline occupancy prevents back-to-back trans ops; actual result availability is tracked by `s_waitcnt_depctr`.

From XML `OPR_DELAY`:
- `INSTID_TRANS32_DEP_1` (code 5): Dependent on previous TRANS32, 1 back
- `INSTID_TRANS32_DEP_2` (code 6): 2 instructions back
- `INSTID_TRANS32_DEP_3` (code 7): 3 instructions back

### 8.3 SQ Issue Cost
Trans instructions (`VALUT_4`) have **1-cycle SQ issue cost** — the 4-cycle pipeline is enforced by `trans_pipe_avail` tracking, not by occupying the SQ issue slot for 4 cycles.

---

## 9. S_DELAY_ALU — Software Scheduling (PDF §5.7, pp.53–54; §16.5, pp.244–247)

### 9.1 Encoding
```
S_DELAY_ALU  SIMM16[3:0]=InstID0, SIMM16[6:4]=InstSkip, SIMM16[10:7]=InstID1
```

Packs two delay specifications into one instruction. May execute in **zero cycles** (parallel with prior instruction).

### 9.2 InstID Codes (XML + PDF Table 19)

| Code | Name | Meaning | Base Stall (1-wave) |
|---|---|---|---|
| 0 | INSTID_NO_DEP | No dependency | 0 |
| 1 | INSTID_VALU_DEP_1 | VALU 1 back | 4 (EMP) |
| 2 | INSTID_VALU_DEP_2 | VALU 2 back | 3 (EMP) |
| 3 | INSTID_VALU_DEP_3 | VALU 3 back | 2 (EMP) |
| 4 | INSTID_VALU_DEP_4 | VALU 4 back | 1 (EMP) |
| 5 | INSTID_TRANS32_DEP_1 | TRANS32 1 back | 0 (EMP) |
| 6 | INSTID_TRANS32_DEP_2 | TRANS32 2 back | 0 (EMP) |
| 7 | INSTID_TRANS32_DEP_3 | TRANS32 3 back | 0 (EMP) |
| 8 | INSTID_FMA_ACCUM_CYCLE_1 | FMA accumulation | 1 (reserved) |
| 9 | INSTID_SALU_CYCLE_1 | SALU 1 cycle wait | 1 |
| 10 | INSTID_SALU_CYCLE_2 | SALU 2 cycle wait | 2 (reserved) |
| 11 | INSTID_SALU_CYCLE_3 | SALU 3 cycle wait | 3 (reserved) |

### 9.3 InstSkip Codes

| Code | Name | Meaning |
|---|---|---|
| 0 | INSTSKIP_SAME | Both deps apply to next instruction |
| 1 | INSTSKIP_NEXT | Dep0 → next inst, Dep1 → inst after |
| 2–5 | INSTSKIP_SKIP_1..4 | Skip 1–4 instructions between Dep0 and Dep1 |

### 9.4 Multi-Wave Behavior
With N waves round-robin: stall = max(0, base_stall - (N-1)). The compiler may not know EXEC mask status for wave64, so `S_DELAY_ALU` encodes dependency *type* so HW can apply correct delay for 1 vs 2 passes.

### 9.5 Restrictions
- Must not appear inside VALU clauses
- Must not come immediately after `S_CLAUSE`
- If two `S_DELAY_ALU` ops are back-to-back, the second replaces the first
- Optional for correctness; only affects performance

---

## 10. S_WAITCNT — Dependency Counters (PDF §5.6, pp.52–53)

### 10.1 Counter Groups

| Counter | Tracks | Instructions |
|---|---|---|
| VMcnt | Vector memory loads | Texture SAMPLE, Buffer/Global/Scratch/Flat loads, atomic-with-return |
| VScnt | Vector memory stores | Buffer/Global/Scratch/Flat stores, atomic-without-return |
| LGKMcnt | LDS + scalar memory | LDS indexed, SMEM loads, GDS, GWS, FLAT (shared w/ VMcnt/VScnt), messages |
| EXPcnt | Exports + LDS params | LDS_PARAM_LOAD, LDS_DIRECT_LOAD, exports |

### 10.2 S_WAITCNT_DEPCTR (VA_VDST / SA_SDST)
From XML and PDF: Additional dependency counters for fine-grained VALU/SALU result tracking.
- `VA_VDST`: Wait for VA_VDST ≤ N (N=15 disables)
- `SA_SDST`: Wait for SA_SDST ≤ N (N=1 disables)

These are used for trans ALU result availability tracking in our emulator.

---

## 11. Instruction Clauses (PDF §5.2, pp.48–49)

`S_CLAUSE` begins a clause of 2–63 same-type instructions. The arbiter is locked to this wave for the clause duration, preventing other waves from using the execution unit.

Clause types: Image Load/Store/Atomic/Sample, Buffer/Global/Scratch Load/Store/Atomic, Flat, LDS, SMEM, VALU.

Clause-internal instructions (don't break clause): `S_NOP`, `S_WAITCNT`, `S_SLEEP`, `S_DELAY_ALU`.
Cannot be in clause: SALU, Export, Branch, Message, GDS, LDSDIR, VINTERP.

Break spans: SIMM16[11:8] = break every N instructions (0 = no breaks, 1–15).

---

## 12. Double-Precision Rate (PDF §3.4.8, p.34)

Register `HW_ID1.DP_RATE[31:29]`: `1+log2(#DP-ALUs)`:
- 0 = none, 1 = 1/32 rate, 2 = 1/16, 3 = 1/8, 4 = 1/4, 5 = 1/2, 6 = full rate

---

## 13. Emulator Constant Validation Summary

| Constant | Emulator Value | Official ISA Support | Status |
|---|---|---|---|
| `_LDS_RD_LATENCY` | 31 cycles | PDF: "as little as 1 cycle (w32)...64 cycles (max bank conflicts)" | **EMP** — no exact official number |
| `_LDS_WR_LATENCY` | 33 cycles | PDF: same range as reads | **EMP** — no exact official number |
| `_SMEM_LATENCY` | 200 cycles | PDF: out-of-order return, through K-cache | **EMP** — highly variable (cache hit/miss) |
| `_VMEM_LATENCY` | 300 cycles | PDF: out-of-order return, through texture cache | **EMP** — highly variable (cache/DRAM) |
| VALU pipeline | 5 cycles | PDF: VALU_DEP_1–4 implies 5-stage pipeline | **Consistent** with ISA (4 delay codes = 5 stages) |
| Trans pipeline occ. | 4 cycles | XML: TRANS32_DEP_1–3 (3 codes) | **Consistent** — 3 dep codes + current = 4 stages |
| Trans result latency | 27–31 cycles | No official number; parallel with VALU | **EMP** |
| `_SGPR_LATENCY` | 4 cycles | PDF: SALU "typically completes in a single cycle"; delay codes 9-11 | **EMP** — SALU→VALU cross-unit forwarding |
| SALU pipeline | ~1 cycle | PDF §5.7 Table 19: "roughly equivalent to 1 SALU op" | **Official** |
| DPP extra delay | +1 cycle | PDF §7.7: "DPP instructions incur an extra cycle" | **Official** |
| S_GETREG latency | ~8 cycles | PDF §3.4.10: "typical latency of around 8 cycles" | **Official** |

### Key Findings
1. **VALU pipeline depth of 5 is well-supported**: The ISA provides 4 VALU_DEP codes (1-4 instructions back), implying a 5-stage pipeline where the 5th instruction can forward without stalls.
2. **Trans pipeline occupancy of 4 is supported**: 3 TRANS32_DEP codes (1-3 back) + current instruction = 4 stages of occupancy.
3. **LDS/SMEM/VMEM latencies are fundamentally empirical**: The ISA spec describes these as variable (cache-dependent) and provides no fixed cycle counts. Our constants represent typical observed values on GFX1100.
4. **S_DELAY_ALU can execute in zero cycles**: It may overlap with the prior instruction, adding no overhead if no delay is needed.
5. **SALU→VALU SGPR forwarding (4 cycles) is not explicitly in the ISA**: The spec describes SALU as ~1 cycle, but cross-unit forwarding to VALU is a microarchitectural detail.

---

## 14. XML Specification Structure (for tooling reference)

The `amdgpu_isa_rdna3.xml` provides:

### Instruction Fields
- `InstructionName`: Full mnemonic (e.g., `V_ADD_F32`)
- `EncodingName`: Microcode format (e.g., `ENC_VOP2`)
- `FunctionalGroup/Name`: Execution unit (VALU, SALU, SMEM, VMEM, BRANCH, etc.)
- `Opcode`: Numeric opcode within encoding
- `Operands`: Input/output operand definitions with types and sizes
- `Description`: Prose description of semantics

### Functional Groups (from XML)
| Group | Count | Execution Unit |
|---|---|---|
| VALU | 508 | Vector ALU (SIMD32) |
| VMEM | 428 | Memory controller (includes LDS/DS) |
| SALU | 155 | Scalar ALU |
| BRANCH | 19 | Branch unit |
| WAVE_CONTROL | 16 | Wave management |
| SMEM | 12 | Scalar memory (K-cache) |
| MESSAGE | 7 | Message passing |
| TRAP | 2 | Trap handling |
| EXPORT | 1 | Export unit |

### Timing-Relevant XML Fields
- `OPR_DELAY` operand type: Encodes `S_DELAY_ALU` dependency codes (VALU_DEP, TRANS32_DEP, SALU_CYCLE, FMA_ACCUM)
- `VA_VDST` / `SA_SDST` field names: Dependency counter checks in `S_WAITCNT_DEPCTR`
- `IsBranch`, `IsConditionalBranch`, `IsIndirectBranch`, `IsProgramTerminator`: Control flow classification
- `IsImmediatelyExecuted`: Marks instructions that execute without SQ delay

### What the XML Does NOT Contain
The XML is focused on **encoding and semantics**, not timing. It does not include:
- Cycle counts or latency values
- Pipeline stage assignments
- Throughput/issue rate annotations
- Cache hierarchy details
- Functional unit mapping (beyond broad FunctionalGroup)

All cycle-level timing must come from the ISA PDF prose, AMD optimization guides, or empirical measurement.

---

## 15. References

1. AMD. "RDNA3" Instruction Set Architecture. February 2023. 609 pages.
2. AMD. Machine-Readable GPU ISA Specification (`amdgpu_isa_rdna3.xml`). Schema v1.1.1.
3. [GPUOpen-Tools/isa_spec_manager](https://github.com/GPUOpen-Tools/isa_spec_manager) — C++ decoder API.
4. AMD Matrix Instruction Calculator: https://github.com/RadeonOpenCompute/amd_matrix_instruction_calculator
5. tinygrad RDNA3 emulator: `test/mockgpu/amd/emu.py`
