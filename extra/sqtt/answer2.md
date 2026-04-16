# RDNA3 Emulator Timing: Answers to ISA Questions Round 2

## Status

Answers to the four questions in `isa_questions2.md`. These build directly on the
SGPR write-back model from `answer.md` (completion buffer depth ≈2, 2-cycle commit
gap, +1 availability offset).

Notation follows `answer.md` conventions:

| Symbol | Meaning |
|--------|---------|
| `d[n]` | Total cycle cost of instruction n (stall + 1 issue cycle). d=1 → no stall. |
| `T[n]` | Eligible time: `T[n] = T[n-1] + d[n-1]` |
| `W[n]` | Write-back ready time: `I[n] + 5` (VALU pipeline latency) |
| `C[n]` | Commit time: `max(W[n], C[n-1] + 2)` (write-port serialization) |
| `A[n]` | Readable time: `C[n] + 1` |

---

## Q1: v_cndmask_b32_e64 with Non-VCC SGPR — Position-Dependent Delta

### Short Answer

The position-dependent stall is **not** a read-port arbitration or operand collector
bandwidth issue. The leading hypothesis is that it is the **same SGPR write-back
serialization** from `answer.md`: the 2-cycle commit gap delays later SGPR writes,
making them unavailable when the corresponding reads arrive.

The Block A vs Block B variation is consistent with **global completion buffer
state** — prior SGPR-writing VALUs leave residual entries in the buffer, shifting
the entire write-commit timeline.

### Why Read-Port Arbitration Is Unlikely

If the stall were caused by limited SGPR read ports or operand collector bandwidth:
- The stall pattern would depend on **which** registers were being read simultaneously
- Stalls would appear on any SGPR-reading instruction, not specifically on
  instructions reading recently-written SGPRs
- The pattern would not vary between Block A and Block B (same registers, same
  micro-ops)

Instead, the pattern correlates with **write recency**: how recently each SGPR
committed relative to when it is read. This points to the write-back path, not the
read path.

### Mechanism (Same as answer.md)

The `probe_sgpr_cmps` sequence writes 4 SGPRs then reads them back:

```
v_cmp_gt_f32_e64  → writes s[4]    ; [0] each issues d=1 (back-to-back)
v_cmp_lt_f32_e64  → writes s[5]    ; [1]
v_cmp_gt_f32_e64  → writes s[6]    ; [2]
v_cmp_lt_f32_e64  → writes s[7]    ; [3]
v_cndmask_b32_e64 ← reads  s[4]   ; [4] d varies
v_cndmask_b32_e64 ← reads  s[5]   ; [5] d varies
v_cndmask_b32_e64 ← reads  s[6]   ; [6] d varies
v_cndmask_b32_e64 ← reads  s[7]   ; [7] d varies
```

**Write side** — the 4 writes issue back-to-back (assuming no buffer stall),
but write-port serialization delays their commits:

| Inst | Register | I (entry) | W (wb ready) | C (commit) | A (readable) |
|------|----------|-----------|--------------|------------|--------------|
| [0]  | s[4]     | 0         | 5            | 5          | 6            |
| [1]  | s[5]     | 1         | 6            | max(6, 5+2) = **7** | 8  |
| [2]  | s[6]     | 2         | 7            | max(7, 7+2) = **9** | 10 |
| [3]  | s[7]     | 3         | 8            | max(8, 9+2) = **11** | 12 |

Note: each successive commit is pushed later by the 2-cycle gap constraint.
s[7] commits 6 cycles after s[4] despite being written only 3 cycles later.

**Read side** — the first read is eligible at T=4:

| Inst | Reads | T (eligible) | A (src) | Stall | d |
|------|-------|--------------|---------|-------|---|
| [4]  | s[4]  | 4            | 6       | 2     | 3 |
| [5]  | s[5]  | 7            | 8       | 1     | 2 |
| [6]  | s[6]  | 9            | 10      | 1     | 2 |
| [7]  | s[7]  | 11           | 12      | 1     | 2 |

**Model prediction**: `[3, 2, 2, 2]`

**Team observation**: `s[4] = 1 or 3, s[5] ≈ 2, s[6] = 1 or 2, s[7] ≈ 1`

### What the Model Gets Right

1. **s[5] ≈ 2**: The model correctly predicts a consistent 2-cycle cost for s[5],
   matching the team's observation.
2. **Stall decreases toward the end**: The model shows stalls concentrated on
   earlier reads, which the team confirms.
3. **Block variation**: Different initial buffer state produces different
   d-value distributions, explaining A vs B differences.

### What the Model Doesn't Fully Explain

1. **s[4] = 1 in some blocks**: The model predicts d=3 for s[4] from a clean start.
   When s[4] shows d=1, it suggests the buffer had enough prior drain time that
   s[4]'s commit was not delayed — or that the buffer depth is slightly larger
   than 2 for this scenario, allowing all writes to issue without buffer-full stalls
   (which would shift the commit timeline).

2. **s[7] ≈ 1**: The model predicts d=2. Getting d=1 for s[7] requires either:
   - A smaller availability offset (commit + 0 instead of commit + 1)
   - Or prior read stalls accumulating enough time for s[7]'s commit to complete

3. **Block A vs Block B**: The exact d-value set for each block depends on global
   state (prior SGPR writes in the buffer and the last_commit_time). Without
   the full instruction trace before each block, we cannot predict the exact values.

### Emulator Implementation

**Use the same `SGPRWriteTracker` from answer.md.** The `v_cndmask_b32_e64` reading
a non-VCC SGPR condition calls `read_sgpr(src_sgpr, current_cycle)` just like any
other SGPR-reading instruction. No special handling is needed for the condition mask
operand (SRC2) — it reads from the same SGPR file with the same availability
constraint.

```python
# In issue_instruction(), when processing v_cndmask_b32_e64:
if instr.opcode == 'v_cndmask_b32_e64' and instr.src2_is_sgpr:
    # Same read_sgpr() call as any other SGPR source
    stall = sgpr_tracker.read_sgpr(instr.src2, current_cycle)
    # No special "condition mask" latency — it's just an SGPR read
```

**Key point**: `v_cndmask` with a non-VCC SGPR is NOT faster or slower than any
other VALU instruction reading that same SGPR. The position-dependent delta is
entirely from the write-back timing of the source SGPR, not from any property of
the `v_cndmask` instruction itself.

### Confidence: Medium

The model captures the qualitative pattern but does not perfectly reproduce all
observed d-values. The remaining discrepancy is likely from:
- Unmodeled global buffer state (prior SGPR-writing VALUs before each block)
- Possible buffer depth > 2 in this context (the depth may be 3 or dynamic)
- The availability offset (commit + 1) may vary by context

---

## Q2: s_nop(N) SQTT Token Timing

### Short Answer

Your empirical formula is correct:

```
SQTT_delta = N + 2  (post-issue cycles until the SQTT token fires)
```

Where N is SIMM16[3:0] from the s_nop encoding.

### ISA Confirmation

From the RDNA3 ISA PDF:

> **S_NOP**: Insert 0..15 wait states based on SIMM16[3:0].
> 0x0 means next instruction can issue on next clock,
> 0xf means 16 clocks later.

Pseudocode:
```
for i in 0 : SIMM16[3:0] do
    nop()
endfor
```

So `s_nop(N)` consumes **N+1 cycles**: 1 issue cycle + N wait states.

LLVM confirms this in `SIInstrInfo.cpp`:
```cpp
// s_nop N → (N+1) cycles
return MI.getOperand(0).getImm() + 1;
```

### Derivation of N+2

Your measurement: `s_nop(10)` shows SQTT delta = 13 from the previous
instruction's timestamp.

Working backward:
- `s_nop(10)` execution time = 10 + 1 = 11 cycles
- If the previous instruction issued at time T₀ and took 1 cycle, the s_nop
  issues at T₀ + 1
- The s_nop completes at T₀ + 1 + 11 = T₀ + 12
- SQTT delta = 13, so the token fires at T₀ + 13

This means the SQTT token fires **1 cycle after the s_nop's last active cycle**:

```
token_time = nop_issue + (N+1) + 1 = nop_issue + N + 2
delta_from_prev = (prev→nop gap) + N + 2
```

For back-to-back issue (gap=1): delta = 1 + N + 2 = N + 3... but you measured 13
for N=10, which is N + 3 = 13. ✓

Wait — let me re-check: if your formula is `stamp = nop_issue + nop_cycles + 1`:
- nop_cycles = N + 1 = 11
- stamp = nop_issue + 11 + 1 = nop_issue + 12
- delta from prev = 12 + (nop_issue - prev_stamp) = 12 + 1 = 13 ✓

### What Does the +1 Represent?

The measurements are consistent with a SQTT timestamp convention equivalent to
`issue_time + execution_cycles + 1`. However, from a single data point we cannot
independently separate:

1. Instruction execution time (which we know is N+1)
2. SQTT token emission convention (unknown)
3. Any fixed pipeline commit latency

The +1 could represent:
- **Retirement latency**: the token fires when the nop retires, 1 cycle after its
  last active cycle
- **SQTT convention**: the token represents "next slot available" rather than
  "instruction completed"
- **Pipeline commit**: a fixed 1-cycle delay for any instruction to commit its
  state after execution

All three explanations are observationally equivalent from timing data alone.

### Emulator Implementation

```python
def sqtt_stamp_for_snop(self, nop_issue_time, simm16_3_0):
    """SQTT timestamp for s_nop(N)."""
    nop_cycles = simm16_3_0 + 1  # N wait states + 1 issue cycle
    return nop_issue_time + nop_cycles + 1  # the observed +1
```

**For general instructions**, if the pattern holds:
```python
def sqtt_stamp(self, issue_time, execution_cycles):
    """Generic SQTT timestamp. Needs validation against more instruction types."""
    return issue_time + execution_cycles + 1
```

### Confidence: High (for the formula), Low (for the mechanism)

The `N + 2` post-issue formula is well-supported by the measurement and the ISA's
definition of s_nop cycle count. The physical meaning of the +1 is uncertain.

---

## Q3: Scalar Branch NOT-TAKEN Cost — 8-9 Cycles

### Short Answer

The 8-9 cycle SQTT cost for a NOT-TAKEN `s_cbranch_scc1` is a **front-end /
control-flow pipeline effect**, much larger than any SCC read-after-write hazard.
The SCC RAW latency (SALU pipeline = 1-2 cycles) is a small component at most.

### What the ISA Says

**S_CBRANCH_SCC1** pseudocode (from PDF):
```
if SCC == 1'1U then
    PC = PC + signext(SIMM16.i16 * 4) + 4
else
    PC = PC + 4
endif
```

The ISA documentation provides **no timing information** for branches. The branch
instruction table (PDF section on program flow) lists opcodes and conditions but
not cycle counts.

### Evidence That SCC Hazard Is Not the Primary Cause

**S_DELAY_ALU SALU_CYCLE codes** (PDF):
- `INSTID_SALU_CYCLE_1` (0x9): 1 cycle penalty for prior SALU instruction
- `INSTID_SALU_CYCLE_2` (0xa): reserved
- `INSTID_SALU_CYCLE_3` (0xb): reserved

Only `SALU_CYCLE_1` is non-reserved, indicating SALU data forwarding latency is
just 1 cycle for most cases. LLVM's scheduling model uses `WriteSALU = 2 cycles`.

If SCC read latency were the primary cause, we'd expect:
- A consistent 2-3 cycle cost (SALU pipeline depth)
- No variation between instances
- A fix via inserting 1 s_nop between s_cmp and s_cbranch

Instead, the cost is 8-9 cycles with 1-cycle variance — far beyond any data hazard.

### What Causes the 8-9 Cycles

The cost is consistent with **instruction fetch/issue pipeline overhead** for
branch resolution:

1. **SALU branch evaluation**: The branch condition (SCC) is read and the
   taken/not-taken decision is made. This takes ~2 cycles (SALU pipeline depth).

2. **Front-end pipeline effects**: Even for a not-taken branch, the hardware must:
   - Confirm the sequential path is correct
   - Potentially flush any speculatively fetched instructions
   - Refill the instruction buffer if it was stalled waiting for branch resolution

3. **SQTT instrumentation overhead**: The SQTT token for a branch may include
   additional accounting cycles.

**Important caveat**: This 2 + pipeline-effects breakdown is a hypothesis, not a
confirmed decomposition. The ISA PDF does not document any of these pipeline stages
individually. The total cost of 8-9 cycles is an empirical observation.

### The 1-Cycle Variance (8 vs 9)

Possible causes:
- **Instruction cache alignment**: Slight variation in fetch latency depending
  on where the branch target falls in a cache line
- **Wave scheduling**: Other waves may interleave during the branch resolution
  window, adding jitter
- **Instruction buffer occupancy**: If the IB was fuller or emptier at branch
  time, refill takes different time

### Does s_cmp → s_cbranch Need a Gap?

**No mandatory NOP is required.** The ISA does not document any minimum gap between
an SCC-writing instruction and an SCC-reading branch. LLVM does not insert any
hazard NOPs between s_cmp and s_cbranch.

However, S_DELAY_ALU with `SALU_CYCLE_1` exists precisely for cases where SALU
data forwarding needs a hint. The branch evaluation appears to handle SCC reads
internally without requiring explicit delay.

### The delta=3 on the Next Instruction

After two back-to-back branches, the next instruction shows SQTT delta=3. This
represents the tail of the front-end pipeline refill — the instruction was fetched
and decoded but needed 2 additional cycles to traverse the issue pipeline after the
branch resolution cleared the way.

### Emulator Implementation

```python
def cost_scalar_branch_not_taken(self):
    """Cycle cost for s_cbranch_* when NOT taken."""
    # Base cost: observed 8-9 cycles from SQTT measurements
    # Use 8 as the base with possible +1 jitter
    return 8  # or 9; the variance is not predictable from instruction state

def cost_scalar_branch_taken(self):
    """Cycle cost for s_cbranch_* when TAKEN."""
    # Taken branches have higher cost due to actual PC redirect
    # LLVM scheduling model: WriteBranch = 32 (likely worst-case/conservative)
    # Real hardware is likely lower; needs measurement
    return 16  # placeholder — measure with SQTT
```

**Note on `WriteBranch = 32` in LLVM**: This is a scheduling model constant, not
a direct hardware cycle count. LLVM scheduling models are conservative estimates
used for instruction scheduling heuristics. Do not use 32 as the actual branch
latency.

### Confidence: Medium

The 8-9 cycle range is well-established empirically. The decomposition into
sub-components is speculative. The claim that SCC hazard is not the primary cause
is supported by the magnitude mismatch (1-2 vs 8-9 cycles).

---

## Bonus: VALU→VMEM Forwarding After s_nop

### Short Answer

**Yes, `s_nop(N)` counts toward any cycle-based forwarding or readiness window.**
During the N+1 cycles of s_nop execution, real time passes — VALU write-backs
complete, dependency counters advance, and SGPR values become available in the
register file.

If the VALU result committed during the nop stall (at `VALU_issue + 5`), then by
the time the store issues after the nop, the data is sitting in the register file
with no forwarding needed.

### ISA Basis

s_nop is defined as:
```
for i in 0 : SIMM16[3:0] do
    nop()
endfor
```

These are real pipeline cycles. The nop does not freeze the pipeline — it holds the
**issue slot** for N+1 cycles while everything else (write-backs, memory returns,
counter decrements) continues normally.

### Analysis of the Team's Scenario

Sequence:
1. VALU instruction issues at time T_valu (produces VGPR or SGPR result)
2. `s_nop(10)` issues at T_valu + k (11 cycles of stall)
3. `global_store_b32` issues at T_valu + k + 12 (approximately)

The VALU result write-back completes at T_valu + 5. By the time the store issues
(~T_valu + k + 12), the result has been in the register file for many cycles.

The store's observed delta = 5 likely represents the VMEM store pipeline's own
processing time:
- Address calculation
- Data formatting / lane masking
- Submission to the memory system

This is **not** a forwarding stall — the data was ready long before the store issued.

### VALU→VMEM Hazard Specifics

LLVM's `GCNHazardRecognizer.cpp` documents these VALU→VMEM hazards for GFX11:

1. **VALU→VMEM address SGPR read**: 5 wait states required between a VALU that
   writes an SGPR and a VMEM instruction that reads that SGPR as an address
   component.

2. **VALU→VMEM general**: 1 wait state (inserted as s_nop if needed).

These hazards are about **minimum wait states between issue**, not about a
forwarding deadline. If s_nop(10) provides 11 wait states, any 1-wait-state
or 5-wait-state hazard is trivially satisfied.

### Important Distinction: VGPR vs SGPR Store Data

The team should verify whether the store is consuming:
- **VGPR data** (the store data operand): VGPR forwarding follows the standard
  VALU 5-cycle pipeline, and the data is in the VGPR file by issue + 5.
- **SGPR address** (base address or offset): Subject to the 5-wait-state
  SGPR hazard above.

These are different hazard classes. The "21-cycle forwarding deadline" the team
mentions may apply to a specific hazard class that I haven't found documented.
If this number comes from your own measurements, it may reflect a real hardware
constraint — please share the measurement methodology so we can validate.

### Emulator Implementation

```python
def check_valu_vmem_hazard(self, vmem_instr, current_cycle):
    """Check if VMEM instruction needs to wait for prior VALU results."""
    stall = 0

    # Check SGPR address operands (5 wait states from VALU SGPR write)
    for sgpr in vmem_instr.address_sgprs:
        if sgpr in self.sgpr_tracker.sgpr_available:
            avail = self.sgpr_tracker.sgpr_available[sgpr]
            stall = max(stall, max(0, avail - current_cycle))

    # VGPR data operands: standard 5-cycle VALU pipeline latency
    for vgpr in vmem_instr.data_vgprs:
        vgpr_avail = self.vgpr_tracker.get_available(vgpr)
        stall = max(stall, max(0, vgpr_avail - current_cycle))

    return stall

def process_snop(self, simm16, current_cycle):
    """s_nop burns real cycles — all counters/timers advance."""
    nop_cycles = (simm16 & 0xF) + 1
    return current_cycle + nop_cycles  # time advances, nothing else changes
```

### Confidence: High (s_nop counts toward windows), Low (21-cycle deadline)

The principle that s_nop advances real time is well-established from the ISA
definition. The team's "VALU_issue + 21cy forwarding deadline" is not found in
any official source and needs independent validation.

---

## Cross-Cutting Findings

### LLVM AMDGPU Scheduling Constants (for reference)

These are **scheduling model hints**, not exact hardware cycle counts. Use them
as starting points for emulator tuning, not as ground truth.

| Operation | LLVM Cycles | Notes |
|-----------|-------------|-------|
| Write32Bit (FP32 VALU) | 5 | Well-confirmed by S_DELAY_ALU |
| Write64Bit (FP64 VALU) | 6 | |
| WriteTrans32 | 10 | Transcendental (sin, cos, etc.) |
| WriteSALU | 2 | SALU pipeline depth |
| WriteSFPU | 4 | Scalar FPU |
| WriteSMEM | 20 | Scalar memory (conservative) |
| WriteVMEM | 320 | Vector memory (very conservative) |
| WriteBranch | 32 | Branch (conservative; real is much lower) |
| WriteExport | 16 | Export (GDS, parameter, etc.) |
| WriteLDS | 20 | LDS/GDS |

### SGPR Dependency Counter Names (from LLVM SIDefines.h)

The hardware tracks in-flight SGPR writes using these counters, accessible via
`S_WAITCNT_DEPCTR`:

| Counter | Tracks |
|---------|--------|
| VA_SDST | VALU → named SGPR (s[N]) write-backs in flight |
| VA_VCC  | VALU → VCC write-backs in flight |
| SA_SDST | SALU → named SGPR write-backs in flight |
| VM_VSRC | VMEM → VGPR write-backs in flight |

### Key ISA PDF References

| Topic | PDF Line(s) | Content |
|-------|-------------|---------|
| S_NOP definition | 27076-27085 | Wait state count, pseudocode |
| S_DELAY_ALU | 27268-27273 | SALU_CYCLE_1/2/3 dependency codes |
| S_CBRANCH_SCC1 | 27507-27517 | Branch pseudocode |
| Branch instructions | 4669-4721 | Branch opcode table |
| Data dependencies | 4741-4870 | S_DELAY_ALU semantics, VALU pipeline |
| VGPR read ports | 7783-7791 | 3 read ports per VGPR bank |
| SGPR = VCC physical | 1077 | VCC stored in SGPR file |
| SGPR-writing VALU | 1255 | Never skipped even when EXEC=0 |

---

## Summary of Answers

| Question | Answer | Confidence | Impact on Mismatch Count |
|----------|--------|------------|--------------------------|
| Q1: v_cndmask position stall | Same write-back serialization as answer.md; NOT read-port arbitration | Medium | Should improve ~4-6 mismatches if combined with answer.md fixes |
| Q2: s_nop SQTT timing | `N + 2` post-issue confirmed; mechanism of +1 is uncertain | High (formula) | Fixes SQTT delta calculation for all s_nop instructions |
| Q3: branch NOT-TAKEN cost | 8-9 cycles from front-end effects; SCC hazard is small component | Medium | Use 8 or 9 as fixed cost; 1-cycle variance is inherent |
| Bonus: VALU→VMEM after s_nop | s_nop counts toward all cycle-based windows | High | Eliminates false forwarding-deadline mismatches after nops |

### Remaining Open Questions

1. **Completion buffer depth**: 2 fits answer.md's Block 1 perfectly but Q1's
   `probe_sgpr_cmps` data suggests it may be 3 or variable. Needs more test cases
   with known prior state.

2. **21-cycle VALU→VMEM forwarding deadline**: Not found in any official source.
   If this is empirically derived, please share the measurement so we can validate.

3. **Branch prediction**: The ISA PDF does not document a branch predictor for
   scalar branches. The 1-cycle variance (8 vs 9) could be from prediction or
   from other front-end effects.

4. **SQTT token semantics**: The +1 beyond instruction execution cycles is
   consistent across s_nop measurements, but we need data from other instruction
   types to confirm it's a universal SQTT convention vs s_nop-specific.
