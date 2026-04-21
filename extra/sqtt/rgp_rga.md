# RGP + RGA Data Proposal

**Addressed to:** tinygrad HW-access team
**From:** emulator bounty team (2187Nick)
**Date:** 2026-04-21
**Companion to:** `wave.md` (wave-slot placement) and `mes_notes.md` (MES review)
**Status:** MODAL 66055/69862 (94.6%), strict 55677/69862 (79.7%) at commit `1bec956bf`.

---

## TL;DR

Two high-leverage changes would extract *much* more signal from runs we're
already doing. Neither needs a new capture rig.

1. **Enable `SQ_TT_TOKEN_PERF`** in the SQTT token mask — a **one-line
   change** in `tinygrad/runtime/ops_amd.py:278`. Unlocks per-instruction
   HW-measured latency in every SQTT blob we already take. Projected close:
   **~1 500 tokens** (the sgpr-mixed, cndmask/depctr, trans, vcmp buckets).
2. **Static VGPR-bank analysis** of the compiled kernels — using RGA or
   our own ISA parser. Tells us which vopd/raw pairs hit the same bank.
   Projected close: **~250 tokens** (raw-banks / vopd bucket).

Together: realistic ceiling **~98–99 % MODAL / ~92–94 % strict** without
any further HW-side work. The remaining gap is memory-subsystem
stochasticity — separate, already tracked in `STOCHASTIC_SCHEDULER_PLAN.md`.

Third item, longer-horizon: proper SPM streaming via `regRLC_SPM_*` if we
want time-evolution data. Not required for the near-term accuracy push.

---

## 1. Where we are now — what RGP captures we actually keep

`tinygrad/runtime/ops_amd.py:229-292` sets up the SQ thread-trace. Three
filters are currently pulling useful data *off the floor*:

| Filter | Set at | Effect | Already addressed? |
|---|---|---|---|
| `simd==0` filter in decoder | `tinygrad/renderer/amd/sqtt.py:657` | One SIMD of one CU visible per capture | Bypassed by `capture_raw_sqtt.py` (wave.md script #2) |
| `regSQ_THREAD_TRACE_MASK.simd_sel/wgp_sel/sa_sel` | `ops_amd.py:275` | Trace points to one SIMD/WGP/SA | Same — raw blob has all units, decoder filters |
| **`SQ_TT_TOKEN_EXCLUDE_PERF`** | `ops_amd.py:278` | **PERF token type is suppressed on gfx11** | **Not yet — this doc** |

The third is the one we've been ignoring. RGP visualizes SQTT without
needing the PERF packets, so tinygrad drops them to save space. For
cycle-accurate emulation they are the richest signal in the trace.

## 2. Proposal (a) — Enable `SQ_TT_TOKEN_PERF`

### 2.1 The packet

SQTT emits a `SQ_TT_TOKEN_PERF_COUNTER` packet alongside other tokens. The
packet format is already defined in our decoder:

```python
# tinygrad/renderer/amd/sqtt.py:330
class PERF(PacketType):  # exclude: 1 << 11
  encoding = bits[4:0] == 0b10110
  delta    = bits[7:5]
  arg      = bits[27:8]       # 20-bit payload on gfx11

class PERF_RDNA4(PacketType):
  encoding = bits[4:0] == 0b10110
  delta    = bits[9:7]
  arg      = bits[31:10]      # 22-bit on gfx12
```

The PERF packet's `arg` field carries a cycle-count payload associated
with the preceding instruction — in practice, the wave's clock-cycle
delta since the last PERF. Cross-reference: Mesa's
`src/amd/common/ac_sqtt.c` + `ac_sqtt.h`
(`AC_SQTT_TOKEN_TYPE_PERF = 0xb`), and RGP's internal
`SqttFileChunkSpmDb` documentation.

**What we get:** one measured `arg` value per *issue slot*, attributable
to the VALU/SALU/VMEM instruction that issued in that slot. Our current
emulator *predicts* the number of cycles each instruction should take;
the PERF token *measures* it.

### 2.2 The one-line change

`tinygrad/runtime/ops_amd.py:278`:

```python
# Before:
token_exclude = SQTT_TOKEN_EXCLUDE.value | ((1 << self.soc.SQ_TT_TOKEN_EXCLUDE_PERF_SHIFT) if self.dev.target < (12,0,0) else 0)

# After:
token_exclude = SQTT_TOKEN_EXCLUDE.value
```

That's it on the capture side. Gated by `SQTT_TOKEN_EXCLUDE` env var,
so we can even keep the default conservative and override:

```bash
SQTT_TOKEN_EXCLUDE=0 ... python extra/sqtt/wave_probe/capture_raw_sqtt.py
```

**Trace-buffer impact:** each PERF packet is 4 bytes. With ~1 PERF per
instruction, a kernel emitting 200 instructions × 16 waves × 30 kernels
≈ 400 KB extra across the whole microbench run. Negligible — the 256 MB
per-SE SQTT buffer has headroom.

### 2.3 Decoder work

The packet class already exists. What's missing: interpretation of `arg`.

Likely semantics (Mesa/ac_sqtt has the authoritative answer):
- `arg` = wave-clock delta since the previous PERF emit, in 1-cycle units.
- Attribution: the PERF packet appears *after* the instruction whose
  latency it reports; pair by timestamp.

Scoring plan:
1. Run `capture_raw_sqtt.py` with PERF-token enabled on real HW.
2. In `decode_all_simds.py`, collect PERF deltas alongside VALUINST / ALUEXEC
   / VMEMEXEC stamps.
3. Build a `{(kernel, inst_idx): measured_cycles}` map.
4. Compare against `test/mockgpu/amd/sq_timing/*` predictions per instruction.
5. Anywhere delta > 3 cycles, that's our emulator miss — we now know which
   instruction and by how much.

Decoding the `arg` semantics is ~1 afternoon of hacking against Mesa
source + a reference trace. Main-line cost is re-running the captures
once we're ready to decode.

### 2.4 Bucket coverage (from wave.md §2)

| Bucket | Misses | PERF-token-covered? |
|---|---:|:---:|
| cndmask / depctr | 386 | ✓ — scalar-pipe contention measured at instruction |
| sgpr-mixed (s_bfe → VMEM) | 311 | ✓ — SALU→VMEM delay measured at the VMEM issue |
| trans (exp/log/rcp) | 303 | ✓ — trans pipe stagger visible in PERF delta |
| vcmp (cndmask) | 302 | ✓ — vcc propagation = measurable gap |
| trans-chain (n≥4) | 181 | ✓ — per-instruction chain latency |
| **sum** | **1 483** | **~40 % of total gap** |

## 3. Proposal (b) — Static VGPR-bank analysis

### 3.1 Why it matters

RDNA3 has 4 VGPR banks. If two source operands of adjacent instructions
live in the same bank, the second one stalls. Which operands land in
which bank is **compile-time deterministic** but isn't in the ISA text:
it's determined by the LLVM register allocator. We currently model
bank conflict with a conservative ~1-cy penalty on *all* VOPD pairs;
our `raw-banks / vopd` bucket (246 tokens) is the cost of that
uniform guess.

### 3.2 Two ways to get bank maps

**Option 1 — RGA** (https://github.com/GPUOpen-Tools/radeon_gpu_analyzer):

```bash
# RGA 2.x for gfx1100
rga -s rocm-cl -c gfx1100 --livereg livereg.txt --isa isa.amdisa \
    --bank-conflicts bankconflicts.txt kernel.cl
```

RGA emits `bankconflicts.txt` with per-instruction VGPR bank assignments
and a list of conflicting source pairs. Accepts OpenCL / HLSL / GLSL /
SPIR-V input.

**Caveat:** our microbenches are built by tinygrad's DSL
(`extra/sqtt/rigorous_hw_test.py`), not OpenCL. Options to make them RGA-
digestible:
- Port the 30 priority kernels to OpenCL (tedious, one-time).
- Use `rocdisasm` / `llvm-objdump -d --mcpu=gfx1100` on the captured
  ELF (`extra/sqtt/captures/rigorous/*.pkl['lib']`) — gets us the ISA
  but not bank assignments directly.
- Use RGA's SPIR-V path against LLVM bitcode emitted by tinygrad's HIP
  backend.

**Option 2 — Write our own analyzer.**

The RDNA3 bank rule is public: each VGPR `vN` is in bank `N % 4` for a
wave64 lane, with dual-issue (VOPD) adding bank constraints on operand
pairs. Since our emulator already has the ISA in structured form
(`tinygrad/runtime/autogen/amd/rdna3/ins.py`), we can walk the kernel's
instruction stream and flag `(src_a, src_b)` pairs that hit the same
bank. No external tool required.

Pseudocode:

```python
def bank_of(vgpr_num: int) -> int: return vgpr_num % 4

def vopd_bank_conflict(inst):
  srcs = inst.src_vgprs
  if len(srcs) < 2: return 0
  return 1 if bank_of(srcs[0]) == bank_of(srcs[1]) else 0
```

Then `simd_arbiter.py` consults this per-instruction table instead of
applying a uniform VOPD penalty. We already maintain ISA metadata — this
is ~a day of plumbing.

**Recommendation:** do Option 2 locally (no HW time needed, no external
tool dependency), use Option 1 as ground-truth validation if something
looks off.

### 3.3 Bucket coverage

| Bucket | Misses | Bank-map-covered? |
|---|---:|:---:|
| raw-banks / vopd | 246 | ✓ — this is *exactly* this bucket |

## 4. Proposal (c) — Streaming Performance Monitor (longer-term)

Not required for the near-term push. Files here for completeness.

`capture_spm.py` (already in wave_probe) gives us per-kernel aggregate
PMC counts broken down per (SE, SA, WGP). That's enough for "where did
the waves go" but not for "when did each wave arrive / drain." True SPM
would sample counters at a fixed cycle cadence during the kernel.

### 4.1 What proper SPM enables

- `SQ_WAVES` over time → watch wave-count rise and fall; see the
  `s_endpgm` drain directly (wave.md §5 Q5).
- `SQ_BUSY_CYCLES` over time → which SIMD is active at each sample →
  direct view of the arbiter's cross-SIMD scheduling.
- Cache hit/miss rate vs time → LGKM variance correlation (wave.md §5 Q6).

### 4.2 What it costs

- `regRLC_SPM_MC_CNTL` + `regRLC_SPM_GLOBAL_MUXSEL_*` setup to pick
  counters and sample interval.
- Ring-buffer allocation + wptr poll (similar to SQTT buffer).
- Decoder for the SPM stream format (documented in ROCm's
  `aqlprofile` + Mesa's `ac_sqtt.c`).
- Estimated effort: ~1 week focused.

Projected close: ~140 tokens (lds bucket) + qualitative wins on
drain/startup accounting. Not the cheapest cycle — defer until after
(a) and (b).

## 5. Expected outcome, honestly

Attribution of the current 3 807-token gap, with each proposal's effect:

| Bucket | Misses | Close with (a) | Close with (b) | Still open |
|---|---:|:---:|:---:|:---:|
| "other" (mostly 16-miss kernels) | 1 494 | | | → wave.md (HW_ID/SQTT probes) |
| cndmask / depctr | 386 | ✓ | | |
| sgpr-mixed | 311 | ✓ | | |
| trans | 303 | ✓ | | |
| vcmp | 302 | ✓ | | |
| raw-banks / vopd | 246 | | ✓ | |
| trans-chain | 181 | ✓ | | |
| lds | 138 | | | → (c) SPM |
| remaining | ~446 | partial | partial | memory stochastic |

After (a) + (b) + wave.md landing: **~98 % MODAL / ~92 % strict** is a
realistic target. The last ~2 % is memory-timing stochasticity that
cycle-accurate deterministic emulation fundamentally can't predict —
the `STOCHASTIC_SCHEDULER_PLAN.md` path addresses that separately.

100 % is not realistic without either (i) modeling cache-line / MALL
state stochastically or (ii) relaxing the scorer on genuinely
non-deterministic events. Both are known options; both sit after the
accuracy work in this doc.

## 6. Concrete run plan for HW team

Assumes you've already run `wave.md` §1 scripts (hw_id, raw_sqtt,
decode_all_simds, pmc). If not, those come first.

### 6.1 Re-run with PERF token enabled

Apply the one-line change to `ops_amd.py:278` (or pass
`SQTT_TOKEN_EXCLUDE=0` to disable the default exclusion — same effect):

```bash
# Edit tinygrad/runtime/ops_amd.py:278 to remove the PERF_SHIFT | OR,
# leave the file alone and just override via env var on the command line.
sudo DEV=AMD AM_RESET=1 PROFILE=1 SQTT=1 MICROBENCH=1 VIZ=-2 \
     SQTT_TOKEN_EXCLUDE=0 \
     PYTHONPATH=. .venv/bin/python extra/sqtt/wave_probe/capture_raw_sqtt.py
```

(`SQTT_TOKEN_EXCLUDE` is already a `ContextVar` —
`ops_amd.py:24-26` — so env-var override is the minimal path.)

Expected: `extra/sqtt/wave_probe/captures/raw_sqtt_<ts>/` blobs will be
~5–10 % larger. No functional change otherwise.

Send back the same directory as before.

### 6.2 (Optional) Run the 30 priority kernels through RGA

If the HW box has RGA installed and the HIP backend can spit out LLVM
bitcode for our microbenches:

```bash
# Extract compiled kernels (already have these)
# For each kernel in captures/rigorous/*.pkl, pickle['lib'] is the ELF
rga -s rocm-bc --livereg livereg_<kernel>.txt --isa isa_<kernel>.amdisa \
    --bank-conflicts banks_<kernel>.txt <kernel>.bc
```

Send back the `livereg_*.txt` + `banks_*.txt` files. If the BC-extraction
path is a hassle, skip this step — we can do equivalent analysis locally
from the ISA dumps we already have.

### 6.3 Nothing else needed from HW

The PERF-token rerun is the whole ask for this doc. Everything else is
emulator-side work that happens back on our end.

## 7. What we do with the data

### Phase 1 — PERF decode
1. Parse PERF `arg` field per-instruction, build ground-truth latency table.
2. Update `test/mockgpu/amd/sq_timing/*` rules with measured values where
   they differ from our predictions.
3. Re-score. Expect +1 000 – +1 500 MODAL tokens.

### Phase 2 — Bank analysis
1. Local ISA walk (no HW time) → per-instruction bank-conflict table.
2. `simd_arbiter.py` consumes table; replaces uniform VOPD penalty.
3. Re-score. Expect +200 – +250 strict tokens.

### Phase 3 — If we still have budget
- (c) SPM integration, mostly for drain modeling.
- Stochastic memory latency per `STOCHASTIC_SCHEDULER_PLAN.md`.

## 8. Pointers (for reviewers)

- `tinygrad/runtime/ops_amd.py:278` — the one-line PERF-mask change.
- `tinygrad/renderer/amd/sqtt.py:330-338` — `PERF` / `PERF_RDNA4`
  packet decoders (already exist; just need semantic interpretation).
- `extra/hip_gpu_driver/soc21_enum.h:16128` —
  `SQ_TT_TOKEN_EXCLUDE_PERF_SHIFT = 0xb` definition.
- Mesa `src/amd/common/ac_sqtt.c` — authoritative PERF-arg format.
- `tinygrad/runtime/autogen/amd/rdna3/ins.py` — structured ISA metadata
  we'd use for local bank analysis.
- `extra/sqtt/wave.md` — wave-slot placement data request (run that
  first; PERF rerun piggybacks on the same capture scripts).
- `extra/sqtt/mes_notes.md` — MES review (for context, not needed here).
- `extra/sqtt/STOCHASTIC_SCHEDULER_PLAN.md` — memory-stochastic approach
  for the residual gap beyond what (a)+(b) can close.
- `github.com/GPUOpen-Tools/radeon_gpu_analyzer` — RGA, used in §3.
- `github.com/GPUOpen-Tools/radeon_gpu_profiler` — RGP reference.
- Commit `1bec956bf` — current snapshot of work.
