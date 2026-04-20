# MES Specification Review — Timing-Relevance Notes

**Source:** `extra/sqtt/micro_engine_scheduler.pdf` (AMD MES Specification, April 2024, 54 pages)
**Reviewer:** emulator bounty team (2187Nick)
**Date:** 2026-04-20
**Context:** MODAL 66055/69862 (94.6%), strict 55677/69862 (79.7%) at commit `8597550f8`.
Looking for anything in MES that explains our remaining gap.

---

## TL;DR

**MES is queue-level scheduling firmware, not wave-slot allocation.** It maps user
queues onto the GPU's 8 compute HW queues (HQDs) and 2 GFX HQDs. It decides
*which queue runs next*. It does **not** decide which SIMD a wave lands on — that's
the **SPI (Shader Processor Input)** block, which sits downstream of the queue
manager. SPI/SQ behavior is the unknown we flagged in `wave.md`; MES does not
fill it in.

That said, the MES doc gives us three useful things:

1. **Confirmation** that wave-slot placement is *not* under MES control — so the
   data we asked for in `wave.md` is still the right thing to chase.
2. **A cheap way to timestamp dispatches** (the MES scheduler log) that can be
   cross-referenced with SQTT time to pin the dispatch-to-WAVESTART latency.
   That is a candidate source for the ~250–280 cy `s_endpgm` drain and the
   LGKM variance we can't currently explain (question 5–6 in `wave.md`).
3. **Controls that *force* placement determinism** (CU reservation, wave
   limiting, SET_SE_MODE=SINGLE_SE) which the HW team can toggle to isolate
   which of our possibilities is correct.

Nothing in the doc contradicts our current emulator model. Nothing tells us
directly how to fix the 3 807-token gap. The MES doc is context, not a fix.

---

## 1. What MES actually does (so we can stop looking here)

MES is a firmware engine on the `ME` block. KMD submits API frames (64 DWORDs,
`mes_api_def.h`) to a ring buffer. MES then:

- Holds up to 8 compute queues connected to HW queues (`MAX_COMPUTE_PIPES=8`,
  per-pipe `compute_hqd_mask`).
- Holds up to 2 GFX queues (`MAX_GFX_PIPES=2`, `second_gfx_pipe_enabled`).
- Rotates user queues onto HW queues when over-subscribed (round-robin with
  priority adjustments).
- Applies prioritization via: mid-command-buffer preemption, wave limiting,
  pipe priority, dispatch tunneling, queue quantum, queue connection priority,
  **compute unit reservation**.

Once a queue is connected to a pipe, the *Queue Manager HW* (not MES) selects
which connected queue actually issues the next packet. From there, the CP
(Command Processor) drives PM4 packets into the dispatch path, which eventually
hands work to the SPI, which finally allocates `(SE, SA, WGP, CU, SIMD, wave_slot)`.

**Implication for us:** The `wave_idx → SIMD` question (`emu.py:3536`,
`simd_arbiter.py:141`) lives entirely in SPI/SQ. Nothing MES can tell us about
it. Our `capture_hw_id.py` / `capture_raw_sqtt.py` scripts are still the right
probe.

## 2. Timing-relevant things we extracted

### 2.1 `process_quantum` in 100 ns units

`MES_SCH_API_ADD_QUEUE.process_quantum` (pg 23) is measured in 100 ns units.
This is the minimum time a process stays mapped before MES considers
unmapping it. For our microbenches this never matters — the kernel is
single-digit microseconds and nothing else is competing — but it confirms the
timing unit MES uses matches the `time_before_call` / `time_after_call` GPU
timestamps in the scheduler log.

### 2.2 MES scheduler log = cheap dispatch-edge timestamp

**Most useful single item in the doc** for our purposes. Enabled via:

```
MES_SCH_API_SET_HW_RSRC.event_intr_history_gpu_mc_ptr  (debug-only)
MES_SCH_API_SET_HW_RSRC.enable_mes_event_int_logging = 1
MES_SCH_API_SET_HW_RSRC.enable_mes_sch_stb_log = 1     (Smart Trace Buffer)
```

The log has three parallel circular buffers (pg 47–49):

- `api_history[]`: KMD→MES events, with `time_before_call` + `time_after_call`
  (GPU timestamps) per entry.
- `event_log_history[]`: MES→CP events (`MES_EVT_LOG_MAP_QUEUE`,
  `UNMAP_QUEUE`, `QUERY_STATUS`, `UNMAP_RESET_QUEUE`) with
  `doorbell_offset` + timestamps.
- `interrupt_history[]`: CP→MES interrupts with `time_trace`.

**Why we care:** these are GPU-clock timestamps bracketing the moment a queue
gets connected to a pipe. If we can align them with the SQTT WAVESTART
timestamps in our captures, we learn:

- How much wall time passes between queue-map and first WAVESTART (and
  whether it's deterministic).
- Whether the wave-0 "extra" time we attribute to "`s_endpgm` drain or LGKM
  first-hit latency" is actually a pre-kernel setup cost that we are
  misattributing to the wrong edge of the pipeline.

Neither we nor the upstream kernel has to parse MES firmware — KMD has to
allocate the log buffer, MES fills it, userspace (or a KMD debugfs export)
reads it.

### 2.3 Prioritization knobs that force determinism

These don't model what hardware does *by default* — they let the HW team pin
the experiment so we get repeatable `(cu, simd)` placements (question 3 of
`wave.md`):

- **Compute unit reservation** (pg 14): "a certain number of compute units to
  be carved out and only made available for a particular queue." If the HW
  team can reserve only one CU for our probe queue, we've eliminated
  cross-CU drift as a confound.
- **Wave limiting**: caps the number of concurrent waves a queue can issue.
  Forcing the limit low (e.g., `=1`) would serialize waves and tell us if
  any of the "16-miss" pattern we see is specifically about concurrent
  launch (arbitration) rather than about sequencing.
- **Dispatch tunneling**: not useful for us — it affects other queues.
- **Queue quantum**: not useful — our runs don't oversubscribe.

### 2.4 `MES_SCH_API_SET_SE_MODE` (pg 40)

Switches between `SINGLE_SE` / `DUAL_SE` / `LOWER_POWER`. The 7900 XTX has 6
SEs but the default RGP-style "one CS workload" typically runs DUAL_SE. If
HW puts us into `SINGLE_SE` mode, wave placement is forced into a single
shader engine, which narrows the `wave_idx → (CU, SIMD)` function.
`log_seq_time` bit in the API means the mode-switch sequence itself can be
timestamped, so we can verify the mode actually took effect.

### 2.5 `SET_SHADER_DEBUGGER` (pg 45) — SQ_DEBUG registers

```c
uint32_t single_memop  : 1;  // SQ_DEBUG.single_memop
uint32_t single_alu_op : 1;  // SQ_DEBUG.single_alu_op
uint32_t spi_gdbg_per_vmid_cntl;
uint32_t tcp_watch_cntl[4];
```

Two hardware-level step-mode bits:

- `SINGLE_MEMOP` — one memory operation per issue; serializes VMEM.
- `SINGLE_ALU_OP` — one ALU op per issue; serializes VALU.

These are the exact knobs that would let HW decouple VALU-pipe effects from
scheduling effects. If we can run `mb_valu_add_n16` with `SINGLE_ALU_OP=1`
and compare the WAVE_ID-per-cycle pattern, we can tell whether our
`simd_arbiter` "same-SIMD peer cluster" model is operating on real peers or
on imagined peers. **Not a near-term ask** — RGP/SQTT tool integration
required — but worth filing for later.

### 2.6 `debug_vmid` (pg 25, 38)

Flag in `ADD_QUEUE` saying "this queue is the one RGP is profiling."
Confirms our existing SQTT captures already go through the MES debug VMID
path. Nothing to change.

### 2.7 `alignment_mode_setting` → `SH_MEM_CONFIG`

Programs the shader memory-alignment mode. Controls whether unaligned
global_load/store faults or silently coalesces. The sgpr-mixed bucket
(311 tokens, mostly `global_store_b32` timings after SALU) is consistent
with sub-optimal alignment handling changing the VMEM cycle cost — but this
is set once at queue-add time, not per-kernel, so it can't explain
per-kernel variance. **Not a root cause of our gap.**

### 2.8 `is_long_running` flag (pg 26)

"Indicates that the queue has a long running compute job." If set, MES
treats the queue differently for preemption. Our microbenches are not
long-running so this flag is 0; worth verifying the HW team's captures also
run with it 0.

## 3. What is NOT in MES that we still need

- **Per-wave SIMD_ID assignment policy** — SPI, not MES.
- **VALU / VMEM / LGKM cycle budgets** — those live in the execution units
  (VALU/VMEM pipelines, L0/L1/L2 cache), not in scheduling firmware.
- **Wave lifetime / `s_endpgm` drain** — neither the MES spec nor the CP
  touches this; it's SPI tearing down the wave slot.
- **Stochastic SMEM/VMEM latency source** — memory subsystem (cache, BCache,
  L1, L2, MALL), not MES.

In short: MES gives us the *timeline brackets* around a kernel launch. It
does not give us the internal clockwork.

## 4. Testable locally — what we checked

I did quick greps against the repo to see if anything here maps to actual
emulator state:

- `grep -rn "MES\|mes_" test/mockgpu/amd/` → one hit (an SDMA opcode name
  collision), no MES modeling. Expected: our emu skips MES entirely, starts
  from the PM4 dispatch packet.
- The `simd_arbiter.py` assumption `SIMD_ID = wave_idx % 4` is unaffected
  by any of these flags. No change needed there until `decode_all_simds`
  data arrives.
- `SH_MEM_CONFIG.alignment_mode_setting` is not modeled in emu; our
  `global_store_b32` timing uses a fixed 1-cycle issue regardless of
  alignment. This is fine for the current test corpus (all aligned) but
  worth a note for future unaligned-access microbenches.

**Nothing in MES maps to a code change we could make today without HW data
to validate it.**

## 5. Test suggestions for the HW team (real hardware + MES)

File these under the `wave.md` handoff — they're follow-ups, not blockers.
All require `sudo` (MES log buffer allocation) or equivalent debug access
(KFD/amdgpu debugfs).

### 5.1 Enable MES scheduler log during a microbench capture (highest priority)

```bash
# Pseudo-code — exact knob depends on amdgpu debugfs / KFD version
echo 1 > /sys/kernel/debug/dri/0/amdgpu_mes_stb_log     # enable STB log
echo 1 > /sys/kernel/debug/dri/0/amdgpu_mes_event_log   # enable event log
# Then run the microbench suite as in wave.md §1.
# After the run, dump the log buffer (format: struct MES_EVT_INTR_HIST_LOG, pg 47)
```

What to extract from the log:

- For each microbench kernel launch, the `time_before_call` /
  `time_after_call` of the `MES_EVT_LOG_MAP_QUEUE` event.
- Compare to the first WAVESTART timestamp in the corresponding SQTT blob.
- Compute the delta distribution. If it's ~constant, it's a fixed CP
  overhead we can add to the emu as a one-shot constant. If it varies
  with launch size, that's where our wave-0 LGKM variance is hiding.

Expected deliverable: one extra JSON beside each SQTT blob:
`{"kernel": "mb_valu_add_n16", "queue_map_ts": 12345678, "first_wavestart_ts": 12349012, "delta_cy": 3334}`.

### 5.2 Pin to a single CU via compute-unit reservation

If the HW team's driver exposes CU reservation (it does via
`COMPUTE_STATIC_THREAD_MGMT_SE0..5` on RDNA3 — see `gfx_v11_0.c` in the
amdgpu driver, or `MES_SCH_API_ADD_QUEUE` indirectly), rerun
`capture_hw_id.py` with the probe queue reserved to a single CU:

- Does `cu_id` always match the reserved CU? (sanity check the plumbing)
- Do all 16 waves of a 16-wave launch still spread across the 4 SIMDs of
  that one CU, or collapse onto SIMD 0 of it?

If they collapse onto SIMD 0, possibility #1 in `wave.md` §3 ("all 16
waves on SIMD 0 of one CU") is confirmed and the arbiter rewrite is
straightforward.

### 5.3 Run with `SET_SE_MODE = SINGLE_SE`

Compare placements against default DUAL_SE:

- Does a 16-wave launch under SINGLE_SE force everything into one SE,
  and if so, is SIMD distribution the same as in DUAL_SE (just on one
  SE) or different?
- If distribution changes, the emu's 1-SE assumption is wrong for multi-
  SE modes. We'd then need an SE-count parameter in the arbiter.

### 5.4 Wave limiting sweep

Via MES queue priority / wave-limit: run the same 16-wave microbench with
wave limit = 1, 2, 4, 8, 16. For each, observe:

- `capture_hw_id` placement histogram.
- Per-wave SQTT wall-clock time (start→end).

Hypothesis: at wave-limit=1 the kernel serializes through one SIMD, SQTT
matches our single-wave model exactly; at wave-limit=16 we see whatever
natural contention the SPI allocator produces. The *shape* of the
wall-clock curve between those extremes tells us whether our arbiter's
"dispatch-vs-execute" handling (see `simd_arbiter.py:87-106`) is linear,
capped, or something else.

### 5.5 Record `interrupt_history` during s_endpgm

Wave-0's ~260 cy drain from last `global_store_b32` to `s_endpgm` stamp
is question 5 in `wave.md`. The MES interrupt history (`time_trace` per
interrupt) should bracket the CP-side completion of the dispatch. If the
interrupt arrives within ~10 cy of the `s_endpgm` stamp, the drain is
inside the SQ and we need to model a WAVEEND-latency constant. If it
arrives hundreds of cycles later, the drain is downstream (cache flush,
write-combining, DRAM settle) and we can model it as a per-queue tail
constant.

## 6. Revised gap-attribution in light of MES

No change to the `wave.md` §2 breakdown. MES gives us a complementary data
stream but the failure modes still point at SPI/SQ arbitration and memory
subsystem latency, not scheduling. The one update: add to the wave.md
question list:

- "If the MES scheduler log is enabled during the captures, send the log
  dumps alongside the SQTT blobs so we can correlate dispatch-edge
  timestamps."

## 7. Pointers

- `extra/sqtt/micro_engine_scheduler.pdf` — source document.
- `extra/sqtt/wave.md` — primary handoff doc; this is a supplement.
- `test/mockgpu/amd/emu.py:3536` — where wave-slot assignment lives in
  emu (unchanged; MES does not affect this).
- Pages worth re-reading in the PDF if HW team has specific questions:
  - pg 14–15: prioritization features table.
  - pg 23–26: ADD_QUEUE fields (process_quantum, is_long_running, debug_vmid).
  - pg 40–41: SET_SE_MODE.
  - pg 45–46: SET_SHADER_DEBUGGER (SQ_DEBUG bits).
  - pg 47–50: Scheduler log format (api_history / event_log_history /
    interrupt_history).
