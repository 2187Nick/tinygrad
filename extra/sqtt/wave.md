# Wave-Slot Allocation: HW Data Request

**Addressed to:** tinygrad HW-access team
**From:** emulator bounty team (2187Nick)
**Date:** 2026-04-20
**Status:** MODAL 66055/69862 (94.6%), strict 55677/69862 (79.7%). Closing the last 5.4% is blocked on data only the real 7900 XTX can produce.

---

## 1. TL;DR — what we need

Run three scripts on a real gfx1100 box and ship the outputs back:

```bash
# 1. Wave → (SIMD, CU) placement for varying launch sizes (no sudo needed, fast)
DEV=AMD PYTHONPATH=. .venv/bin/python extra/sqtt/wave_probe/capture_hw_id.py

# 2. Raw SQTT blobs for the 30 top-miss microbenches (preserves all SIMDs)
sudo DEV=AMD AM_RESET=1 PROFILE=1 SQTT=1 MICROBENCH=1 VIZ=-2 \
     PYTHONPATH=. .venv/bin/python extra/sqtt/wave_probe/capture_raw_sqtt.py

# 3. Decode the raw blobs across all SIMDs and produce a summary JSON
.venv/bin/python extra/sqtt/wave_probe/decode_all_simds.py \
     extra/sqtt/wave_probe/captures/raw_sqtt_<timestamp>

# 4. Per-WGP performance counters (SQ_WAVES etc.) across all microbenches
#    — cross-checks the per-wave placement from script #2 (see §9).
sudo DEV=AMD AM_RESET=1 PROFILE=1 PMC=1 SQTT=1 MICROBENCH=1 VIZ=-2 \
     PYTHONPATH=. .venv/bin/python extra/sqtt/wave_probe/capture_spm.py
```

Send back:
- `extra/sqtt/wave_probe/captures/hw_id_<ts>.json`
- `extra/sqtt/wave_probe/captures/raw_sqtt_<ts>/` (the whole directory incl. `all_simds_summary.json`)
- `extra/sqtt/wave_probe/captures/pmc_<ts>/` (the whole directory incl. `summary.json`)

Nothing private — the blobs contain only instruction-level SQTT packets
(no register contents, no data) and the kernel binaries already live in
`extra/sqtt/captures/rigorous/`.

---

## 2. Where the 5.4% gap lives

Ran the MICROBENCH harness comparison today at commit `8597550f8`:

| Category             | Missed tokens | Kernels | Comment                                    |
|----------------------|--------------:|--------:|--------------------------------------------|
| other (mostly vopd)  |         1 494 |     125 | 90+ kernels missing exactly 16 tokens each |
| cndmask / depctr     |           386 |      16 | scalar-pipe contention                     |
| sgpr-mixed (s_bfe…)  |           311 |      31 | SALU → VMEM store 8-cycle delay            |
| trans (exp/log/rcp)  |           303 |      23 | trans-pipe wave stagger                    |
| vcmp (cndmask)       |           302 |      17 | vcc-propagation timing                     |
| raw-banks / vopd     |           246 |      15 | VGPR bank / wave arbitration               |
| trans-chain (f4)     |           181 |      18 | trans chains longer than 4                 |
| lds                  |           138 |       8 | lds write→read forwarding                  |
| (~15 more buckets)   |           … |      … |                                            |
| **Total**            |    **3 807** | **295** | (out of 69 862 tokens across 806 kernels)  |

**The single biggest signal** — 84 of the 90+ 16-miss kernels have a `_n4`
suffix, 21 have `_n8`. That's *one token per wave* in a 16-wave launch.
Almost all of them mis-stamp the trailing `s_endpgm`, or mis-stamp a
VALU / VOPD at a burst-transition boundary. Both symptoms point at one
question: **which SIMD is each wave actually running on?**

## 3. Why wave-slot placement is the load-bearing unknown

The emulator assumes `SIMD_ID = wave_idx % 4`
(`test/mockgpu/amd/emu.py:3536`, `test/mockgpu/amd/sq_timing/simd_arbiter.py:141`).
That's a guess calibrated against the captures we already have.

But the existing captures in `extra/sqtt/captures/rigorous/*.pkl` were
decoded through `tinygrad/renderer/amd/sqtt.py:657`, which **hard-filters
`simd == 0`**:

```python
if isinstance(p, (WAVESTART, WAVESTART_RDNA4, CDNA_WAVESTART)) and simd == 0:
  traced_cu = cu
return cu == traced_cu and simd == 0
```

So every wave we think we have data for lives on SIMD 0 of one CU. We've
never observed the true per-wave (CU, SIMD) distribution. Three possibilities:

1. **All 16 waves land on SIMD 0** of a single CU (the SPI packs tightly).
   If true, `wave_idx % 4` is wrong — all waves serialise through **one**
   VALU port, not four. The arbiter's modelling is off by 4×.
2. **Waves spread 4-4-4-4 across SIMDs** of one WGP.
   `wave_idx % 4` is right and the remaining gap is elsewhere (drain,
   memory latency).
3. **Waves spread across multiple CUs.**
   Then stamp ordering is cross-CU, not single-CU, and the arbiter needs a
   totally different model.

We cannot distinguish these without reading `HW_ID` on real hardware.

Already-known dead-end (`simd_arbiter.py:87-106`): applying a blanket
1-cy-per-SIMD port-availability rule regresses strict -24 K tokens.
That's the DISPATCH-vs-EXECUTE insight — SQTT stamps at dispatch, not at
execute, so the queue absorbs back-to-back same-SIMD issues invisibly.
Which means a narrow rule (gated on actual same-SIMD peer clustering)
is the correct approach — but "same-SIMD" needs ground truth.

## 4. The three scripts

### 4.1 `capture_hw_id.py` — primary data

Dispatches a 7-instruction kernel (one VALU, one SMEM load, `s_getreg_b32`,
one `global_store_b32`) across launch sizes 1 / 2 / 4 / 8 / 16 / 32 / 64
waves. Each wave writes its 32-bit `HW_ID` register to `buffer[lane]`. We
decode bits on readback:

```
[3:0]  WAVE_ID       wave slot within SIMD (0-15)
[5:4]  SIMD_ID       0-3
[7:6]  PIPE_ID       always 0 for CS
[11:8] CU_ID         CU within WGP
[14:12] SH_ID
[18:15] SE_ID
[23:20] TG_ID        workgroup within CU
```

Each sweep runs 10 times so we can see whether placement is deterministic
across repeats. The tool prints a per-sweep placements histogram and
saves full per-run detail to JSON.

**What we learn:** the SPI's allocator policy. Does it fill SIMD 0 first
then 1-2-3, or round-robin, or load-balance on occupancy? Does `wave_idx`
correlate with `SIMD_ID` at all?

### 4.2 `capture_raw_sqtt.py` — 30 priority kernels, raw blobs

Runs the 30 microbenches with the largest MODAL miss counts, but saves
the **pre-decode SQTT blob** plus the kernel ELF. With the raw blob we
can re-decode without the `simd == 0` filter, so we see WAVESTART /
WAVEEND packets for every SIMD that participated.

One file per kernel under `captures/raw_sqtt_<ts>/`.

### 4.3 `decode_all_simds.py` — offline analysis

Walks the raw-blob directory, decodes every (CU, SIMD) unit, and emits
`all_simds_summary.json`:

```
{
  "mb_vopd_chain_n4_raw": {
    "total_waves_observed": 16,
    "unique_units": 1,
    "per_cu_simd_wave_count": {"cu0_simd0": 16},
    "simd_balance": {"cu0": {"0": 16, "1": 0, "2": 0, "3": 0}},
    "per_wave_placements": {"0": [{"cu":0,"simd":0, …}], …}
  },
  …
}
```

The final block of the script prints a **match rate**: how often does
`wave_idx % 4` match the observed SIMD? If that's <99%, the arbiter's
whole SIMD-mapping assumption needs to change.

## 5. Open questions — please answer with the data

1. **Placement policy.** For a 16-wave launch of 1 wave / WG (i.e. 16
   separate workgroups), do all 16 waves land on SIMD 0? Or spread 4×4?
   Or multi-CU?
2. **Single-WG-multi-wave placement.** For a 1-WG launch with 4 waves
   inside it (e.g. `local_size=256`, `wave_size=64`), do those 4 waves
   stay together on one SIMD or spread across the WGP's 4 SIMDs?
3. **Run-to-run stability.** Does the same launch reproduce the same
   wave→SIMD assignment? Or does it drift?
4. **Cold vs warm dispatcher.** Does the first kernel after reset see
   a different placement than the 10th?
5. **s_endpgm drain.** In wave 0 of nearly every microbench, HW shows
   ~250-280 cycles between the last `global_store_b32` stamp and the
   `s_endpgm` stamp. The emulator emits 1 cycle. What's HW actually
   draining — VMEM write queue, VALU pipeline, WAVEEND packet latency?
   An RGP timeline view around the final wave would pin this down.
6. **LGKM_WAIT variance.** First `s_waitcnt_lgkmcnt` after an SMEM load
   varies 747-990 cy on HW across the kernels we've sampled. Is this
   pure DRAM variance or is part of it deterministic (e.g. wave slot
   affects cache homing)?

## 6. What we'll do with the data

If `simd_balance` shows all 16 waves on SIMD 0:

- Rewrite `simd_arbiter.simd_for_wave()` to return 0 for every wave.
- The per-SIMD VALU port becomes a single port shared by all 16 waves
  of the workgroup — narrow-gating on "peer cluster ≥ 4" becomes the
  whole arbitration signal, not a supplement.
- Expected gain: ~1 500 of the 1 494 "other" bucket misses (the 16-miss
  kernels).

If it shows 4×4 spread matching `wave_idx % 4`: arbiter SIMD mapping is
correct, gap is elsewhere (drain, memory). We switch focus to s_endpgm
drain modelling.

If it shows something else (e.g. occupancy-based or multi-CU): rewrite
the arbiter per the observed policy. Priority will match whichever
launch-size bucket drives the most misses — sweep data from script #1
tells us.

## 7. Format / handoff

- Scripts are self-contained. No new deps beyond what tinygrad already uses.
- `capture_hw_id.py` needs no sudo — just `DEV=AMD`.
- `capture_raw_sqtt.py` needs sudo (SQTT reg access).
- `capture_spm.py` needs sudo (PMC reg access).
- All outputs are JSON except the raw blobs (pickle, ~10-100 KB each).
- Anything that errors in the middle of a sweep is logged inline and doesn't
  halt the rest — partial results are still useful.
- Send the whole `extra/sqtt/wave_probe/captures/` directory back; we'll
  fold the findings into `simd_arbiter.py` and document the new placement
  model inline.

## 8. Pointers (for reviewers)

- `test/mockgpu/amd/emu.py:3536` — where EMU decides wave → SIMD mapping.
- `test/mockgpu/amd/sq_timing/simd_arbiter.py:139-147` — the one place the
  mapping is *used*; updating only this function is enough to change the
  model.
- `tinygrad/renderer/amd/sqtt.py:298-307` — `WAVESTART` packet layout
  (has `simd`, `cu`, `wave` fields we need).
- `extra/sqtt/rigorous_hw_test.py` — harness that produces the
  `.pkl` trace files we score against.
- `extra/sqtt/mes_notes.md` — separate review of AMD's MES firmware spec
  (queue-level scheduler). TL;DR: MES doesn't own wave-slot placement, so
  it can't directly answer this handoff's questions, but it does expose
  determinism knobs (CU reservation, wave limiting, `SET_SE_MODE=SINGLE_SE`)
  that would help HW team isolate the variable if §5-§8 data is ambiguous.
- `github.com/tinygrad/7900xtx` — tinygrad's public 7900 XTX reverse-eng
  notes. `docs/CU.md`, `docs/MEC.md`, `docs/CP.md` cover the dispatch path
  from the IP-block side; useful cross-reference if a reviewer wants the
  "what are CU/MEC/CP" picture.
- Commit `8597550f8` — current committed snapshot of the work.

## 9. Optional: per-CU/WGP performance counters (script #4)

**Script:** `extra/sqtt/wave_probe/capture_spm.py`

Exact answer to the "monitor all workgroups" idea. tinygrad already has a
PMC (performance-counter) infrastructure — `tinygrad/runtime/ops_amd.py:1031`,
enabled via `PMC=1`. On gfx11, SQ-block counters are broken down **per
(SE, SA, WGP)** — i.e. we get the count *per compute-unit-pair* for each
kernel run, from one PM4 `COPY_DATA` packet pair bracketing the dispatch.

**What we capture:**

| Counter | What it tells us |
|---|---|
| `SQ_WAVES` | Total waves that ran on each WGP — *direct* placement histogram |
| `SQ_BUSY_CYCLES` | How long each WGP was active |
| `SQ_INSTS_VALU` | VALU issue rate per WGP — reveals VALU-pipe contention |
| `SQ_INSTS_SALU` | SALU issue — cross-check against scalar-pipe bucket |
| `SQ_INSTS_VMEM` | VMEM issue — cross-check against memory-latency variance |
| `SQ_INSTS_LDS` | LDS instructions retired per WGP |
| `GRBM_GUI_ACTIVE` | Total GPU-active cycles (scalar denominator) |
| `GL2C_HIT` / `GL2C_MISS` | Per-instance L2 cache behavior — partial answer to §5 Q6 |

**Why this complements script #2.** SQTT (script #2) with the `simd==0`
filter removed gives per-WAVE placement. PMC (script #4) gives per-WGP
*aggregate* totals. Together they cross-check: if script #2 says "16 waves
on `cu0_simd0`" but script #4 says "SQ_WAVES has 4 WGPs active," we know
one of the decoders is wrong.

**Streaming vs aggregate.** True SPM (Streaming Performance Monitor)
samples counters at a fixed cadence during the kernel, letting us see
time-evolution of occupancy. tinygrad's PMC gives the *sum* over the
kernel lifetime. For now the sum is enough: "which WGPs did the 16 waves
of an `_n16` launch hit?" is answered by `SQ_WAVES` alone. If a future
question depends on *when* waves arrived (e.g., "did waves 8-15 start
after waves 0-7 drained?") we'd need to revisit and wire up true SPM
via `regRLC_SPM_MC_CNTL` + a ring buffer.

**Run command:**

```bash
sudo DEV=AMD AM_RESET=1 PROFILE=1 PMC=1 SQTT=1 MICROBENCH=1 VIZ=-2 \
     PMC_COUNTERS=SQ_WAVES,SQ_BUSY_CYCLES,SQ_INSTS_VALU,SQ_INSTS_SALU,SQ_INSTS_VMEM,SQ_INSTS_LDS,GRBM_GUI_ACTIVE,GL2C_HIT,GL2C_MISS \
     PYTHONPATH=. .venv/bin/python extra/sqtt/wave_probe/capture_spm.py
```

**Output:** `extra/sqtt/wave_probe/captures/pmc_<ts>/<kernel>.json` per
kernel, plus `summary.json` with top-level analysis. Each file has:

```json
{
  "kernel": "mb_vopd_chain_n4_raw",
  "counters": {
    "SQ_WAVES": {"block": "SQ", "rows": [{"se":0,"sa":0,"wgp":0,"value":16}, ...], "total": 16},
    ...
  },
  "wave_distribution": {"per_wgp": {"se0_sa0_wgp0": 16}, "total_waves": 16, "unique_wgps_used": 1},
  "balance": {"SQ_WAVES": {"nonzero_units": 1, "total_units": 60, "balance_ratio": 1.0}}
}
```

**Key value added:** if `wave_distribution.unique_wgps_used == 1` across
all microbenches, the SPI is *definitely* packing tight — possibility #1
from §3. If it's >1 and the balance_ratio is near 1.0, we're seeing even
load-balance across WGPs; script #2's `simd_balance` field then tells us
the within-WGP split.

Run #4 after #1/#2/#3. It's cheap — adds ~1 ms per kernel — and the
`SQ_WAVES` histogram is by far the most direct signal we can get for
the wave-placement question.
