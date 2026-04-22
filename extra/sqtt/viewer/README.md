# SQTT Cycle Viewer

Interactive cycle-level visualizer for SQTT captures. Pre-processes a capture
directory into a compact JSON document, then renders it in a single-page
viewer with play/pause/step controls over a wave-per-row timeline.

## Session roadmap

- **Session 1 (landed):** ingest `extra/sqtt/captures/rigorous/*.pkl`, one row
  per wave with cycle-aligned instruction spans, category colouring, a draggable
  cycle cursor, play/pause/step ±1 controls, and a speed/zoom slider.
- **Session 2 (landed):** utilization strip (stacked per-category wave count
  over time) above the timeline, 48-WGP × 4-SIMD topology grid in a right panel,
  click-to-pin instruction detail with ±5 surrounding instructions and cycle
  deltas, wave / (cu,simd) filtering, keyboard jumps to next VMEM / next stall /
  next wave start / ±1 cycle nudge.
- **Session 3 (landed):** HW-vs-Emu diff colouring with toggle (`D` key) and
  per-kernel strict-match stat in the header bar (`emu strict  X% (exact/compared)`);
  tooltip + pinned-instruction detail show `hw Δ / emu Δ / diff / status`;
  `M` jumps to the next mismatched instruction; build script grows
  `--emu-timing <timing_data.json>` (per-instruction deltas) and `--rgp
  <rgp_dir>` (SE-wide WAVESTART placement list attached to each kernel entry).

## Quick start

```bash
# 1. generate the JSON (once per capture dir)
.venv/bin/python extra/sqtt/viewer/build_viewer_data.py \
    --captures extra/sqtt/captures/rigorous \
    [--raw extra/sqtt/wave_probe/captures/raw_sqtt_<ts>] \
    [--rgp extra/sqtt/rgp/captures] \
    [--emu-timing extra/sqtt/timing_data.json] \
    [--out  extra/sqtt/viewer/viewer_data.json]

# 2. serve the viewer
.venv/bin/python -m http.server 8765 --directory extra/sqtt/viewer
# open http://localhost:8765/
```

The `--raw` argument is optional and enriches each wave with its observed
`(cu, simd)` placement by re-decoding the matching raw SQTT blob (same logic
as `extra/sqtt/wave_probe/decode_all_simds.py`). Without it, wave rows are
labelled by `wave_id` only.

## Data schema

`viewer_data.json` is deliberately flat so the browser can parse it quickly:

```
{ schema: 1, arch: "gfx1100", kernel_count: N,
  kernels: { "<name>": {
    total_waves, total_instructions, time_min, time_max,
    waves: [{
      wave_id, cu, simd, inst_count, time_min, time_max,
      instructions: [{ idx, pc, t, type, inst, cat }, ...]
    }, ...]
  }, ... } }
```

Times (`t`, `time_min/max`) are cycles, normalised so the earliest sample in
the kernel sits at 0. `cat` is one of `valu | salu | vmem | smem | lds |
vopd | trans | branch | wait | other`.

## Keyboard shortcuts

| Key       | Action                              |
|-----------|-------------------------------------|
| Space     | Play / pause                        |
| ← / →     | Step one instruction event ∓        |
| , / .     | Nudge cursor ±1 cycle               |
| V         | Jump to next VMEM instruction       |
| S         | Jump to next wait/stall instruction |
| N         | Jump to next wave start             |
| Home / End| Jump to cycle 0 / time_max          |
| Esc       | Clear pinned instruction            |
| F         | Clear wave / SIMD filter            |
| D         | Toggle category / diff colouring    |
| M         | Jump to next emu-mismatch           |

## Interactions

- Click an instruction span → **pin** it (full detail in right panel with ±5
  context, `Δprev` / `Δnext`, category chips). Cursor snaps to that cycle.
- Click a wave's row label → **filter** the timeline to just that wave.
- Click a topology cell (when `--raw` was supplied) → **filter** to all waves
  on that `(cu, simd)` pair.
- Click the filter badge in the stats bar (or press `F`) → clear the filter.
