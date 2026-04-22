# SQTT Cycle Viewer

Interactive cycle-level visualizer for SQTT captures. Pre-processes a capture
directory into a compact JSON document, then renders it in a single-page
viewer with play/pause/step controls over a wave-per-row timeline.

## Session roadmap

- **Session 1 (landed):** ingest `extra/sqtt/captures/rigorous/*.pkl`, one row
  per wave with cycle-aligned instruction spans, category colouring, a draggable
  cycle cursor, play/pause/step ±1 controls, and a speed/zoom slider.
- **Session 2 (planned):** live utilization heatmap per unit at the cursor,
  instruction detail panel, pan + per-track horizontal scroll sync,
  keyboard-rich jumping (next stall, next VMEM, etc.).
- **Session 3 (planned):** RGP `.rgp` binary ingestion via `extra/sqtt/rgptool.py`,
  PMC per-WGP overlay, HW-vs-Emu diff inline, multi-run switcher within a
  single capture session.

## Quick start

```bash
# 1. generate the JSON (once per capture dir)
.venv/bin/python extra/sqtt/viewer/build_viewer_data.py \
    --captures extra/sqtt/captures/rigorous \
    [--raw extra/sqtt/wave_probe/captures/raw_sqtt_<ts>] \
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

| Key       | Action                         |
|-----------|--------------------------------|
| Space     | Play / pause                   |
| ← / →     | Step one instruction event ∓   |
| Home / End| Jump to cycle 0 / time_max     |
