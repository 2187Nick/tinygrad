# RGP Capture Pipeline (Vulkan RADV → .rgp → tinygrad parser)

This directory captures Radeon GPU Profiler traces of minimal Vulkan compute
probes and cross-references them with tinygrad's existing SQTT pickles to get
extra per-wave metadata (SIMD placement, timestamps) that `ops_amd.py`'s AM
backend capture does not expose.

## Why this path (and not ROCm/HIP rocprof)

tinygrad's AM backend takes over `/dev/kfd` directly, so neither the Radeon
Developer Panel nor rocprof can attach. Mesa's RADV driver, however, writes
.rgp files natively when `MESA_VK_TRACE=rgp` is set. We therefore mirror each
probe kernel as a Vulkan compute shader and use the HW scheduler's placement as
the cross-reference signal.

## Files

```
vkrun.c                 minimal C Vulkan compute harness (~200 LOC, libvulkan only)
build.sh                glslangValidator → SPIR-V + gcc → vkrun
capture_all.sh          runs every shader under MESA_VK_TRACE=rgp, saves .rgp
parse_rgp.py            loads .rgp → WAVESTART placement + timestamp clustering
shaders/*.comp          GLSL compute probes (1/2/4-wave workgroups)
out/                    build outputs (.spv, vkrun binary)
captures/               .rgp files + .waves.json sidecars + _summary.txt
```

## Build + capture

```bash
./build.sh              # compiles shaders + vkrun
./capture_all.sh        # GX=256 ./capture_all.sh to override WG count
PYTHONPATH=../../.. ../../../.venv/bin/python parse_rgp.py captures/*.rgp
```

Requires: `libvulkan-dev`, `glslang-tools`, Mesa RADV, a Radeon 7000/9000.

## Findings — 2026-04-18 (first capture run, 7900 XTX, Mesa 26.0.5)

**Shared-SIMD serialization hypothesis is FALSIFIED for 2-wave workgroups.**

For every 2-wave compute probe (`data_deps`, `probe_vmem_chain`, `exp_chain`,
`probe_cmp_chain`, `probe_branch_cost`), the RGP cluster analysis shows
`1-wave clusters: 256` — i.e., every workgroup's two waves landed on
**different SIMDs** (and therefore cannot contend for a shared VALU pipe).

Control experiments:
- `probe_single_wave.rgp` (32 threads, 1 wave/WG): 256 × 1-wave clusters ✓
- `probe_four_wave.rgp` (128 threads, 4 waves/WG): 64 × 2-wave + 128 × 3-wave
  clusters — i.e., 4-wave WGs split across 2 SIMDs (roughly 2+2 per WGP).

This means the remaining 28 exact-mismatches are **not** due to shared-SIMD
VALU contention. The session note's proposed "shared-SIMD serialization on
n==2" was correctly reverted — implementing it would model a contention that
doesn't exist. Root cause is elsewhere (SQ arbitration state across waves,
context-sensitive s_nop cycles, VOPD inter-pattern spacing).

## Handoff to remote team

All `.rgp` captures, `.waves.json` extractions, and scripts are checked in.
Pull and open the .rgp files in `RadeonGPUProfiler` (GUI) to see instruction
timing, HW utilization counters, and synchronization events — these may
reveal the SQ arbitration signal our headless parse is missing.

Key .rgp files of interest:
- `probe_data_deps.rgp` + `probe_vmem_chain.rgp` — the W0/W1 arbitration split
  that motivated the investigation. Both confirmed different-SIMD here.
- `probe_exp_chain.rgp` — VOPD inter-pattern spacing (12 of the 28 mismatches).
- `probe_single_wave.rgp` / `probe_four_wave.rgp` — controls.
