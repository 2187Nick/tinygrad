#!/usr/bin/env bash
# Drive the BEAM_EMU vs BEAM_HW selection-quality sweep.
# Each step runs in a fresh process so env vars don't bleed across modes.
#
#   1. step_hw     — BEAM=N timing on real silicon for every workload. NEEDS sudo.
#   2. step_emu    — BEAM=N timing via cycle emulator. No sudo. Slow (1-3 min/workload).
#   3. step_verify — run emu-chosen kernels on real HW to time them. NEEDS sudo.
#   4. step_report — tabulate selection quality.
#
# Usage:
#   bash extra/sqtt/beam_emu_vs_hw.sh [BEAM_N] [WORKLOADS]
#   bash extra/sqtt/beam_emu_vs_hw.sh 4
#   bash extra/sqtt/beam_emu_vs_hw.sh 2 matmul_64,softmax_64
set -u
cd "$(dirname "$0")/../.."

BEAM_N="${1:-4}"
WORKLOADS="${2:-matmul_64,matmul_128,softmax_64,elementwise_4096}"
echo "=== BEAM_EMU selection-quality sweep (BEAM=${BEAM_N}, workloads=${WORKLOADS}) ==="
echo

echo "--- step 1/4: HW timing ---"
sudo --preserve-env=PATH env DEV=AMD AM_RESET=1 PYTHONPATH=. .venv/bin/python \
  extra/sqtt/beam_emu_vs_hw.py hw --beam-n "$BEAM_N" --workloads "$WORKLOADS"

echo
echo "--- step 2/4: EMU timing (no HW, no sudo) — slow, ~1-3 min per workload ---"
DEV=AMD MOCKGPU=1 PYTHON_REMU=1 PROFILE=1 SQTT=1 BEAM_EMU=1 PYTHONPATH=. .venv/bin/python \
  extra/sqtt/beam_emu_vs_hw.py emu --beam-n "$BEAM_N" --workloads "$WORKLOADS"

echo
echo "--- step 3/4: Verify EMU-chosen kernels on real HW ---"
sudo --preserve-env=PATH env DEV=AMD AM_RESET=1 PYTHONPATH=. .venv/bin/python \
  extra/sqtt/beam_emu_vs_hw.py verify --beam-n "$BEAM_N" --workloads "$WORKLOADS"

echo
echo "--- step 4/4: Report ---"
PYTHONPATH=. .venv/bin/python extra/sqtt/beam_emu_vs_hw.py report --workloads "$WORKLOADS"
