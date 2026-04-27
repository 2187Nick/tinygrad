#!/usr/bin/env bash
# One-kernel diagnostic to confirm SQTT_INCLUDE_PERF token_exclude write.
# Reads the [SQTT-DEBUG] line emitted from tinygrad/runtime/ops_amd.py.
set -u
cd "$(dirname "$0")/../../.."

export DEV=AMD AM_RESET=1 VIZ=-2 SQTT=1 PROFILE=1 MICROBENCH=1 SQTT_INCLUDE_PERF=1
export PYTHONPATH=.

.venv/bin/python - <<'PY' 2>&1 | grep -E 'SQTT-DEBUG|---DONE---|Traceback|rror' | head -20
from tinygrad import Device, Tensor
from extra.sqtt import rigorous_hw_test as rht
rht._clear()
Tensor.arange(64).contiguous().realize()
Device[Device.DEFAULT].synchronize()
Device[Device.DEFAULT]._at_profile_finalize()
print("---DONE---")
PY
