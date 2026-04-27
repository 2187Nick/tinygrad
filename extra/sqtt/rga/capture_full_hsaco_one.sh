#!/usr/bin/env bash
# Capture ONE kernel's full unstripped HSACO via tinygrad's compile path.
#
# Sets KEEP_FULL_HSACO=1 so compiler_amd.HIPCompiler stashes the relocatable bytes
# (with .symtab + .note.AMDGPU.metadata) onto the ProfileProgramEvent. Saves a
# self-contained .pkl that extra/sqtt/rga/run_rga.py can hand directly to RGA.
#
# Run with sudo on the SQTT rig (same shell as perf_token_diag.sh / capture_raw_sqtt.py):
#   sudo bash extra/sqtt/rga/capture_full_hsaco_one.sh
#
# Output: extra/sqtt/rga/captures/<ts>/arange64.pkl
set -u
cd "$(dirname "$0")/../../.."

stamp="$(date +%Y%m%d_%H%M%S)"
outdir="extra/sqtt/rga/captures/${stamp}"
mkdir -p "${outdir}"

export DEV=AMD AM_RESET=1 VIZ=-2 SQTT=1 PROFILE=1 KEEP_FULL_HSACO=1
export PYTHONPATH=.
export OUTDIR="${outdir}"

.venv/bin/python - <<'PY'
import os, pickle, pathlib
from tinygrad import Device, Tensor
from tinygrad.device import Compiled

outdir = pathlib.Path(os.environ["OUTDIR"])

# Single trivial kernel — keeps the capture small and deterministic.
Tensor.arange(64).contiguous().realize()
dev = Device[Device.DEFAULT]
dev.synchronize()
dev._at_profile_finalize()

prg_evs = [e for e in Compiled.profile_events if type(e).__name__ == "ProfileProgramEvent"]
sqtt_evs = [e for e in Compiled.profile_events if type(e).__name__ == "ProfileSQTTEvent"]
prg_by_tag = {pe.tag: pe for pe in prg_evs}
target = dev.arch

# Match the lib bytes to whichever SQTT event we have (if any), else fall through.
best_blob = b""
best_pe = None
for ev in sqtt_evs:
  pe = prg_by_tag.get(ev.kern)
  if pe is not None and len(ev.blob) > len(best_blob):
    best_blob, best_pe = bytes(ev.blob), pe

if best_pe is None and prg_evs: best_pe = prg_evs[-1]
if best_pe is None: raise SystemExit("no ProfileProgramEvent captured")

out = {
  "kernel": "arange64",
  "target": target,
  "lib": bytes(best_pe.lib) if best_pe.lib else b"",
  "blob": best_blob,
  "full_hsaco": bytes(best_pe.full_hsaco) if best_pe.full_hsaco else None,
}
out_path = outdir / "arange64.pkl"
with open(out_path, "wb") as f: pickle.dump(out, f)
print(f"saved {out_path}  lib_len={len(out['lib'])}  blob_len={len(out['blob'])}  "
      f"full_hsaco_len={len(out['full_hsaco']) if out['full_hsaco'] else 0}")
PY
