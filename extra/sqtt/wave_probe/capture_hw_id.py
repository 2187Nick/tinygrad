#!/usr/bin/env python3
"""Probe HW_ID per wave on real gfx1100 hardware.

Launches a custom kernel that reads the HW_ID register (SIMD_ID, CU_ID,
WAVE_ID, WGP_ID, SA_ID, SE_ID) via s_getreg_b32 and stores it to a per-wave
slot in memory. Sweeps workgroup count / wave count so we can see how the
SPI dispatcher actually places waves onto SIMDs.

Run on real hardware:
  DEV=AMD PYTHONPATH=. .venv/bin/python extra/sqtt/wave_probe/capture_hw_id.py

Output: extra/sqtt/wave_probe/captures/hw_id_<ts>.json

HW_ID register layout (RDNA3 ISA §3.5, hwRegId=4):
    [3:0]  WAVE_ID        wave slot within SIMD (0-15)
    [5:4]  SIMD_ID        SIMD within WGP (0-3)
    [7:6]  PIPE_ID        always 0 for CS
    [11:8] CU_ID          CU within WGP (0-1 on RDNA3 WGP)
    [14:12] SH_ID / SA_ID shader array id
    [18:15] SE_ID         shader engine id
    [23:20] TG_ID         workgroup id within CU
    [27:24] VM_ID         virtual memory id
    [31:28] STATE_ID      always 0 for CS
"""
import os, sys, json, pathlib
from datetime import datetime

_repo_root = pathlib.Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path: sys.path.insert(0, str(_repo_root))

os.environ.setdefault("DEV", "AMD")

from tinygrad import Tensor, Device, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates
from tinygrad.runtime.autogen.amd.rdna3.ins import (
  s_load_b64, s_waitcnt_lgkmcnt, s_waitcnt_vmcnt, s_getreg_b32, v_mov_b32_e32,
  v_lshlrev_b32_e32, global_store_b32, s_endpgm, NULL,
)
from tinygrad.renderer.amd.dsl import s, v

# HW_ID encoded simm16: (size-1)<<11 | offset<<6 | hwRegId
# Full HW_ID: hwRegId=4, offset=0, size=32 → simm16 = (32-1)<<11 | 4 = 0xF804
HWREG_HW_ID_FULL = (32 - 1) << 11 | 0 << 6 | 4


def custom_probe_hw_id(A):
  """Kernel: every wave stores its 32-bit HW_ID to A[gidx0]."""
  A = A.flatten()
  assert A.dtype.base == dtypes.uint32
  threads = UOp.special(A.size, "lidx0")
  insts = [
    s_load_b64(s[0:1], s[0:1], soffset=NULL),           # buffer ptr
    s_waitcnt_lgkmcnt(sdst=NULL, simm16=0),
    s_getreg_b32(s[2], simm16=HWREG_HW_ID_FULL),         # read full HW_ID → s2
    v_mov_b32_e32(v[1], s[2]),                           # broadcast to v1
    v_lshlrev_b32_e32(v[0], 2, v[0]),                    # lane*4 byte offset
    global_store_b32(addr=v[0], data=v[1], saddr=s[0:1]),
    s_endpgm(),
  ]
  sink = UOp.sink(A.base, threads,
                  arg=KernelInfo("probe_hw_id", estimates=Estimates(ops=A.size, mem=A.size*4)))
  return UOp(Ops.PROGRAM,
             src=(sink, UOp(Ops.DEVICE, arg="AMD"),
                  UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))


def decode_hw_id(v: int) -> dict:
  return {
    "wave_id":  v & 0xF,
    "simd_id": (v >> 4) & 0x3,
    "pipe_id": (v >> 6) & 0x3,
    "cu_id":   (v >> 8) & 0xF,
    "sh_id":   (v >> 12) & 0x7,
    "se_id":   (v >> 15) & 0xF,
    "tg_id":   (v >> 20) & 0xF,
    "vm_id":   (v >> 24) & 0xF,
    "raw_hex": f"0x{v:08x}",
  }


def run_probe(n_threads: int, runs: int = 5) -> list[dict]:
  """Launch probe kernel N times with `n_threads` lanes. Returns list of per-run decoded HW_IDs per wave."""
  import functools
  results = []
  for r in range(runs):
    buf = Tensor.zeros(n_threads, dtype=dtypes.uint32).contiguous().realize()
    buf = Tensor.custom_kernel(buf, fxn=custom_probe_hw_id)[0]
    buf.realize()
    Device[Device.DEFAULT].synchronize()
    arr = buf.numpy().tolist()
    # Each wave has 64 lanes; all lanes of a wave see the same HW_ID.
    per_wave = []
    for wave_start in range(0, n_threads, 64):
      wave_ids = set(arr[wave_start:wave_start + 64])
      if len(wave_ids) != 1:
        per_wave.append({"error": f"wave at lane {wave_start} has {len(wave_ids)} distinct HW_IDs: {wave_ids}"})
        continue
      per_wave.append(decode_hw_id(wave_ids.pop()))
    results.append({"run": r, "n_threads": n_threads, "waves": per_wave})
  return results


def main():
  arch = Device["AMD"].arch
  assert arch.startswith("gfx11"), f"expected gfx11*, got {arch} — this probe uses RDNA3 encoding"
  # Warmup
  (Tensor([1.]) + Tensor([1.])).realize()
  Device[Device.DEFAULT].synchronize()

  out_dir = pathlib.Path(__file__).resolve().parent / "captures"
  out_dir.mkdir(parents=True, exist_ok=True)
  stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  out_path = out_dir / f"hw_id_{stamp}.json"

  # Sweep: 1 wave, 2 waves, 4 waves, 8 waves, 16 waves, 32 waves, 64 waves (1 wave = 64 threads)
  # 64 waves should force spillover across CUs / WGPs.
  sweep = [1, 2, 4, 8, 16, 32, 64]
  report = {"arch": arch, "timestamp": stamp, "sweeps": []}

  for n_waves in sweep:
    n_threads = n_waves * 64
    print(f"=== sweep: {n_waves} waves ({n_threads} threads) ===")
    try:
      runs = run_probe(n_threads, runs=10)
    except Exception as e:
      print(f"  FAILED: {e}")
      report["sweeps"].append({"n_waves": n_waves, "error": str(e)})
      continue
    # Summarize placement across runs
    placements = {}    # (cu, simd) -> count
    per_wave_simd = {} # wave_idx -> set of SIMDs observed across runs
    for run in runs:
      for widx, w in enumerate(run["waves"]):
        if "error" in w: continue
        key = (w["cu_id"], w["simd_id"])
        placements[key] = placements.get(key, 0) + 1
        per_wave_simd.setdefault(widx, set()).add(w["simd_id"])
    print(f"  placements (cu,simd): {sorted(placements.items())}")
    print(f"  per-wave SIMD stability: "
          f"{sum(1 for s in per_wave_simd.values() if len(s) == 1)} stable / {len(per_wave_simd)} total")
    report["sweeps"].append({
      "n_waves": n_waves,
      "n_threads": n_threads,
      "runs": runs,
      "placements_summary": {f"cu{k[0]}_simd{k[1]}": v for k, v in placements.items()},
      "wave_simd_stable_count": sum(1 for s in per_wave_simd.values() if len(s) == 1),
    })

  with open(out_path, "w") as f:
    json.dump(report, f, indent=2)
  print(f"\nReport saved: {out_path}")


if __name__ == "__main__":
  main()
