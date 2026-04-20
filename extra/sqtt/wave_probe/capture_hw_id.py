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
# RDNA3 HW_ID1: hwRegId=23, offset=0, size=32 → simm16 = (32-1)<<11 | 23
# RDNA3 HW_ID1 layout: [3:0]WAVE_ID [5:4]SIMD_ID [9:6]WGP_ID [12:10]SA_ID [14:13]SE_ID
# RDNA3 HW_ID2 (hwRegId=24): [3:0]QUEUE_ID [7:4]PIPE_ID [11:8]ME_ID [15:12]STATE_ID
#                            [19:16]WG_ID [23:20]VM_ID
HWREG_HW_ID1_FULL = (32 - 1) << 11 | 0 << 6 | 23
HWREG_HW_ID2_FULL = (32 - 1) << 11 | 0 << 6 | 24


def custom_probe_hw_id(A):
  """Kernel: every wave stores its HW_ID1 to A[gidx0]. HW_ID1 has {WAVE_ID, SIMD_ID, WGP_ID, SA_ID, SE_ID}."""
  A = A.flatten()
  assert A.dtype.base == dtypes.uint32
  threads = UOp.special(A.size, "lidx0")
  insts = [
    s_load_b64(s[0:1], s[0:1], soffset=NULL),           # buffer ptr
    s_waitcnt_lgkmcnt(sdst=NULL, simm16=0),
    s_getreg_b32(s[2], simm16=HWREG_HW_ID1_FULL),        # HW_ID1 → s2
    v_mov_b32_e32(v[1], s[2]),                           # broadcast HW_ID1
    v_lshlrev_b32_e32(v[0], 2, v[0]),                    # lane*4 byte offset
    global_store_b32(addr=v[0], data=v[1], saddr=s[0:1]),
    s_endpgm(),
  ]
  sink = UOp.sink(A.base, threads,
                  arg=KernelInfo("probe_hw_id", estimates=Estimates(ops=A.size, mem=A.size*4)))
  return UOp(Ops.PROGRAM,
             src=(sink, UOp(Ops.DEVICE, arg="AMD"),
                  UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))


def decode_hw_id(v1: int, v2: int) -> dict:
  """Decode RDNA3 HW_ID1 (v1) and HW_ID2 (v2)."""
  return {
    # HW_ID1
    "wave_id":  v1 & 0xF,
    "simd_id": (v1 >> 4) & 0x3,
    "wgp_id":  (v1 >> 6) & 0xF,
    "sa_id":   (v1 >> 10) & 0x7,
    "se_id":   (v1 >> 13) & 0x3,
    "dp_rate": (v1 >> 15) & 0x7,
    # HW_ID2
    "queue_id": v2 & 0xF,
    "pipe_id": (v2 >> 4) & 0xF,
    "me_id":   (v2 >> 8) & 0xF,
    "state_id":(v2 >> 12) & 0xF,
    "wg_id":   (v2 >> 16) & 0xF,
    "vm_id":   (v2 >> 20) & 0xF,
    "raw1_hex": f"0x{v1:08x}",
    "raw2_hex": f"0x{v2:08x}",
  }


def run_probe(n_waves: int, runs: int = 5, lanes_per_wave: int = 32) -> list[dict]:
  """Launch probe kernel N times with `n_waves` waves. Each thread writes HW_ID1 (1 uint32)."""
  n_threads = n_waves * lanes_per_wave
  results = []
  for r in range(runs):
    buf = Tensor.zeros(n_threads, dtype=dtypes.uint32).contiguous().realize()
    buf = Tensor.custom_kernel(buf, fxn=custom_probe_hw_id)[0]
    buf.realize()
    Device[Device.DEFAULT].synchronize()
    arr = buf.numpy().tolist()
    per_wave = []
    for wave_idx in range(n_waves):
      base = wave_idx * lanes_per_wave
      hwid1s = set(arr[base+i] for i in range(lanes_per_wave))
      if len(hwid1s) != 1:
        per_wave.append({"error": f"wave {wave_idx} non-uniform hwid1={[hex(v) for v in sorted(hwid1s)[:4]]}"})
        continue
      per_wave.append(decode_hw_id(hwid1s.pop(), 0))
    results.append({"run": r, "n_threads": n_threads, "n_waves": n_waves, "waves": per_wave})
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
      runs = run_probe(n_waves, runs=10)
    except Exception as e:
      print(f"  FAILED: {e}")
      report["sweeps"].append({"n_waves": n_waves, "error": str(e)})
      continue
    # Summarize placement across runs: key on (se, sa, wgp, simd) which uniquely
    # identifies a SIMD across the whole chip.
    placements = {}
    per_wave_simd = {}
    for run in runs:
      for widx, w in enumerate(run["waves"]):
        if "error" in w: continue
        key = (w["se_id"], w["sa_id"], w["wgp_id"], w["simd_id"])
        placements[key] = placements.get(key, 0) + 1
        per_wave_simd.setdefault(widx, set()).add(w["simd_id"])
    print(f"  placements (se,sa,wgp,simd): {sorted(placements.items())}")
    print(f"  per-wave SIMD stability: "
          f"{sum(1 for s in per_wave_simd.values() if len(s) == 1)} stable / {len(per_wave_simd)} total")
    # Show first run's wave→slot mapping
    if runs and runs[0]["waves"] and "error" not in runs[0]["waves"][0]:
      r0_map = [(i, (w["se_id"], w["sa_id"], w["wgp_id"], w["simd_id"], w["wave_id"]))
                for i, w in enumerate(runs[0]["waves"]) if "error" not in w]
      print(f"  run0 wave→(se,sa,wgp,simd,wave_slot): {r0_map[:8]}{' ...' if len(r0_map) > 8 else ''}")
    report["sweeps"].append({
      "n_waves": n_waves,
      "n_threads": n_threads,
      "runs": runs,
      "placements_summary": {f"se{k[0]}_sa{k[1]}_wgp{k[2]}_simd{k[3]}": v for k, v in placements.items()},
      "wave_simd_stable_count": sum(1 for s in per_wave_simd.values() if len(s) == 1),
    })

  with open(out_path, "w") as f:
    json.dump(report, f, indent=2)
  print(f"\nReport saved: {out_path}")


if __name__ == "__main__":
  main()
