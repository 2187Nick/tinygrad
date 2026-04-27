#!/usr/bin/env python3
"""Capture raw SQTT blobs on real gfx1100 — preserves ALL SIMDs.

Existing rigorous_hw_test captures filter WAVESTART/INST packets to a single
(CU, SIMD=0) unit via tinygrad/renderer/amd/sqtt.py:657. That loses the data
we need to answer "how does HW place waves across SIMDs?" This script runs
the same kernels but saves the RAW SQTT blob (pre-decode) so we can
retroactively decode every SIMD via decode_all_simds.py.

Run on real hardware:
  sudo DEV=AMD AM_RESET=1 PROFILE=1 SQTT=1 MICROBENCH=1 VIZ=-2 \\
       PYTHONPATH=. .venv/bin/python extra/sqtt/wave_probe/capture_raw_sqtt.py

Output: extra/sqtt/wave_probe/captures/raw_sqtt_<ts>/<kernel>.pkl
        (each pkl holds {'blob': bytes, 'lib': bytes, 'target': 'gfx1100'})
"""
import os, sys, pickle, pathlib
from datetime import datetime

_repo_root = pathlib.Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path: sys.path.insert(0, str(_repo_root))

os.environ.setdefault("DEV", "AMD")
os.environ.setdefault("PROFILE", "1")
os.environ.setdefault("SQTT", "1")
os.environ.setdefault("VIZ", "-2")
os.environ.setdefault("MICROBENCH", "1")

from tinygrad import Device, Tensor
from tinygrad.device import Compiled, ProfileProgramEvent, ProfileDeviceEvent
from extra.sqtt import rigorous_hw_test as rht

# Subset of kernels where wave-slot placement matters most (many waves / RAW chains).
# Full rht.KERNELS has ~800; we grab 30 with the highest miss counts as representatives.
PRIORITY_KERNELS = [
  "mb_f2_raw_then_vopd", "mb_c4_depctr_chain_n3", "mb_c2_depctr_cmp3_cnd3_vopd",
  "mb_f4_log_chain_n8", "mb_vcmp_spaced_cndmask_nop12", "mb_g2_ds_max_u32_n1",
  "mb_f2_raw_all_banks_n4", "mb_vopd_chain_n4_raw", "mb_vopd_pair_raw_x",
  "mb_vopd_pair_raw_y", "mb_vopd_pair_raw_xy", "mb_g4_s_bfe_u32_n4",
  "mb_g4_s_mul_i32_n4", "mb_g4_s_add_u32_n8", "mb_f4_exp_chain_n8",
  "mb_f4_rcp_chain_n8", "mb_f4_sqrt_chain_n8", "mb_valu_add_n16",
  "mb_f2_raw_indep_interleave_n8", "mb_cndmask_read_vcc_then_sgpr",
  "mb_vcmp_cndmask_k2", "mb_vcmp_cndmask_k8", "mb_trans_raw_with_depctr",
  "mb_lds_store_then_valu_forward", "mb_g2_ds_and_b32_n1",
  "mb_waitcnt_lgkmcnt_null_lds", "mb_snop_15_chain_after_vmcnt_n3",
  "mb_vopd_fmac_mul_n4", "mb_vopd_dualmov_sgpr_chain_n4", "mb_f3_store_pair_then_pair",
]


def capture_raw(name: str, run_fn, max_attempts: int = 10):
  for attempt in range(max_attempts):
    try:
      rht._clear()
      run_fn()
      Device[Device.DEFAULT].synchronize()
      Device[Device.DEFAULT]._at_profile_finalize()
    except Exception as e:
      if attempt == 0: print(f"    {name}[{attempt}]: error {e}")
      continue
    sqtt_events = [e for e in Compiled.profile_events if type(e).__name__ == 'ProfileSQTTEvent']
    program_events = {e.tag: e for e in Compiled.profile_events if type(e).__name__ == 'ProfileProgramEvent'}
    # Pick the first SQTT event whose kern program is present. A kernel may emit multiple SQTT blobs;
    # for the wave-placement question we want the one with the most data.
    best = None
    for ev in sqtt_events:
      if ev.kern in program_events:
        if best is None or len(ev.blob) > len(best[0].blob):
          best = (ev, program_events[ev.kern])
    if best is None or len(best[0].blob) < 64:
      continue
    ev, kev = best
    out = {"blob": bytes(ev.blob), "lib": bytes(kev.lib), "target": "gfx1100",
           "kernel": name, "attempt": attempt, "sqtt_event_count": len(sqtt_events)}
    # KEEP_FULL_HSACO=1 path: include the unstripped HSACO so RGA --livereg/--isa works.
    # Backward-compatible: omitted when None, older readers ignore unknown keys.
    full = getattr(kev, "full_hsaco", None)
    if full is not None: out["full_hsaco"] = bytes(full)
    return out
  return None


def main():
  arch = Device["AMD"].arch
  assert arch.startswith("gfx11"), f"expected gfx11*, got {arch}"
  (Tensor([1.]) + Tensor([1.])).realize()
  Device[Device.DEFAULT].synchronize()

  stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  out_dir = pathlib.Path(__file__).resolve().parent / "captures" / f"raw_sqtt_{stamp}"
  out_dir.mkdir(parents=True, exist_ok=True)

  ok, fail = 0, 0
  for name in PRIORITY_KERNELS:
    if name not in rht.KERNELS:
      print(f"  {name}: NOT IN KERNELS")
      continue
    run_fn, _ = rht.KERNELS[name]
    print(f"  capturing {name} ...")
    res = capture_raw(name, run_fn)
    if res is None:
      print(f"    → FAILED")
      fail += 1
      continue
    out_path = out_dir / f"{name}.pkl"
    with open(out_path, "wb") as f:
      pickle.dump(res, f)
    print(f"    → saved {len(res['blob'])} bytes blob, {len(res['lib'])} bytes lib")
    ok += 1

  print(f"\nDone: {ok} ok, {fail} fail → {out_dir}")


if __name__ == "__main__":
  main()
