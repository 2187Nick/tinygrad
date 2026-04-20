#!/usr/bin/env python3
"""Capture per-(SE, SA, WGP) performance counters alongside SQTT.

The original "SPM" idea was streaming per-cycle counters. tinygrad already
has a close-enough mechanism: PMC (Performance Monitor Counters) gives
per-kernel aggregate counts, broken down per (XCC, INST, SE, SA, WGP) for
the SQ block on gfx11. That's exactly the per-CU wave-distribution picture
we need to answer "where did the waves go?" without the streaming overhead.

For each priority microbench this captures:
  - SQ_WAVES               # total waves launched, per WGP  <- placement answer
  - SQ_BUSY_CYCLES         # how long each WGP was active
  - SQ_INSTS_VALU          # VALU instructions retired, per WGP
  - SQ_INSTS_SALU          # SALU instructions retired, per WGP
  - SQ_INSTS_VMEM          # VMEM instructions retired, per WGP
  - SQ_INSTS_LDS           # LDS instructions retired, per WGP
  - SQ_WAIT_INST_LDS       # cycles stalled on LDS wait, per WGP
  - GRBM_GUI_ACTIVE        # total GPU active cycles (scalar)
  - GL2C_HIT / GL2C_MISS   # L2 cache hit/miss (32 instances)

The WGP-level breakdown of SQ_WAVES directly tells us the wave allocator's
policy: if all waves go to WGP0, the SPI packs tight; if they spread evenly,
the SPI load-balances.

Run on real hardware:
  sudo DEV=AMD AM_RESET=1 PROFILE=1 PMC=1 SQTT=1 MICROBENCH=1 VIZ=-2 \\
       PMC_COUNTERS=SQ_WAVES,SQ_BUSY_CYCLES,SQ_INSTS_VALU,SQ_INSTS_SALU,SQ_INSTS_VMEM,SQ_INSTS_LDS,GRBM_GUI_ACTIVE,GL2C_HIT,GL2C_MISS \\
       PYTHONPATH=. .venv/bin/python extra/sqtt/wave_probe/capture_spm.py

Output: extra/sqtt/wave_probe/captures/pmc_<ts>/<kernel>.json
        (decoded counter table, one file per kernel)
"""
import os, sys, json, itertools, pathlib
from collections import defaultdict
from datetime import datetime

_repo_root = pathlib.Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path: sys.path.insert(0, str(_repo_root))

os.environ.setdefault("DEV", "AMD")
os.environ.setdefault("PROFILE", "1")
os.environ.setdefault("PMC", "1")
os.environ.setdefault("SQTT", "1")
os.environ.setdefault("VIZ", "-2")
os.environ.setdefault("MICROBENCH", "1")
# Wave-placement focused counter set. PMC_COUNTERS overrides the default in ops_amd.py.
os.environ.setdefault("PMC_COUNTERS",
  "SQ_WAVES,SQ_BUSY_CYCLES,SQ_INSTS_VALU,SQ_INSTS_SALU,SQ_INSTS_VMEM,"
  "SQ_INSTS_LDS,GRBM_GUI_ACTIVE,GL2C_HIT,GL2C_MISS")

from tinygrad import Device, Tensor
from tinygrad.device import Compiled
from extra.sqtt import rigorous_hw_test as rht

# Same priority set as capture_raw_sqtt.py so PMC and SQTT line up per kernel.
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


def decode_pmc_event(ev) -> dict:
  """Walk a ProfilePMCEvent, return per-counter per-(xcc,inst,se,sa,wgp) values.

  Mirrors tinygrad/viz/serve.py:unpack_pmc but returns JSON-friendly structure.
  """
  import struct
  view = memoryview(ev.blob).cast('Q')
  ptr = 0
  out: dict[str, dict] = {}
  for s in ev.sched:
    rows = []
    for xcc, inst, se, sa in itertools.product(range(s.xcc), range(s.inst), range(s.se), range(s.sa)):
      for wgp in range(s.wgp):
        val = int(view[ptr]); ptr += 1
        rows.append({"xcc": xcc, "inst": inst, "se": se, "sa": sa, "wgp": wgp, "value": val})
    out[s.name] = {"block": s.block, "rows": rows, "total": sum(r["value"] for r in rows)}
  return out


def wave_distribution(decoded: dict) -> dict:
  """Summarise SQ_WAVES per-(SE, SA, WGP) — primary wave-placement signal."""
  waves = decoded.get("SQ_WAVES")
  if not waves: return {}
  bucket: dict[str, int] = defaultdict(int)
  for r in waves["rows"]:
    bucket[f"se{r['se']}_sa{r['sa']}_wgp{r['wgp']}"] += r["value"]
  return {"per_wgp": dict(bucket), "total_waves": waves["total"],
          "unique_wgps_used": sum(1 for v in bucket.values() if v > 0)}


def balance_scores(decoded: dict) -> dict:
  """Per-counter: how lopsided is the distribution? 1.0 = perfectly balanced."""
  result = {}
  for name, data in decoded.items():
    vals = [r["value"] for r in data["rows"] if r["value"] > 0]
    if not vals: continue
    mean = sum(vals) / len(vals)
    if mean == 0: continue
    max_v = max(vals)
    result[name] = {
      "nonzero_units": len(vals),
      "total_units": len(data["rows"]),
      "balance_ratio": mean / max_v,  # 1.0 = even; 0.0 = all on one unit
      "mean_per_nonzero": mean,
      "max": max_v,
    }
  return result


def capture_with_pmc(name: str, run_fn, max_attempts: int = 10) -> dict | None:
  for attempt in range(max_attempts):
    try:
      rht._clear()
      run_fn()
      Device[Device.DEFAULT].synchronize()
      Device[Device.DEFAULT]._at_profile_finalize()
    except Exception as e:
      if attempt == 0: print(f"    {name}[{attempt}]: error {e}")
      continue
    pmc_events = [e for e in Compiled.profile_events if type(e).__name__ == 'ProfilePMCEvent']
    program_events = {e.tag: e for e in Compiled.profile_events if type(e).__name__ == 'ProfileProgramEvent'}
    # Match PMC event to the kernel's program event via kern tag.
    for ev in pmc_events:
      if ev.kern in program_events:
        kname = program_events[ev.kern].name
        if name in kname:
          decoded = decode_pmc_event(ev)
          return {
            "kernel": name,
            "kernel_full_name": kname,
            "attempt": attempt,
            "counters": decoded,
            "wave_distribution": wave_distribution(decoded),
            "balance": balance_scores(decoded),
          }
    continue
  return None


def main():
  arch = Device["AMD"].arch
  assert arch.startswith("gfx11"), f"expected gfx11*, got {arch}"
  (Tensor([1.]) + Tensor([1.])).realize()
  Device[Device.DEFAULT].synchronize()

  stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  out_dir = pathlib.Path(__file__).resolve().parent / "captures" / f"pmc_{stamp}"
  out_dir.mkdir(parents=True, exist_ok=True)

  ok, fail = 0, 0
  all_waves: dict[str, dict] = {}
  for name in PRIORITY_KERNELS:
    if name not in rht.KERNELS:
      print(f"  {name}: NOT IN KERNELS")
      continue
    run_fn, _ = rht.KERNELS[name]
    print(f"  capturing {name} ...")
    res = capture_with_pmc(name, run_fn)
    if res is None:
      print(f"    → FAILED")
      fail += 1
      continue
    with open(out_dir / f"{name}.json", "w") as f:
      json.dump(res, f, indent=2)
    wd = res["wave_distribution"]
    print(f"    → {wd.get('total_waves', 0)} waves across {wd.get('unique_wgps_used', 0)} WGPs: "
          f"{sorted(wd.get('per_wgp', {}).items(), key=lambda kv: -kv[1])[:4]}")
    all_waves[name] = wd
    ok += 1

  # Top-level summary: what does wave placement look like across all kernels?
  summary = {
    "arch": arch, "timestamp": stamp, "counters_used": os.environ["PMC_COUNTERS"].split(","),
    "kernel_wave_distributions": all_waves,
    "analysis": _analyze(all_waves),
  }
  with open(out_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

  print(f"\nDone: {ok} ok, {fail} fail → {out_dir}")
  print(f"\n=== Wave-placement summary ===")
  for line in summary["analysis"]["narrative"]:
    print(f"  {line}")


def _analyze(all_waves: dict) -> dict:
  """Produce a human-readable narrative about placement patterns."""
  if not all_waves:
    return {"narrative": ["no data"]}

  narrative = []
  wgp_counts = [w.get("unique_wgps_used", 0) for w in all_waves.values()]
  if max(wgp_counts) == 1:
    narrative.append("ALL kernels ran on a SINGLE WGP — wave_probe launch size is small.")
    narrative.append("→ SPI packs tight; arbiter's per-SIMD modelling becomes the whole story.")
  else:
    narrative.append(f"WGP usage range: {min(wgp_counts)}..{max(wgp_counts)} WGPs per kernel.")
    # Are waves ever spread across SEs?
    ses_used = set()
    for w in all_waves.values():
      for key in w.get("per_wgp", {}):
        ses_used.add(key.split("_")[0])
    narrative.append(f"Shader engines touched across all kernels: {sorted(ses_used)}.")

  # Which kernels spread the widest?
  widest = sorted(all_waves.items(), key=lambda kv: -kv[1].get("unique_wgps_used", 0))[:3]
  narrative.append(f"Most-spread kernels: {[(k, v.get('unique_wgps_used', 0)) for k, v in widest]}")

  return {"narrative": narrative}


if __name__ == "__main__":
  main()
