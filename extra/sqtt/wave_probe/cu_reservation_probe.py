#!/usr/bin/env python3
"""CU-reservation before/after probe (mes_notes.md §5.2).

Runs the existing HW_ID probe twice:
  (a) with the driver's default CU_EN mask (whatever SQTT has already set
      for tracing), so we record the "natural" wave placement.
  (b) with `COMPUTE_STATIC_THREAD_MGMT_SE0` forced so that only a single CU
      can receive waves on SE0 (all other SEs zeroed → no waves land there).

For each mode, we launch the same wave sweep (1, 2, 4, 8, 16, 32, 64 waves)
and record the (se, sa, wgp, simd, cu_id) histogram so we can see whether
the "even/odd WGP split" we observe with the default placement collapses
onto a single WGP+CU when reservation is on.

Output one JSON under extra/sqtt/wave_probe/captures/cu_res_<ts>/ with:

    {"default": {<sweep>: [<per-run per-wave placement>, ...]},
     "pinned":  {<sweep>: [...],
                 "mask_value": 0x00010001,
                 "mask_description": "..."},
     "comparison": {...}}

REGISTER BIT LAYOUT — RDNA3 `COMPUTE_STATIC_THREAD_MGMT_SE<n>` (32-bit)
──────────────────────────────────────────────────────────────────────
Per-SE CU enable mask. Each bit disables one CU from receiving new waves
when 0, enables when 1. The layout is:

    bit[15: 0] — SA0 CU enable mask (1 bit per CU, up to 16 CUs/SA)
    bit[31:16] — SA1 CU enable mask (1 bit per CU, up to 16 CUs/SA)

On a Navi31 (7900 XTX), each SE has 2 SAs × 3 WGPs × 2 CUs/WGP = 12 CUs,
so only bits[5:0] (SA0) and bits[21:16] (SA1) are populated; the rest are
don't-care. On smaller parts the high bits of each 16-bit half are unused.

Within one SA, CU numbering pairs up into WGPs:
    bit 0, 1  → WGP0 (CU0 + CU1)
    bit 2, 3  → WGP1 (CU2 + CU3)
    bit 4, 5  → WGP2 (CU4 + CU5)
    ...

To reserve *exactly one CU* (SE0, SA0, WGP0, CU0) you write 0x00000001:
  SA0 mask = 0b000001 (only CU0 enabled)
  SA1 mask = 0b000000 (all SA1 CUs disabled)

This script uses 0x00000001 on SE0 and 0 on all other SEs, which pushes
every wave to one physical CU.

SOURCE TO CONSULT for the authoritative bit layout:
  amdgpu driver: drivers/gpu/drm/amd/amdgpu/gfx_v11_0.c
                 search for `COMPUTE_STATIC_THREAD_MGMT_SE0`.
  Register header: tinygrad/runtime/autogen/amd_gpu.py (regCOMPUTE_STATIC_THREAD_MGMT_SE0..7)
  Existing usage: tinygrad/runtime/ops_amd.py:220 (sqtt_setup_exec) — uses the
                  same field with a per-WGP/CU mask and confirms
                  `cu_bits = 0b11 << (wgp_sel * 2)` pattern.
  RDNA3 ISA Guide: section on SPI Compute Unit programming.
  TODO: a pure WGP-to-bit mapping table would be nice to verify CU_ID ordering
        on Navi31 specifically — consult `gfx_v11_0_0_sq_reg.h` and the
        `regCOMPUTE_STATIC_THREAD_MGMT_SE0` comment header in
        drivers/gpu/drm/amd/amdgpu/gfx_v11_0.c:~2400 (line number depends on
        kernel version).

MECHANISM — HOW WE WRITE THE REGISTER
─────────────────────────────────────
The register is normally programmed by the command processor at dispatch
time; there is no userspace sysfs knob for it. We set it through the same
path the existing SQTT code uses: `sqtt_setup_exec()` in ops_amd.py
already writes `regCOMPUTE_STATIC_THREAD_MGMT_SE{i}` per-dispatch. We
monkey-patch that call so our "pinned" run programs our 1-CU mask. This
lasts only for the duration of the python process — it does not persist.

═══════════════════════════════════════════════════════════════════════════════
  RUN (real GPU, needs sudo):

    sudo DEV=AMD PROFILE=1 SQTT=1 VIZ=-2 \
         PYTHONPATH=. .venv/bin/python extra/sqtt/wave_probe/cu_reservation_probe.py
═══════════════════════════════════════════════════════════════════════════════
"""
import os, sys, json, pathlib
from datetime import datetime

_repo_root = pathlib.Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path: sys.path.insert(0, str(_repo_root))

os.environ.setdefault("DEV", "AMD")
os.environ.setdefault("PROFILE", "1")
os.environ.setdefault("SQTT", "1")
os.environ.setdefault("VIZ", "-2")

# Pinning mask for `COMPUTE_STATIC_THREAD_MGMT_SE0`: enable only SA0/CU0.
PIN_MASK_SE0 = 0x00000001
PIN_MASK_OTHERS = 0x00000000
PIN_MASK_DESCRIPTION = (
  "SE0: SA0 bits[15:0]=0x0001 (CU0 only), SA1 bits[31:16]=0x0000 (all disabled). "
  "SE1..SE7: 0x00000000 (all CUs disabled). Expected: every wave lands on "
  "(se=0, sa=0, wgp=0, cu=0) or the dispatcher stalls indefinitely."
)


def _run_hw_id_sweep(label: str, sweep: list[int], runs_per_sweep: int = 5):
  """Re-use capture_hw_id.run_probe() directly."""
  from extra.sqtt.wave_probe.capture_hw_id import run_probe
  out: dict[str, list] = {}
  for n_waves in sweep:
    print(f"  [{label}] {n_waves} waves × {runs_per_sweep} runs")
    try:
      runs = run_probe(n_waves, runs=runs_per_sweep)
    except Exception as e:
      print(f"    FAILED: {e}")
      out[str(n_waves)] = [{"error": str(e)}]
      continue
    out[str(n_waves)] = runs
  return out


def _summarize_placement(sweep_result: dict) -> dict:
  """Reduce a sweep result to {n_waves: {(se,sa,wgp,simd): count}} for easy diffing."""
  summary: dict[str, dict[str, int]] = {}
  for n_waves_str, runs in sweep_result.items():
    counts: dict[str, int] = {}
    for run in runs:
      if isinstance(run, dict) and "error" in run: continue
      for w in run.get("waves", []):
        if "error" in w: continue
        key = f"se{w['se_id']}_sa{w['sa_id']}_wgp{w['wgp_id']}_simd{w['simd_id']}"
        counts[key] = counts.get(key, 0) + 1
    summary[n_waves_str] = counts
  return summary


def _patch_sqtt_setup_exec_to_pin_cu():
  """Monkey-patch AMDQueue.sqtt_setup_exec so that the CU mask is forced.

  The default code in tinygrad/runtime/ops_amd.py:199-220 writes per-SE masks
  based on SQTT_LIMIT_SE/SQTT_WGP_SEL. We replace that with our single-CU mask.
  Must be called BEFORE the probe kernels are built/dispatched."""
  from tinygrad.runtime import ops_amd

  orig = ops_amd.AMDComputeQueue.sqtt_setup_exec

  def patched(self, prg, global_size):
    # Emit the standard userdata markers (copied from ops_amd.py:200-202)
    from tinygrad.runtime.autogen.amd import sqtt
    from tinygrad.helpers import data64_le, prod
    self.sqtt_userdata(sqtt.struct_rgp_sqtt_marker_pipeline_bind(
      identifier=sqtt.RGP_SQTT_MARKER_IDENTIFIER_BIND_PIPELINE,
      bind_point=(1), api_pso_hash=data64_le(prg.libhash[0])))
    self.sqtt_userdata(sqtt.struct_rgp_sqtt_marker_event(
      has_thread_dims=1, cmd_id=next(prg.dev.sqtt_next_cmd_id)), *global_size)
    # Program the pinning mask on every SE register this chip has (8 on gfx11).
    n_se_regs = 8 if prg.dev.target >= (11,0,0) else 4
    for xcc in range(self.dev.xccs):
      with self.pred_exec(xcc_mask=1 << xcc):
        for i in range(n_se_regs):
          mask = PIN_MASK_SE0 if i == 0 else PIN_MASK_OTHERS
          self.wreg(getattr(self.gc, f'regCOMPUTE_STATIC_THREAD_MGMT_SE{i}'), mask)

  ops_amd.AMDComputeQueue.sqtt_setup_exec = patched
  return orig


def main():
  from tinygrad import Device, Tensor
  arch = Device["AMD"].arch
  assert arch.startswith("gfx11"), f"expected gfx11*, got {arch}"
  (Tensor([1.]) + Tensor([1.])).realize()
  Device[Device.DEFAULT].synchronize()

  stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  out_dir = pathlib.Path(__file__).resolve().parent / "captures" / f"cu_res_{stamp}"
  out_dir.mkdir(parents=True, exist_ok=True)
  print(f"Writing outputs to: {out_dir}")

  sweep = [1, 2, 4, 8, 16, 32, 64]

  print("\n== phase 1: default placement ==")
  default_result = _run_hw_id_sweep("default", sweep)
  default_summary = _summarize_placement(default_result)

  print("\n== phase 2: CU-pinned placement (SE0 / SA0 / CU0 only) ==")
  print(f"  mask={PIN_MASK_SE0:#010x}  {PIN_MASK_DESCRIPTION}")
  _patch_sqtt_setup_exec_to_pin_cu()
  pinned_result = _run_hw_id_sweep("pinned", sweep)
  pinned_summary = _summarize_placement(pinned_result)

  comparison = {}
  for n in sweep:
    k = str(n)
    d_keys = set(default_summary.get(k, {}).keys())
    p_keys = set(pinned_summary.get(k, {}).keys())
    comparison[k] = {
      "default_slots_used": len(d_keys),
      "pinned_slots_used": len(p_keys),
      "pinned_collapsed_to_one": len(p_keys) == 1,
      "slots_exclusive_to_default": sorted(d_keys - p_keys),
      "slots_exclusive_to_pinned": sorted(p_keys - d_keys),
    }

  report = {
    "arch": arch,
    "timestamp": stamp,
    "sweep": sweep,
    "default": {
      "summary": default_summary,
      "raw": default_result,
    },
    "pinned": {
      "mask_value": PIN_MASK_SE0,
      "mask_others": PIN_MASK_OTHERS,
      "mask_description": PIN_MASK_DESCRIPTION,
      "summary": pinned_summary,
      "raw": pinned_result,
    },
    "comparison": comparison,
  }
  out_path = out_dir / "comparison.json"
  with open(out_path, "w") as f:
    json.dump(report, f, indent=2)
  print(f"\nReport saved: {out_path}")

  # Pretty-print the per-sweep comparison
  print("\n== comparison ==")
  print(f"  {'n_waves':>8s}  {'default_slots':>13s}  {'pinned_slots':>12s}  collapsed?")
  for n in sweep:
    k = str(n)
    c = comparison[k]
    print(f"  {n:>8d}  {c['default_slots_used']:>13d}  {c['pinned_slots_used']:>12d}  "
          f"{'YES' if c['pinned_collapsed_to_one'] else 'no'}")


if __name__ == "__main__":
  main()
