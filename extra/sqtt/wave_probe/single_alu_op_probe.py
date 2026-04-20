#!/usr/bin/env python3
"""SQ_DEBUG.single_alu_op probe (mes_notes.md §2.5 + §5 extension).

Intent: run `mb_valu_add_n16` and `mb_f2_raw_indep_interleave_n8` with
SQ_DEBUG.single_alu_op asserted so that the VALU pipe issues one ALU op
per wave-quantum. Comparing the resulting WAVE_ID-per-cycle pattern with
the default capture tells us whether our `simd_arbiter` "same-SIMD peer
cluster" model is operating on real peers or on imagined peers
(mes_notes.md §2.5).

STATUS: NOT CURRENTLY REACHABLE FROM USERSPACE
──────────────────────────────────────────────
The `single_alu_op` bit is inside the **SQ_DEBUG register**. On RDNA3 it
cannot be written from a user-mode dispatch (the register is in the
privileged SH register space). The only sanctioned route is via

    MES_SCH_API_SET_SHADER_DEBUGGER (API id 0x14, pg 45-46 of
    micro_engine_scheduler.pdf)

which is submitted to the MES by the kernel-mode driver (KMD). The amdgpu
mainline driver today only exposes this path through the KFD debugger
ioctl `AMDKFD_IOC_DBG_TRAP` with `KFD_IOC_DBG_TRAP_SET_FLAGS` — and that
API handles `KFD_IOC_DBG_TRAP_FLAG_SINGLE_MEM_OP` / `..._SINGLE_ALU_OP`
only when a user is attached as a ROCm debugger via `rocm-gdb`, not from
plain compute dispatches.

There is no `sysfs`, `debugfs`, or PM4 packet a non-KMD program can emit
to toggle this bit for a normal compute run. Writing SQ_DEBUG via a
user-mode WREG_PKT3 from the dispatch queue is rejected (privileged SH
register).

WHAT A FUTURE KMD PATCH WOULD NEED TO EMIT
──────────────────────────────────────────
The MES specification (pages 45-46) gives the SET_SHADER_DEBUGGER API
frame layout. A future amdgpu patch could populate:

    struct mes_sch_api_set_shader_debugger {
      uint32_t opcode;               // MES_SCH_API_SET_SHADER_DEBUGGER = 0x14
      uint32_t version;
      uint64_t process_context_addr; // per-process context ptr
      struct {
        uint32_t single_memop  : 1;  // SQ_DEBUG.single_memop
        uint32_t single_alu_op : 1;  // SQ_DEBUG.single_alu_op  ← this one
        uint32_t reserved      : 30;
      } flags;
      uint32_t spi_gdbg_per_vmid_cntl;
      uint32_t tcp_watch_cntl[4];
      uint32_t trap_en;
      // ... padding to 64 DW
    };

And write it into the MES API ring (KMD already has this ring for
MES_SCH_API_MAP_PROCESS / MAP_QUEUE; see
`drivers/gpu/drm/amd/amdgpu/mes_v11_0.c: mes_v11_0_submit_pkt_and_poll_completion`).

The matching userspace knob would most naturally be a new ioctl on
`/dev/kfd` or a debugfs write such as

    /sys/kernel/debug/dri/0/amdgpu_sq_single_alu_op

neither of which exists today in upstream amdgpu.

This script exits with a "not possible without KMD patch" message. We
refuse to fake data — if the mechanism ever lands upstream, the probe
body below is ready to flip the flag and run the two target kernels.

FILES TO READ when the KMD patch lands:
  - drivers/gpu/drm/amd/amdgpu/mes_v11_0.c
    (look for a new `mes_v11_0_set_shader_debugger` helper)
  - drivers/gpu/drm/amd/include/mes_v11_api_def.h
    (struct MES_SCH_API_SET_SHADER_DEBUGGER already defined here, used
     today only by the KFD debugger path — extend it or promote to a
     general toggle)
  - drivers/gpu/drm/amd/amdkfd/kfd_debug.c
    (existing consumer via AMDKFD_IOC_DBG_TRAP — this is where a
     userspace-accessible "without attaching as a debugger" path would
     most cleanly live)
  - micro_engine_scheduler.pdf pg 45-46 for the API frame layout.

═══════════════════════════════════════════════════════════════════════════════
  RUN (real GPU, needs sudo, currently exits with status 2):

    sudo DEV=AMD PROFILE=1 SQTT=1 VIZ=-2 MICROBENCH=1 \
         PYTHONPATH=. .venv/bin/python extra/sqtt/wave_probe/single_alu_op_probe.py
═══════════════════════════════════════════════════════════════════════════════
"""
import os, sys, json, pathlib, pickle
from datetime import datetime

_repo_root = pathlib.Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path: sys.path.insert(0, str(_repo_root))

os.environ.setdefault("DEV", "AMD")
os.environ.setdefault("PROFILE", "1")
os.environ.setdefault("SQTT", "1")
os.environ.setdefault("VIZ", "-2")
os.environ.setdefault("MICROBENCH", "1")

TARGET_KERNELS = ["mb_valu_add_n16", "mb_f2_raw_indep_interleave_n8"]


def _probe_single_alu_op_available() -> tuple[bool, str]:
  """Check every known route for toggling SQ_DEBUG.single_alu_op.

  Return (available, explanation). We check:
    1. A hypothetical sysfs/debugfs knob that a KMD patch *might* expose.
    2. The KFD debugger attachment state (would require attaching as
       rocm-gdb, which is not how we capture SQTT).
    3. A tinygrad-local backdoor in ops_amd.py (none exists today).
  """
  candidate_paths = [
    "/sys/kernel/debug/dri/0/amdgpu_sq_single_alu_op",
    "/sys/kernel/debug/dri/0/amdgpu_sq_debug",
    "/sys/module/amdgpu/parameters/sq_single_alu_op",
  ]
  for p in candidate_paths:
    if os.path.exists(p):
      return (True, f"found candidate knob at {p} — KMD patch may have landed")
  # Scan tinygrad runtime for any SQ_DEBUG writer
  try:
    from tinygrad.runtime import ops_amd  # noqa: F401
    import inspect
    src = inspect.getsource(ops_amd)
    if "SQ_DEBUG" in src or "single_alu_op" in src:
      return (True, "tinygrad ops_amd contains an SQ_DEBUG writer — check implementation")
  except Exception:
    pass
  return (False, "no SQ_DEBUG.single_alu_op toggle found in sysfs, debugfs, or tinygrad runtime")


def _run_capture_pair():
  """Scaffold: capture mb_valu_add_n16 and mb_f2_raw_indep_interleave_n8 with
  single_alu_op asserted. Only runs if a toggle mechanism is found."""
  from tinygrad import Device, Tensor
  from extra.sqtt import rigorous_hw_test as rht
  (Tensor([1.]) + Tensor([1.])).realize()
  Device[Device.DEFAULT].synchronize()

  results: dict = {}
  for kname in TARGET_KERNELS:
    if kname not in rht.KERNELS:
      print(f"  {kname}: not in rigorous_hw_test.KERNELS, skip")
      continue
    run_fn, _ = rht.KERNELS[kname]
    rht._clear()
    try:
      run_fn()
      Device[Device.DEFAULT].synchronize()
      Device[Device.DEFAULT]._at_profile_finalize()
    except Exception as e:
      results[kname] = {"error": str(e)}
      continue
    # Snapshot the largest SQTT blob
    from tinygrad.device import Compiled
    sqtt_events = [e for e in Compiled.profile_events if type(e).__name__ == "ProfileSQTTEvent"]
    if not sqtt_events:
      results[kname] = {"error": "no SQTT blob captured"}
      continue
    ev = max(sqtt_events, key=lambda e: len(e.blob))
    results[kname] = {"blob_bytes": len(ev.blob), "exec_tag": ev.exec_tag}
  return results


def main():
  available, explanation = _probe_single_alu_op_available()
  arch = None
  try:
    from tinygrad import Device
    arch = Device["AMD"].arch
  except Exception as e:
    print(f"  (could not determine arch: {e})")

  stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  out_dir = pathlib.Path(__file__).resolve().parent / "captures" / f"single_alu_op_{stamp}"
  out_dir.mkdir(parents=True, exist_ok=True)
  report = {
    "arch": arch,
    "timestamp": stamp,
    "target_kernels": TARGET_KERNELS,
    "mechanism_available": available,
    "mechanism_explanation": explanation,
    "status": "not_runnable" if not available else "scaffold_only",
  }

  if not available:
    print("=" * 72)
    print("  SQ_DEBUG.single_alu_op probe is not runnable on this system.")
    print("=" * 72)
    print()
    print(f"  Mechanism check: {explanation}")
    print()
    print("  Reason: the bit lives in SQ_DEBUG (privileged SH register). The")
    print("  only sanctioned toggle is via the MES API frame")
    print("  MES_SCH_API_SET_SHADER_DEBUGGER (pg 45-46 of")
    print("  micro_engine_scheduler.pdf), which is submitted by the kernel")
    print("  mode driver — not from userspace.")
    print()
    print("  Today upstream amdgpu exposes it *only* via KFD's debugger")
    print("  trap ioctl (AMDKFD_IOC_DBG_TRAP with")
    print("  KFD_IOC_DBG_TRAP_FLAG_SINGLE_ALU_OP), which needs rocm-gdb")
    print("  attached as a debugger — incompatible with the SQTT capture")
    print("  pipeline we run here.")
    print()
    print("  A future KMD patch would need to:")
    print("    1. Construct a MES_SCH_API_SET_SHADER_DEBUGGER frame with")
    print("       flags.single_alu_op = 1.")
    print("    2. Submit it to the MES API ring via")
    print("       mes_v11_0_submit_pkt_and_poll_completion() (see")
    print("       drivers/gpu/drm/amd/amdgpu/mes_v11_0.c).")
    print("    3. Expose a userspace toggle (sysfs, debugfs, or new ioctl)")
    print("       that is orthogonal to the KFD debugger attach path.")
    print()
    print("  See single_alu_op_probe.py's top-of-file docstring for the full")
    print("  struct layout and files to read.")
    report_path = out_dir / "not_runnable.json"
    with open(report_path, "w") as f:
      json.dump(report, f, indent=2)
    print(f"\n  Not-runnable report written: {report_path}")
    sys.exit(2)

  # Would get here only if _probe_single_alu_op_available() flipped to True,
  # i.e., a KMD patch has landed that exposes a knob. In that case we run
  # the two kernels and save their SQTT blobs side-by-side with a matching
  # default-mode capture for diffing.
  print(f"  mechanism detected: {explanation}")
  print("  capturing target kernels with single_alu_op=1 (scaffold — no toggle actually asserted)")
  results = _run_capture_pair()
  report["results"] = results
  report_path = out_dir / "scaffold.json"
  with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
  print(f"  scaffold report written: {report_path}")


if __name__ == "__main__":
  main()
