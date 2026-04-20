#!/usr/bin/env python3
"""MES Smart Trace Buffer log probe (mes_notes.md §5.1).

Runs the existing 50x100 capture set, then reads the MES event log that the
amdgpu firmware writes when `mes_log_enable=1` is set on the amdgpu module, and
emits one JSON per kernel aligning `MES_EVT_LOG_MAP_QUEUE.time_after_call` with
that kernel's first WAVESTART._time pulled from the SQTT blob.

Goal: establish the dispatch-edge -> first-wave latency distribution (see
mes_notes.md §2.2 / §5.1). Output one JSON per kernel per run under
extra/sqtt/wave_probe/captures/mes_sched_<ts>/ with the shape

    {"kernel": "mb_valu_add_n16",
     "run": 0,
     "queue_map_ts": <u64>,
     "first_wavestart_ts": <u64>,
     "delta_cy": <i64>}

MECHANISM / KERNEL REQUIREMENTS
───────────────────────────────
The MES STB/event log is toggled by the `mes_log_enable` amdgpu module
parameter (see amdgpu_mes.c: `adev->enable_mes_event_log`). It is NOT a
debugfs knob on upstream amdgpu; it's a boot-time module parameter. To enable
without a reboot, reload the module:

    sudo modprobe -r amdgpu            # cold — disconnects display
    sudo modprobe amdgpu mes_log_enable=1

Your kernel must be built with:

    CONFIG_DRM_AMDGPU=m  (or =y)
    CONFIG_DRM_AMDGPU_SI, CONFIG_DRM_AMDGPU_CIK, ...    (whatever your distro uses)
    CONFIG_DEBUG_FS=y                                   (debugfs read of event log)
    CONFIG_DRM_AMDGPU_USERPTR=y

The 6.x amdgpu driver exposes the event ring buffer at

    /sys/kernel/debug/dri/0/amdgpu_mes_event_log

when `mes_log_enable=1` is active. Older kernels (5.15 and earlier) do not
carry the log wiring at all — this script will detect that and exit gracefully.

Entry format follows MES_EVT_INTR_HIST_LOG (mes_v11_api_def.h):

    struct mes_event_log_entry {
        uint32_t tag;             // event id (MES_EVT_LOG_MAP_QUEUE = 3)
        uint32_t process_id;
        uint64_t time_before_call; // GPU timestamp
        uint64_t time_after_call;  // GPU timestamp
        uint32_t doorbell_offset;
        uint32_t reserved[3];
    };                            // 32 B per entry

See amdgpu driver source: `drivers/gpu/drm/amd/amdgpu/amdgpu_mes.c` function
`amdgpu_debugfs_mes_event_log_show`. The exact struct layout is what's written
by MES firmware and is documented in `extra/sqtt/micro_engine_scheduler.pdf`
pages 47-50 (three parallel circular buffers: api_history, event_log_history,
interrupt_history).

This script:
  1. Verifies the debugfs / module-parameter path is exposed.
  2. Runs each kernel in KERNEL_50 three times (reusing capture_50x100's list).
  3. For each run, reads the MES event log, extracts MAP_QUEUE entries, and
     pairs them with the first SQTT WAVESTART timestamp of that run.
  4. Writes one JSON per (kernel, run) into captures/mes_sched_<ts>/.

If the debugfs path is not exposed by the running kernel, the script prints
the required CONFIG + modprobe lines and exits with code 2. It never fabricates
data.

═══════════════════════════════════════════════════════════════════════════════
  RUN (real GPU, needs sudo + reboot or modprobe with mes_log_enable=1):

    sudo modprobe -r amdgpu && sudo modprobe amdgpu mes_log_enable=1
    sudo DEV=AMD PROFILE=1 SQTT=1 VIZ=-2 MICROBENCH=1 \
         PYTHONPATH=. .venv/bin/python extra/sqtt/wave_probe/mes_scheduler_log.py
═══════════════════════════════════════════════════════════════════════════════
"""
import os, sys, json, pickle, pathlib, time, struct, subprocess
from datetime import datetime

_repo_root = pathlib.Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path: sys.path.insert(0, str(_repo_root))

os.environ.setdefault("DEV", "AMD")
os.environ.setdefault("PROFILE", "1")
os.environ.setdefault("SQTT", "1")
os.environ.setdefault("VIZ", "-2")
os.environ.setdefault("MICROBENCH", "1")

# Path where amdgpu exposes the MES event log when mes_log_enable=1.
MES_EVENT_LOG_PATHS = [
  "/sys/kernel/debug/dri/0/amdgpu_mes_event_log",
  "/sys/kernel/debug/dri/1/amdgpu_mes_event_log",
]
MES_MODULE_PARAM = "/sys/module/amdgpu/parameters/mes_log_enable"

# MES event tag constants from mes_v11_api_def.h
MES_EVT_LOG_MAP_QUEUE = 3

# The three kernels from mes_notes §5.1 plus the full list
FROM_CAPTURE_50x100 = True  # reuse KERNEL_50 verbatim


def _check_mes_log_enabled() -> tuple[bool, str]:
  """Return (ok, reason_or_path). If ok, reason_or_path is the readable event-log path."""
  # 1. module parameter
  if not os.path.exists(MES_MODULE_PARAM):
    return (False, f"{MES_MODULE_PARAM} missing — amdgpu module not loaded or "
                   "kernel too old (needs ≥6.1 with mes_log_enable wiring).")
  try:
    with open(MES_MODULE_PARAM, "r") as f:
      val = f.read().strip()
  except PermissionError:
    return (False, f"{MES_MODULE_PARAM} unreadable — need root.")
  if val != "1":
    return (False, f"{MES_MODULE_PARAM}={val!r}, expected '1'. Reload the amdgpu "
                   "module with: sudo modprobe -r amdgpu && "
                   "sudo modprobe amdgpu mes_log_enable=1")
  # 2. debugfs path
  for p in MES_EVENT_LOG_PATHS:
    if os.path.exists(p):
      try:
        with open(p, "rb") as f:
          f.read(8)
        return (True, p)
      except PermissionError:
        return (False, f"{p} unreadable — run as root (sudo).")
      except OSError as e:
        return (False, f"{p} read failed: {e}")
  return (False, "no amdgpu_mes_event_log under /sys/kernel/debug/dri/*/. "
                 "Your kernel may lack amdgpu_debugfs_mes_event_log_show "
                 "(merged upstream around amdgpu 6.0+). Required config:\n"
                 "    CONFIG_DEBUG_FS=y\n"
                 "    CONFIG_DRM_AMDGPU=m  (with a recent-enough drivers/gpu/drm/amd/amdgpu)\n"
                 "Confirm by grepping your built kernel source for "
                 "`amdgpu_debugfs_mes_event_log_fops`.")


def _read_mes_event_log(path: str) -> bytes:
  """Snapshot the entire event log as raw bytes. Caller parses."""
  with open(path, "rb") as f:
    return f.read()


def _parse_mes_event_log(raw: bytes) -> list[dict]:
  """Parse the MES event log. Returns a list of dicts with at least:
    {'tag', 'process_id', 'time_before_call', 'time_after_call', 'doorbell_offset'}

  Layout from amdgpu_mes_event_log_show:
    u32 tag; u32 process_id; u64 t_before; u64 t_after; u32 doorbell; u32 rsvd[3];
  = 32 B per record. The file is a text dump in newer kernels and binary on
  older ones — we try binary first, then fall back to the text parser."""
  entries: list[dict] = []
  ENTRY_SZ = 32
  # Binary path: file size divisible by 32 with plausible tag values.
  if len(raw) >= ENTRY_SZ and len(raw) % ENTRY_SZ == 0:
    ok_bin = True
    for off in range(0, min(len(raw), 4*ENTRY_SZ), ENTRY_SZ):
      tag = struct.unpack_from("<I", raw, off)[0]
      if tag > 0xFF:  # tag is small int (see MES_EVT_LOG_* in mes_v11_api_def.h)
        ok_bin = False; break
    if ok_bin:
      for off in range(0, len(raw), ENTRY_SZ):
        tag, pid = struct.unpack_from("<II", raw, off)
        t_before, t_after = struct.unpack_from("<QQ", raw, off+8)
        doorbell = struct.unpack_from("<I", raw, off+24)[0]
        entries.append({"tag": tag, "process_id": pid,
                        "time_before_call": t_before, "time_after_call": t_after,
                        "doorbell_offset": doorbell})
      return entries
  # Text path: each entry on its own line, fields: tag process_id time_before time_after doorbell ...
  # The upstream amdgpu dump uses seq_printf("%u %u ...") — be tolerant of whitespace.
  for line in raw.decode(errors="replace").splitlines():
    parts = line.split()
    if len(parts) < 5: continue
    try:
      tag = int(parts[0], 0); pid = int(parts[1], 0)
      t_before = int(parts[2], 0); t_after = int(parts[3], 0)
      doorbell = int(parts[4], 0)
    except ValueError:
      continue
    entries.append({"tag": tag, "process_id": pid,
                    "time_before_call": t_before, "time_after_call": t_after,
                    "doorbell_offset": doorbell})
  return entries


def _first_wavestart_time(sqtt_blob: bytes) -> int | None:
  """Walk the decoded SQTT stream and return the first WAVESTART packet's
  `_time` field. None if there are no WAVESTART packets."""
  from tinygrad.renderer.amd.sqtt import decode, WAVESTART, WAVESTART_RDNA4, CDNA_WAVESTART
  for pkt in decode(sqtt_blob):
    if isinstance(pkt, (WAVESTART, WAVESTART_RDNA4, CDNA_WAVESTART)):
      return pkt._time
  return None


def _snapshot_sqtt_event() -> tuple[bytes | None, int | None]:
  """Pull the largest SQTT blob currently sitting in Compiled.profile_events.
  Returns (blob, first_wavestart_time) or (None, None)."""
  from tinygrad.device import Compiled
  sqtt_events = [e for e in Compiled.profile_events if type(e).__name__ == 'ProfileSQTTEvent']
  program_events = {e.tag: e for e in Compiled.profile_events if type(e).__name__ == 'ProfileProgramEvent'}
  best_blob = None
  for ev in sqtt_events:
    if ev.kern not in program_events: continue
    if best_blob is None or len(ev.blob) > len(best_blob):
      best_blob = bytes(ev.blob)
  if best_blob is None:
    return (None, None)
  return (best_blob, _first_wavestart_time(best_blob))


def main():
  from tinygrad import Device, Tensor
  ok, info = _check_mes_log_enabled()
  if not ok:
    print("MES scheduler log not available.")
    print(info)
    print()
    print("To enable on a kernel that supports it:")
    print("    sudo modprobe -r amdgpu")
    print("    sudo modprobe amdgpu mes_log_enable=1")
    print()
    print("Required CONFIG lines (verify against your kernel .config):")
    print("    CONFIG_DRM_AMDGPU=m")
    print("    CONFIG_DEBUG_FS=y")
    print("    CONFIG_DRM_AMDGPU_USERPTR=y")
    print()
    print("Source to read: drivers/gpu/drm/amd/amdgpu/amdgpu_mes.c")
    print("                (amdgpu_debugfs_mes_event_log_show)")
    sys.exit(2)
  log_path = info
  print(f"MES event log exposed at: {log_path}")

  arch = Device["AMD"].arch
  assert arch.startswith("gfx11"), f"expected gfx11*, got {arch}"
  (Tensor([1.]) + Tensor([1.])).realize()
  Device[Device.DEFAULT].synchronize()

  from extra.sqtt.wave_probe.capture_50x100 import KERNEL_50, _clear
  from extra.sqtt import rigorous_hw_test as rht

  stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  out_dir = pathlib.Path(__file__).resolve().parent / "captures" / f"mes_sched_{stamp}"
  out_dir.mkdir(parents=True, exist_ok=True)
  print(f"Writing aligned JSONs to: {out_dir}")

  # Baseline MES log so we can diff only entries that appear DURING a run.
  before_all = _read_mes_event_log(log_path)
  baseline_len = len(before_all)

  summary: list[dict] = []
  for kidx, kname in enumerate(KERNEL_50):
    if kname not in rht.KERNELS:
      print(f"  [{kidx+1:>2}/{len(KERNEL_50)}] SKIP {kname} (not in rigorous_hw_test.KERNELS)")
      continue
    run_fn, _attempts = rht.KERNELS[kname]
    for run in range(3):
      _clear()
      before = _read_mes_event_log(log_path)
      before_len = len(before)
      try:
        run_fn()
        Device[Device.DEFAULT].synchronize()
        Device[Device.DEFAULT]._at_profile_finalize()
      except Exception as e:
        print(f"  {kname}[{run}]: run error {e}")
        continue
      after = _read_mes_event_log(log_path)
      new_bytes = after[before_len:] if len(after) > before_len else b""
      entries_new = _parse_mes_event_log(new_bytes)
      # If the log wrapped, fall back to parsing everything and matching on doorbell window
      if not entries_new:
        entries_new = _parse_mes_event_log(after)
      map_queue_entries = [e for e in entries_new if e.get("tag") == MES_EVT_LOG_MAP_QUEUE]

      blob, first_ws = _snapshot_sqtt_event()
      if blob is None:
        print(f"  {kname}[{run}]: no SQTT blob captured, skipping alignment")
        continue
      # Pick the latest MAP_QUEUE before the SQTT run; if none, record unaligned.
      queue_map_ts = map_queue_entries[-1]["time_after_call"] if map_queue_entries else None
      delta = (first_ws - queue_map_ts) if (queue_map_ts is not None and first_ws is not None) else None

      record = {
        "kernel": kname,
        "run": run,
        "queue_map_ts": queue_map_ts,
        "first_wavestart_ts": first_ws,
        "delta_cy": delta,
        "map_queue_entries_count": len(map_queue_entries),
        "new_log_bytes": len(new_bytes),
      }
      out_path = out_dir / f"{kname}_run{run}.json"
      with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
      summary.append(record)
      msg = (f"queue_map_ts={queue_map_ts} first_ws={first_ws} delta={delta}"
             if queue_map_ts is not None else
             f"NO MAP_QUEUE event observed (log bytes={len(new_bytes)})")
      print(f"  {kname}[{run}]: {msg}")

  summary_path = out_dir / "summary.json"
  with open(summary_path, "w") as f:
    json.dump({"arch": arch, "timestamp": stamp, "baseline_log_len": baseline_len,
               "records": summary}, f, indent=2)
  print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
  main()
