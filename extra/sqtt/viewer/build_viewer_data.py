#!/usr/bin/env python3
"""Build viewer_data.json from SQTT capture artifacts.

Session-1 scope: ingest the per-wave instruction traces that
extra/sqtt/rigorous_hw_test.py saves under extra/sqtt/captures/rigorous/
(dict[wave_id] -> [(pc, time, type_str, inst_str), ...]) and emit a single
JSON file the HTML viewer consumes with no further preprocessing.

We also try to enrich each wave with (cu, simd) placement if a raw-SQTT
capture directory is passed alongside — decode_all_simds.py already knows
how to recover that from the raw blobs.

Usage:
  .venv/bin/python extra/sqtt/viewer/build_viewer_data.py \\
      --captures extra/sqtt/captures/rigorous \\
      [--raw extra/sqtt/wave_probe/captures/raw_sqtt_<ts>] \\
      [--out extra/sqtt/viewer/viewer_data.json]

Output schema:
  {
    "schema": 1,
    "arch": "gfx1100",
    "source": "<capture dir>",
    "kernels": {
      "<kernel_name>": {
        "name": str,
        "total_waves": int,
        "total_instructions": int,
        "time_min": int, "time_max": int,     # normalised absolute cycles
        "waves": [
          {
            "wave_id": int,
            "cu": int|null, "simd": int|null,  # only if raw blob supplied
            "inst_count": int,
            "time_min": int, "time_max": int,
            "instructions": [
              {"idx": int, "pc": "0xNN", "t": int,
               "type": "VALUINST"|"INST"|"IMMEDIATE"|...,
               "inst": str, "cat": "valu"|"salu"|"vmem"|...}
            ]
          }, ...
        ]
      }, ...
    }
  }
"""
import os, sys, json, pickle, pathlib, argparse, re
from collections import defaultdict

_repo_root = pathlib.Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path: sys.path.insert(0, str(_repo_root))


# --- Instruction categorization --------------------------------------------
# Keep this intentionally small and regex-driven so new ISA encodings don't
# silently fall into "other". Order matters: more specific patterns first.
CAT_RULES: list[tuple[str, re.Pattern]] = [
  ("vopd",   re.compile(r"^v_dual_")),
  ("trans",  re.compile(r"^v_(exp|log|rcp|rsq|sqrt|sin|cos|rcp_iflag|rsq_iflag)_")),
  ("vmem",   re.compile(r"^(global_|buffer_|flat_|scratch_|image_|tbuffer_)")),
  ("smem",   re.compile(r"^s_(load|buffer_load|store|buffer_store)")),
  ("lds",    re.compile(r"^ds_")),
  ("branch", re.compile(r"^s_(branch|cbranch|setpc|swappc|call|endpgm|sendmsg)")),
  ("wait",   re.compile(r"^s_(waitcnt|delay_alu|barrier|sleep|nop)")),
  ("salu",   re.compile(r"^s_")),
  ("valu",   re.compile(r"^v_")),
]

def categorise(inst: str) -> str:
  s = (inst or "").strip().lower()
  for cat, pat in CAT_RULES:
    if pat.search(s): return cat
  return "other"


# --- Optional RGP enrichment -----------------------------------------------

def load_rgp_placements(rgp_dir: pathlib.Path | None) -> dict[str, list[dict]]:
  """Return {kernel_name: [{se, cu, simd, wave, time}, ...]} from an rgp dir.

  Each kernel matches by stem against <kernel>.rgp / probe_<kernel>.rgp.
  Silently returns {} if the dir or parse_rgp helper is unavailable.
  """
  if rgp_dir is None: return {}
  if not rgp_dir.is_dir():
    print(f"warn: rgp dir not found: {rgp_dir}", file=sys.stderr)
    return {}
  try:
    from extra.sqtt.rgp.parse_rgp import parse as parse_rgp
  except Exception as e:
    print(f"warn: rgp parser unavailable ({e}) — skipping rgp enrichment", file=sys.stderr)
    return {}
  out: dict[str, list[dict]] = {}
  for rgp in sorted(rgp_dir.glob("*.rgp")):
    name = rgp.stem
    if name.startswith("probe_"): name = name[len("probe_"):]
    try:
      per_se = parse_rgp(str(rgp))
      flat = []
      for entry in per_se:
        for w in entry["waves"]:
          flat.append({"se": entry["se"], **w})
      out[name] = flat
    except Exception as e:
      print(f"warn: rgp parse of {rgp.name} failed: {e}", file=sys.stderr)
  return out


# --- Optional raw-blob enrichment ------------------------------------------

def load_placements(raw_dir: pathlib.Path | None) -> dict[str, dict[int, tuple[int, int]]]:
  """Return {kernel_name: {wave_id: (cu, simd)}} from a raw-SQTT capture dir.

  Mirrors the logic in extra/sqtt/wave_probe/decode_all_simds.py but without
  the side-effect JSON writing. Silent empty dict if raw_dir is None or empty.
  """
  if raw_dir is None: return {}
  if not raw_dir.is_dir():
    print(f"warn: raw dir not found: {raw_dir}", file=sys.stderr)
    return {}
  try:
    from tinygrad.renderer.amd.sqtt import decode, WAVESTART, WAVESTART_RDNA4, WAVEEND
  except Exception as e:
    print(f"warn: raw decode unavailable ({e}) — skipping placement enrichment", file=sys.stderr)
    return {}

  out: dict[str, dict[int, tuple[int, int]]] = {}
  for pkl in sorted(raw_dir.glob("*.pkl")):
    try:
      with open(pkl, "rb") as f: rec = pickle.load(f)
      blob = rec.get("blob") if isinstance(rec, dict) else None
      if blob is None: continue
      placements: dict[int, tuple[int, int]] = {}
      starts: dict[tuple[int, int, int], int] = {}
      for p in decode(blob):
        if isinstance(p, (WAVESTART, WAVESTART_RDNA4)):
          starts[(p.cu, p.simd, p.wave)] = p._time
          placements.setdefault(p.wave, (p.cu, p.simd))
      out[pkl.stem] = placements
    except Exception as e:
      print(f"warn: raw decode of {pkl.name} failed: {e}", file=sys.stderr)
  return out


# --- Main ingest -----------------------------------------------------------

def load_rigorous_pkl(path: pathlib.Path) -> dict[int, list[tuple]]:
  """Load a rigorous-format pkl: {wave_id: [(pc, time, type_str, inst_str?), ...]}.

  Older captures store 3-tuples (no inst_str). Normalise both shapes.
  """
  with open(path, "rb") as f: raw = pickle.load(f)
  out: dict[int, list[tuple]] = {}
  for wid, trace in raw.items():
    norm = []
    for entry in trace:
      if len(entry) >= 4:
        pc, t, typ, inst = entry[0], entry[1], entry[2], entry[3]
      elif len(entry) == 3:
        pc, t, typ, inst = entry[0], entry[1], entry[2], ""
      else:
        continue
      norm.append((pc, t, typ, inst))
    out[wid] = norm
  return out


def build_kernel_entry(name: str, traces: dict[int, list[tuple]],
                       placement: dict[int, tuple[int, int]] | None = None,
                       emu: dict | None = None) -> dict:
  all_times = [t for trace in traces.values() for _, t, _, _ in trace]
  if not all_times:
    return {"name": name, "total_waves": 0, "total_instructions": 0,
            "time_min": 0, "time_max": 0, "waves": []}
  tmin = min(all_times)
  # Build positional map of emu-per-wave-per-inst for fast lookup.
  # timing_data.json uses wave_idx (0-based) matching sorted(trace.keys()).
  emu_by_pos: dict[int, dict[int, dict]] = {}
  if emu and isinstance(emu.get("waves"), list):
    for w in emu["waves"]:
      if not w.get("pc_match"): continue
      wi = w.get("wave_idx")
      if wi is None: continue
      emu_by_pos[wi] = {i["idx"]: i for i in (w.get("instructions") or []) if "idx" in i}
  status_counts: dict[str, int] = defaultdict(int)
  waves_out = []
  for pos, wid in enumerate(sorted(traces.keys())):
    trace = traces[wid]
    if not trace: continue
    cu_simd = (placement or {}).get(wid)
    emu_insts = emu_by_pos.get(pos, {})
    insts = []
    for idx, (pc, t, typ, inst) in enumerate(trace):
      item = {
        "idx": idx,
        "pc": f"0x{pc:x}" if isinstance(pc, int) else str(pc),
        "t": int(t - tmin),
        "type": typ,
        "inst": inst,
        "cat": categorise(inst),
      }
      emu_i = emu_insts.get(idx)
      if emu_i is not None:
        item["hw_delta"] = emu_i.get("hw_delta")
        item["emu_delta"] = emu_i.get("emu_delta")
        item["diff"] = emu_i.get("diff")
        item["status"] = emu_i.get("status")
        if emu_i.get("status"): status_counts[emu_i["status"]] += 1
      insts.append(item)
    wt = [i["t"] for i in insts]
    waves_out.append({
      "wave_id": int(wid),
      "cu": cu_simd[0] if cu_simd else None,
      "simd": cu_simd[1] if cu_simd else None,
      "inst_count": len(insts),
      "time_min": min(wt), "time_max": max(wt),
      "instructions": insts,
    })
  out = {
    "name": name,
    "total_waves": len(waves_out),
    "total_instructions": sum(w["inst_count"] for w in waves_out),
    "time_min": 0,
    "time_max": max(w["time_max"] for w in waves_out) if waves_out else 0,
    "waves": waves_out,
  }
  if emu:
    stats = emu.get("stats") or {}
    out["emu"] = {
      "compared": int(stats.get("compared", 0)),
      "exact": int(stats.get("exact", 0)),
      "pct": float(stats.get("pct", 0.0)),
      "status_counts": dict(status_counts),
    }
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--captures", type=pathlib.Path, default=pathlib.Path("extra/sqtt/captures/rigorous"),
                  help="directory of <kernel>.pkl rigorous-format captures")
  ap.add_argument("--raw", type=pathlib.Path, default=None,
                  help="optional raw SQTT capture dir for (cu, simd) enrichment")
  ap.add_argument("--rgp", type=pathlib.Path, default=None,
                  help="optional rgp capture dir (one <kernel>.rgp per kernel) for SE-wide wave placements")
  ap.add_argument("--emu-timing", type=pathlib.Path, default=None,
                  help="optional timing_data.json with HW-vs-Emu per-instruction deltas")
  ap.add_argument("--out", type=pathlib.Path,
                  default=pathlib.Path(__file__).resolve().parent / "viewer_data.json")
  ap.add_argument("--arch", default="gfx1100")
  args = ap.parse_args()

  cap_dir = args.captures.resolve()
  if not cap_dir.is_dir():
    print(f"error: captures dir not found: {cap_dir}", file=sys.stderr); sys.exit(1)

  placements = load_placements(args.raw)
  rgp_waves = load_rgp_placements(args.rgp)
  if rgp_waves:
    print(f"loaded rgp placements for {len(rgp_waves)} kernels from {args.rgp}")

  emu_doc: dict = {}
  if args.emu_timing is not None:
    p = args.emu_timing.resolve()
    if p.is_file():
      try:
        with open(p) as f: emu_doc = (json.load(f) or {}).get("kernels", {}) or {}
        print(f"loaded emu timing for {len(emu_doc)} kernels from {p}")
      except Exception as e:
        print(f"warn: emu timing load failed: {e}", file=sys.stderr)
    else:
      print(f"warn: emu timing file not found: {p}", file=sys.stderr)

  kernels: dict[str, dict] = {}
  for pkl in sorted(cap_dir.glob("*.pkl")):
    name = pkl.stem
    try:
      traces = load_rigorous_pkl(pkl)
      if not traces: continue
      kernels[name] = build_kernel_entry(name, traces, placements.get(name),
                                         emu=emu_doc.get(name))
      if name in rgp_waves:
        kernels[name]["rgp_waves"] = rgp_waves[name]
      k = kernels[name]
      emu_tag = ""
      if "emu" in k and k["emu"]["compared"]:
        emu_tag = f"  emu={k['emu']['pct']:.1f}%({k['emu']['exact']}/{k['emu']['compared']})"
      print(f"  {name:35s} waves={k['total_waves']:4d} insts={k['total_instructions']:5d} "
            f"span={k['time_max']} cyc{emu_tag}")
    except Exception as e:
      print(f"  {name}: FAILED {e}", file=sys.stderr)

  out_doc = {
    "schema": 1, "arch": args.arch, "source": str(cap_dir),
    "has_emu": bool(emu_doc),
    "kernel_count": len(kernels), "kernels": kernels,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  with open(args.out, "w") as f: json.dump(out_doc, f, separators=(",", ":"))
  size_kb = args.out.stat().st_size / 1024
  print(f"\nWrote {args.out} ({size_kb:.1f} KB, {len(kernels)} kernels)")


if __name__ == "__main__":
  main()
