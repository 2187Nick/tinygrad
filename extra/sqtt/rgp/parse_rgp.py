#!/usr/bin/env python3
"""Extract wave placement metadata from an .rgp file captured by Mesa RADV.

Reuses tinygrad's RGP parser (extra/sqtt/rgptool.py) and SQTT decoder
(tinygrad/renderer/amd/sqtt.py) to walk the SQTT packets in every SE's data
chunk and collect (se, cu, simd, wave, time_ns) tuples for each WAVESTART.

Output:
  • stdout: one WAVESTART per line, grouped by SE
  • <out>.json (optional): same data + per-(cu,simd) wave counts
"""
from __future__ import annotations
import sys, os, json, argparse
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

from extra.sqtt.rgptool import RGP
import tinygrad.runtime.autogen.sqtt as sqtt_ct
from tinygrad.renderer.amd.sqtt import decode, WAVESTART, WAVESTART_RDNA4

def parse(path: str):
  with open(path, "rb") as f:
    rgp = RGP.from_bytes(f.read())
  # Pair SQTT_DESC with following SQTT_DATA chunks in file order.
  descs, datas = [], []
  for ch in rgp.chunks:
    cid = ch.header.header.chunk_id.type
    if cid == sqtt_ct.SQTT_FILE_CHUNK_TYPE_SQTT_DESC:  descs.append(ch)
    elif cid == sqtt_ct.SQTT_FILE_CHUNK_TYPE_SQTT_DATA: datas.append(ch)
  if len(descs) != len(datas):
    print(f"WARNING: {path} has {len(descs)} descs and {len(datas)} datas", file=sys.stderr)
  per_se = []
  for desc, data in zip(descs, datas):
    se = desc.header.shader_engine_index
    cu = desc.header.v1.compute_unit_index
    waves = []
    for p in decode(data.data):
      if isinstance(p, (WAVESTART, WAVESTART_RDNA4)):
        waves.append({"wave": p.wave, "simd": p.simd, "cu": p.cu, "time": int(p._time)})
    per_se.append({"se": int(se), "traced_cu": int(cu), "waves": waves})
  return per_se

def fmt_placement(per_se):
  # compact grid of (se, cu, simd) -> [waves ...]
  grid: dict[tuple[int,int,int], list[int]] = defaultdict(list)
  for entry in per_se:
    se = entry["se"]
    for w in entry["waves"]:
      grid[(se, w["cu"], w["simd"])].append(w["wave"])
  return grid

def cluster_by_time(per_se, dt_ns: int = 200):
  """Group wavestarts on same (se,cu,simd) that occur within dt_ns of each other.
  A cluster of size>1 likely belongs to the same workgroup (multiple waves
  issued simultaneously to the same SIMD)."""
  clusters = []
  for entry in per_se:
    by_unit: dict[tuple[int,int,int], list] = defaultdict(list)
    for w in entry["waves"]:
      by_unit[(entry["se"], w["cu"], w["simd"])].append(w)
    for key, ws in by_unit.items():
      ws.sort(key=lambda w: w["time"])
      cur = [ws[0]]
      for w in ws[1:]:
        if w["time"] - cur[-1]["time"] <= dt_ns:
          cur.append(w)
        else:
          clusters.append((key, cur)); cur = [w]
      clusters.append((key, cur))
  return clusters

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("rgp", nargs="+")
  ap.add_argument("--json", action="store_true", help="also write <file>.waves.json")
  args = ap.parse_args()
  for path in args.rgp:
    print(f"\n═══ {os.path.basename(path)} ═══")
    per_se = parse(path)
    total = 0
    for entry in per_se:
      waves = entry["waves"]
      total += len(waves)
      if not waves: continue
      head = waves[:4]
      tail = f" …(+{len(waves)-4} more)" if len(waves) > 4 else ""
      hd = ", ".join(f"w{w['wave']}@cu{w['cu']}.simd{w['simd']}" for w in head)
      print(f"  SE{entry['se']} (traced_cu={entry['traced_cu']}): {len(waves)} WAVESTART → {hd}{tail}")
    if total == 0:
      print("  (no WAVESTART packets found — check RADV_THREAD_TRACE_INSTRUCTION_TIMING)")
      continue
    # cluster-by-time: waves starting within 200ns on same SIMD likely = same WG
    clusters = cluster_by_time(per_se, dt_ns=200)
    sizes: dict[int,int] = defaultdict(int)
    for _, c in clusters: sizes[len(c)] += 1
    if sizes:
      print("  cluster sizes (waves co-issued on same SIMD within 200ns):")
      for n in sorted(sizes):
        print(f"    {n}-wave clusters: {sizes[n]}")
    # largest clusters give direct evidence of workgroup→SIMD packing
    big = [(k, c) for k, c in clusters if len(c) >= 2]
    if big:
      print(f"  sample multi-wave clusters (first 6):")
      for (se, cu, simd), c in big[:6]:
        slots = [w["wave"] for w in c]
        print(f"    SE{se} CU{cu:2d} SIMD{simd}: t0={c[0]['time']} slots={slots}")
    if args.json:
      out = path + ".waves.json"
      with open(out, "w") as f:
        json.dump({"file": os.path.basename(path), "per_se": per_se}, f, indent=2)
      print(f"  -> {out}")

if __name__ == "__main__":
  main()
