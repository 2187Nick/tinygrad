#!/usr/bin/env python3
"""Compare discover_ops HW trace vs emulator trace per-op."""
import os, sys, pickle, pathlib, inspect
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from tinygrad import Tensor, Device, dtypes
from tinygrad.device import Compiled, ProfileProgramEvent, ProfileDeviceEvent
from tinygrad.renderer.amd.sqtt import map_insts

def _clear():
  Compiled.profile_events[:] = [e for e in Compiled.profile_events if isinstance(e, (ProfileProgramEvent, ProfileDeviceEvent))]

def extract_traces(TARGET):
  sqtt_events = [e for e in Compiled.profile_events if type(e).__name__ == 'ProfileSQTTEvent']
  program_events = {e.tag: e for e in Compiled.profile_events if type(e).__name__ == 'ProfileProgramEvent'}
  all_traces = {}
  for ev in sqtt_events:
    if not ev.itrace: continue
    if ev.kern not in program_events: continue
    kev = program_events[ev.kern]
    for pkt, info in map_insts(ev.blob, kev.lib, TARGET):
      if info is None: continue
      w = info.wave
      if w not in all_traces: all_traces[w] = []
      all_traces[w].append((info.pc, pkt._time, type(pkt).__name__, str(info.inst)[:200]))
  return all_traces

if __name__ == "__main__":
  assert os.environ.get('MOCKGPU') == '1', "need MOCKGPU=1 for emulator side"
  TARGET = Device["AMD"].arch
  sys.path.insert(0, str(pathlib.Path(__file__).parent / "examples"))
  import discover_ops
  arch = Device[Device.DEFAULT].renderer.target.arch
  discover_ops.arch = arch
  from tinygrad.runtime.autogen.amd.rdna3.ins import *
  import tinygrad.runtime.autogen.amd.rdna3.ins as all_insts
  discover_ops.all_insts = all_insts
  discover_ops.SKIP.update(["S_FMAAK_F32", "S_FMAMK_F32"])
  for name, obj in inspect.getmembers(all_insts):
    setattr(discover_ops, name, obj)

  alu_insts, mem_insts, skipped = discover_ops.collect_instructions()

  _clear()
  discover_ops.exec_insts(mem_insts + alu_insts)
  Device[Device.DEFAULT].synchronize()
  Device[Device.DEFAULT]._at_profile_finalize()
  emu_traces = extract_traces(TARGET)

  hw_path = pathlib.Path(__file__).parent / "captures" / "rigorous" / "discover_ops.pkl"
  hw_traces = pickle.load(open(hw_path, "rb"))

  print(f"HW waves: {list(hw_traces.keys())}, EMU waves: {list(emu_traces.keys())}")
  hw_wid = list(hw_traces.keys())[0]
  emu_wid = list(emu_traces.keys())[0]
  hw = hw_traces[hw_wid]
  emu = emu_traces[emu_wid]

  # Match by PC index; align lengths
  n = min(len(hw), len(emu))
  print(f"HW={len(hw)}, EMU={len(emu)}, comparing {n}")

  # Compute deltas and classify mismatches by op category
  from collections import defaultdict
  by_cat = defaultdict(lambda: {'exact':0,'pm1':0,'pm2':0,'total':0,'hw_sum':0,'emu_sum':0,'examples':[]})
  mismatch = []
  for i in range(1, n):
    hw_delta = hw[i][1] - hw[i-1][1]
    emu_delta = emu[i][1] - emu[i-1][1]
    if hw[i][0] != emu[i][0]:
      continue  # PC skew; skip
    inst = hw[i][3]
    # op category: first word up to first "("
    op = inst.split('(',1)[0].strip() if inst else 'UNKNOWN'
    if op.startswith('v_'): cat = 'v_'+op.split('_')[1] if len(op.split('_'))>1 else op
    elif op.startswith('s_'): cat = 's_'+op.split('_')[1] if len(op.split('_'))>1 else op
    else: cat = op
    # Handle huge prologue deltas (first loads): skip >100 cycle deltas as outliers
    if hw_delta > 100 or emu_delta > 100:
      continue
    d = by_cat[cat]
    d['total'] += 1
    d['hw_sum'] += hw_delta
    d['emu_sum'] += emu_delta
    diff = emu_delta - hw_delta
    if diff == 0: d['exact'] += 1
    if abs(diff) <= 1: d['pm1'] += 1
    if abs(diff) <= 2: d['pm2'] += 1
    if abs(diff) > 2 and len(d['examples']) < 3:
      d['examples'].append((i, hw_delta, emu_delta, inst[:60]))
    if abs(diff) >= 3:
      mismatch.append((abs(diff), i, hw_delta, emu_delta, op, inst[:80]))

  # Summary per op category
  print(f"\n{'cat':<20} {'N':>4} {'exact%':>7} {'±2%':>6} {'hw_avg':>7} {'emu_avg':>7}")
  for cat, d in sorted(by_cat.items(), key=lambda x: -x[1]['total']):
    if d['total'] < 3: continue
    print(f"{cat:<20} {d['total']:>4} {100*d['exact']/d['total']:>6.1f}% {100*d['pm2']/d['total']:>5.1f}% "
          f"{d['hw_sum']/d['total']:>7.2f} {d['emu_sum']/d['total']:>7.2f}")

  total = sum(d['total'] for d in by_cat.values())
  total_exact = sum(d['exact'] for d in by_cat.values())
  total_pm2 = sum(d['pm2'] for d in by_cat.values())
  print(f"\nTOTAL: {total_exact}/{total} exact ({100*total_exact/total:.1f}%), {total_pm2}/{total} ±2 ({100*total_pm2/total:.1f}%)")

  # Top biggest-diff offenders
  print("\n--- Top 30 biggest single-packet mismatches ---")
  mismatch.sort(reverse=True)
  for d, i, h, e, op, inst in mismatch[:30]:
    print(f"  [{i:4d}] HW={h:3d} EMU={e:3d} diff={e-h:+4d} {inst}")
