#!/usr/bin/env python3
"""Run discover_ops on real HW with SQTT capture and save traces."""
import os, sys, pickle, pathlib, functools, inspect
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
  from test.amd.helpers import TARGET_TO_ARCH
  TARGET = Device["AMD"].arch

  # Import discover_ops machinery
  sys.path.insert(0, str(pathlib.Path(__file__).parent / "examples"))
  import discover_ops

  arch = Device[Device.DEFAULT].renderer.target.arch
  discover_ops.arch = arch
  if arch.startswith("gfx11"):
    from tinygrad.runtime.autogen.amd.rdna3.ins import *
    import tinygrad.runtime.autogen.amd.rdna3.ins as all_insts
    discover_ops.all_insts = all_insts
    discover_ops.SKIP.update(["S_FMAAK_F32", "S_FMAMK_F32"])
    for name, obj in inspect.getmembers(all_insts):
      setattr(discover_ops, name, obj)
  else:
    print(f"{arch} not supported"); sys.exit(1)

  alu_insts, mem_insts, skipped = discover_ops.collect_instructions()
  print(f"collected {len(alu_insts)} ALU + {len(mem_insts)} memory instructions ({len(skipped)} skipped)")

  # Try up to 10 times to get a capture with enough inst packets
  for attempt in range(10):
    try:
      _clear()
      discover_ops.exec_insts(mem_insts + alu_insts)
      Device[Device.DEFAULT].synchronize()
      Device[Device.DEFAULT]._at_profile_finalize()
      traces = extract_traces(TARGET)
      total_insts = sum(sum(1 for _,_,t,_ in pkts if t in ("INST","VALUINST","IMMEDIATE")) for pkts in traces.values())
      if total_insts >= 100:
        out = pathlib.Path(__file__).parent / "captures" / "rigorous" / "discover_ops.pkl"
        out.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(traces, open(out, "wb"))
        print(f"attempt {attempt}: {len(traces)} waves, total insts={total_insts}")
        for wid, pkts in sorted(traces.items()):
          ic = sum(1 for _,_,t,_ in pkts if t in ("INST","VALUINST","IMMEDIATE"))
          print(f"  wave {wid}: {len(pkts)} packets, {ic} inst-level")
        print(f"Saved → {out}")
        break
      print(f"attempt {attempt}: only {total_insts} insts, retrying...")
    except Exception as e:
      print(f"attempt {attempt}: error {e}")
      if attempt >= 3: break
  else:
    print("FAILED to capture after 10 attempts")
