#!/usr/bin/env python3
"""Analyze real SQTT traces to understand token ordering around PGM_LO/HI."""
import pickle, types, sys, pathlib
from collections import Counter

class PU(pickle.Unpickler):
  def find_class(self, module, name):
    mod = sys.modules.get(module)
    if mod is None:
      mod = types.ModuleType(module)
      sys.modules[module] = mod
    obj = getattr(mod, name, None)
    if obj is not None: return obj
    cls = type(name, (object,), {'__init__': lambda s, *a, **k: s.__dict__.update(k)})
    setattr(mod, name, cls)
    return cls

from tinygrad.renderer.amd.sqtt import decode, REG, EVENT_BIG, WAVESTART, WAVEALLOC, SNAPSHOT, TS_WAVE_STATE, LAYOUT_HEADER, EVENT

def token_str(i, t):
  name = type(t).__name__
  if isinstance(t, REG):
    return f"[{i:04d}] REG: slot={t.slot} hi={t.hi_byte:#04x} subop={t.subop:#06x} val={t.val32:#010x} cfg={t.is_config}"
  if isinstance(t, SNAPSHOT): return f"[{i:04d}] SNAPSHOT: snap={t.snap:#x}"
  if isinstance(t, TS_WAVE_STATE): return f"[{i:04d}] TS_WAVE_STATE: coarse={t.coarse:#x} wi={t.wave_interest}"
  if isinstance(t, EVENT): return f"[{i:04d}] EVENT: delta={t.delta} event={t.event:#x}"
  if isinstance(t, LAYOUT_HEADER): return f"[{i:04d}] LAYOUT_HEADER: layout={t.layout} sel_a={t.sel_a}"
  return f"[{i:04d}] {name}: delta={getattr(t, 'delta', '?')}"

pkl_path = pathlib.Path("extra/sqtt/examples/gfx1100/profile_sync_run_0.pkl")
with pkl_path.open("rb") as f:
  data = PU(f).load()

prg_ev = next(ev for ev in data if type(ev).__name__ == "ProfileProgramEvent")
print(f"prg_ev: base={prg_ev.base:#x}")

# Find SQTT event with non-zero PGM values
for ev in data:
  if type(ev).__name__ != "ProfileSQTTEvent": continue
  if not getattr(ev, "itrace", False): continue
  tokens = list(decode(ev.blob))
  pgm_tokens = [(i, t) for i, t in enumerate(tokens) if isinstance(t, REG) and t.subop == 0xC and t.hi_byte == 0x82 and t.val32 != 0]
  if not pgm_tokens: continue
  print(f"Total tokens: {len(tokens)}, non-zero PGM at indices: {[i for i,_ in pgm_tokens]}")
  counts = Counter(type(t).__name__ for t in tokens)
  print("Token counts:", dict(sorted(counts.items(), key=lambda x: -x[1])))
  idx = pgm_tokens[0][0]
  print(f"\nContext around token {idx} (first non-zero PGM_LO):")
  for i in range(max(0, idx-12), min(len(tokens), idx+15)):
    print(" ", token_str(i, tokens[i]))
  break
