#!/usr/bin/env python3
"""Inspect real gfx1100 SQTT blob tokens and compare with emulated blob."""
import sys, pickle, types, pathlib

class PermissiveUnpickler(pickle.Unpickler):
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

from tinygrad.renderer.amd.sqtt import decode, REG, WAVESTART, WAVEALLOC, WAVEEND, LAYOUT_HEADER, SNAPSHOT, TS_WAVE_STATE, INST

def show_tokens(blob: bytes, label: str, max_tokens: int = 60) -> None:
  print(f"\n{'='*60}\n{label}  blob_len={len(blob)}\nFirst 32 bytes: {blob[:32].hex()}\n{'='*60}")
  tokens = list(decode(blob))
  print(f"Total tokens: {len(tokens)}")
  for i, t in enumerate(tokens[:max_tokens]):
    name = type(t).__name__
    if isinstance(t, REG):
      print(f"  [{i:03d}] {name}: delta={t.delta} slot={t.slot} hi_byte={t.hi_byte:#04x} subop={t.subop:#06x} val32={t.val32:#010x} is_cfg={t.is_config}")
    elif isinstance(t, WAVESTART):
      print(f"  [{i:03d}] {name}: delta={t.delta} wave={t.wave} simd={t.simd} cu_lo={t.cu_lo} flag7={t.flag7} id7={t.id7}")
    elif isinstance(t, WAVEALLOC):
      print(f"  [{i:03d}] {name}: delta={t.delta}")
    elif isinstance(t, INST):
      print(f"  [{i:03d}] {name}: delta={t.delta} wave={t.wave} op={t.op}")
    elif isinstance(t, WAVEEND):
      print(f"  [{i:03d}] {name}: delta={t.delta} wave={t.wave} simd={t.simd}")
    elif isinstance(t, SNAPSHOT):
      print(f"  [{i:03d}] {name}: delta={t.delta} snap={t.snap:#x}")
    elif isinstance(t, TS_WAVE_STATE):
      print(f"  [{i:03d}] {name}: delta={t.delta} coarse={t.coarse:#x} wave_interest={t.wave_interest}")
    elif isinstance(t, LAYOUT_HEADER):
      print(f"  [{i:03d}] {name}: layout={t.layout} sel_a={t.sel_a}")
    else:
      print(f"  [{i:03d}] {name}: delta={getattr(t, 'delta', 'N/A')}")
  if len(tokens) > max_tokens:
    print(f"  ... ({len(tokens) - max_tokens} more tokens)")

# Load real gfx1100 blob
pkl_path = pathlib.Path("extra/sqtt/examples/gfx1100/profile_plus_run_0.pkl")
with pkl_path.open("rb") as f:
  data = PermissiveUnpickler(f).load()

sqtt_ev = next(ev for ev in data if type(ev).__name__ == "ProfileSQTTEvent" and getattr(ev, "itrace", False))
prg_ev = next(ev for ev in data if type(ev).__name__ == "ProfileProgramEvent")
print(f"prg_ev: base={prg_ev.base:#x} tag={prg_ev.tag} lib_len={len(prg_ev.lib)}")
print(f"sqtt_ev: kern={sqtt_ev.kern} se={sqtt_ev.se} blob_len={len(sqtt_ev.blob)}")

show_tokens(sqtt_ev.blob, "REAL gfx1100 profile_plus_run_0", max_tokens=20)

# Find all interesting tokens
tokens = list(decode(sqtt_ev.blob))
print(f"\n--- All REG, WAVEALLOC, WAVESTART, SNAPSHOT, TS_WAVE_STATE tokens ---")
for i, t in enumerate(tokens):
  if isinstance(t, (REG, WAVEALLOC, WAVESTART, SNAPSHOT, TS_WAVE_STATE)):
    name = type(t).__name__
    if isinstance(t, REG):
      print(f"  [{i:04d}] REG: slot={t.slot} hi_byte={t.hi_byte:#04x} subop={t.subop:#06x} val32={t.val32:#010x} is_cfg={t.is_config}")
    elif isinstance(t, WAVEALLOC):
      print(f"  [{i:04d}] WAVEALLOC: delta={t.delta}")
    elif isinstance(t, WAVESTART):
      print(f"  [{i:04d}] WAVESTART: delta={t.delta} wave={t.wave} simd={t.simd} cu={t.cu} id7={t.id7}")
    elif isinstance(t, SNAPSHOT):
      print(f"  [{i:04d}] SNAPSHOT: delta={t.delta} snap={t.snap:#x}")
    elif isinstance(t, TS_WAVE_STATE):
      print(f"  [{i:04d}] TS_WAVE_STATE: delta={t.delta} coarse={t.coarse:#x} wave_interest={t.wave_interest}")
  if i > 800: print("... stopping"); break
