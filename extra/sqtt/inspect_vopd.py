#!/usr/bin/env python3
"""Inspect VOPD operands / reg_info extraction for exp_chain kernel."""
import os, sys, pickle, pathlib
os.environ.setdefault("DEV", "AMD")
os.environ.setdefault("PROFILE", "1")
os.environ.setdefault("SQTT", "1")
os.environ.setdefault("VIZ", "-2")

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from rigorous_hw_test import KERNELS, _clear
from tinygrad import Device, Tensor
from tinygrad.device import Compiled, ProfileProgramEvent
from tinygrad.renderer.amd.sqtt import map_insts, VALUINST

(Tensor([1.]) + Tensor([1.])).realize()
Device[Device.DEFAULT].synchronize()

_clear()
KERNELS["exp_chain"][0]()
Device[Device.DEFAULT].synchronize()
Device[Device.DEFAULT]._at_profile_finalize()

# Get program + walk SQTT trace
sqtt_events = [e for e in Compiled.profile_events if type(e).__name__ == 'ProfileSQTTEvent']
program_events = {e.tag: e for e in Compiled.profile_events if type(e).__name__ == 'ProfileProgramEvent'}

# Import the emu _simulate_sq_timing to access wave_events
# Simpler: walk the kernel binary to disassemble VOPD instructions.

# Actually just inspect the instructions from disassembler output at specific PCs
from extra.sqtt.rigorous_hw_test import extract_traces, _clear as c2

c2()
KERNELS["exp_chain"][0]()
Device[Device.DEFAULT].synchronize()
Device[Device.DEFAULT]._at_profile_finalize()
traces, lib = extract_traces()
wid = sorted(traces.keys())[0]
trace = traces[wid]

# Use decoder
import ctypes
print(f"Inspecting VOPD instructions at indices 16, 17, 26, 28, 37, 38, 47, 49, 50, 51, 61")

# Load raw bytes from kernel
# Look at lib
import struct
if lib:
    # Find where the shader code is stored
    # Actually easier: disassemble instruction stream
    pass

# Walk trace and print all VOPDs with their decoded operands
from tinygrad.renderer.amd.sqtt import map_insts
kev = program_events[[e.kern for e in sqtt_events][0]]
idx_of_interest = {16, 17, 26, 28, 37, 38, 47, 49, 50, 51, 61, 71}
cur_idx = -1
for pkt, info in map_insts(sqtt_events[0].blob, kev.lib, "gfx1100"):
  if info is None: continue
  cur_idx += 1
  inst = info.inst
  name = type(inst).__name__
  if cur_idx in idx_of_interest:
    attrs = {a: getattr(inst, a) for a in ('opx','opy','vdstx','vdsty','srcx0','vsrcx1','srcy0','vsrcy1','src0','src1','vsrc1','vdst','op') if hasattr(inst, a)}
    print(f"[{cur_idx}] pc=0x{info.pc:x} {name} op={attrs.get('op','?')} {attrs}")
    print(f"      full: {str(inst)[:120]}")
