#!/usr/bin/env python3
"""Declarative microbenchmark generator for the RDNA3 cycle-accurate emulator.

Lets ~300-600 microbenchmarks be defined without hand-writing each as a separate
Python function. Each microbench declares a `body` of instructions (what we're
measuring); the generator wraps it with a standard prologue (s_load_b64 +
s_waitcnt_lgkmcnt + v_lshlrev for element offset + global_load to warm VGPRs)
and a standard epilogue (global_store + s_endpgm).

Each registered microbench becomes a `Tensor.custom_kernel` fxn that renders to
Ops.PROGRAM UOp with a KernelInfo named after the microbench. This makes them
plug into rigorous_hw_test.py the same way `custom_*` kernels do.

Buffer size: `Tensor.empty(1024, float32)` by default.
  - 1024 threads = 16 WGs of 64 threads = 32 waves total.
  - RGP data showed 64-thread WGs only land wave-1 on traced CU ~15% of the time;
    16 WGs gives ~100% coverage of the traced CU on each of its 2 SIMDs.

Public API:
  @microbench(name="mb_xxx", category="single-inst")
  def _build(k: Kernel) -> None: ...       # emit body instructions on k

  sweep(base_name, param_range, body_builder, **kwargs)  # register a family

  MICROBENCHES: dict[str, MicroBench]      # name -> spec
  _run_microbench(name) -> Callable[[], Any]  # returns a run fn for rigorous_hw_test.py
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Callable, Iterable, Any

from tinygrad import Tensor, Device, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.dtype import AddrSpace
from tinygrad.renderer.amd.dsl import s, v, NULL
from tinygrad.runtime.autogen.amd.rdna3.ins import (
  s_load_b64, s_waitcnt_lgkmcnt, s_waitcnt_vmcnt, v_lshlrev_b32_e32,
  global_load_b32, global_store_b32, s_endpgm,
)

# re-export Kernel builder so microbenches can emit freely
from extra.gemm.amd_asm_matmul import Kernel  # noqa: F401 (re-exported for convenience)

# ── Registry ─────────────────────────────────────────────────────────────────

BodyBuilder = Callable[["Kernel"], None]

@dataclass
class MicroBench:
  """A single microbenchmark specification.

  The generator wraps `body` with a standard prologue/epilogue. If the body
  builder wants a different prologue (e.g. no global_load pre-warm), it can set
  `prologue_builder` to override, and `epilogue_builder` to override the tail.
  """
  name: str                                   # KernelInfo name; shows up in SQTT
  body: BodyBuilder                           # emits the body we're measuring
  size: int = 1024                            # buffer size (threads)
  category: str = "misc"                      # grouping tag for reports
  prologue_builder: BodyBuilder | None = None  # override standard prologue
  epilogue_builder: BodyBuilder | None = None  # override standard epilogue
  extra: dict = field(default_factory=dict)    # free-form metadata

MICROBENCHES: dict[str, MicroBench] = {}

def _standard_prologue(k: "Kernel") -> None:
  """The default prologue for every microbench.

  Loads the output buffer address into s[0:1] via s_load_b64, waits for it,
  computes the element offset in v[0], does a global_load into v[1] to warm
  the VGPR, and waits for the load. After this:
    - s[0:1] = output buffer address
    - v[0]   = element offset (4*thread_idx)
    - v[1]   = the value we loaded (currently unused by body, but warm)
  """
  k.emit(s_load_b64(s[0:1], s[0:1], soffset=NULL))
  k.emit(s_waitcnt_lgkmcnt(sdst=NULL, simm16=0))
  k.emit(v_lshlrev_b32_e32(v[0], 2, v[0]))
  k.emit(global_load_b32(v[1], v[0], saddr=s[0:1]))
  k.emit(s_waitcnt_vmcnt(sdst=NULL, simm16=0))

def _standard_epilogue(k: "Kernel") -> None:
  """The default epilogue: store v[1] back to the buffer, then s_endpgm."""
  k.emit(global_store_b32(addr=v[0], data=v[1], saddr=s[0:1]))
  k.emit(s_endpgm())

def register(mb: MicroBench) -> MicroBench:
  if mb.name in MICROBENCHES:
    raise ValueError(f"microbench {mb.name!r} already registered")
  MICROBENCHES[mb.name] = mb
  return mb

# ── Decorator API ────────────────────────────────────────────────────────────

def microbench(
  name: str,
  *,
  size: int = 1024,
  category: str = "misc",
  prologue: BodyBuilder | None = None,
  epilogue: BodyBuilder | None = None,
  **extra,
):
  """Decorator: register a body builder as a microbench.

  Usage:
    @microbench(name="mb_foo", category="valu", size=1024)
    def _build_foo(k: Kernel) -> None:
      k.emit(v_add_f32_e32(v[1], 1.0, v[1]))
  """
  def deco(body: BodyBuilder) -> BodyBuilder:
    register(MicroBench(
      name=name, body=body, size=size, category=category,
      prologue_builder=prologue, epilogue_builder=epilogue, extra=dict(extra),
    ))
    return body
  return deco

# ── Sweep helper ─────────────────────────────────────────────────────────────

def sweep(
  base_name: str,
  param_range: Iterable,
  body_builder: Callable[[Any], BodyBuilder],
  *,
  name_fn: Callable[[Any], str] | None = None,
  size: int = 1024,
  category: str = "sweep",
  prologue: BodyBuilder | None = None,
  epilogue: BodyBuilder | None = None,
  **extra,
) -> list[MicroBench]:
  """Register a family of microbenches parameterized over `param_range`.

  `body_builder(param)` must return a BodyBuilder (a function taking Kernel).
  Default naming: f"{base_name}_n{param}". Override with `name_fn=lambda p: ...`.

  Example:
    sweep("mb_nop_chain", range(1, 9),
          lambda n: lambda k: [k.emit(s_nop(0)) for _ in range(n)])
  """
  out = []
  for p in param_range:
    name = name_fn(p) if name_fn is not None else f"{base_name}_n{p}"
    body = body_builder(p)
    out.append(register(MicroBench(
      name=name, body=body, size=size, category=category,
      prologue_builder=prologue, epilogue_builder=epilogue, extra=dict(extra),
    )))
  return out

# ── UOp generation ───────────────────────────────────────────────────────────

def _build_program_uop(mb: MicroBench, A: UOp, arch: str) -> UOp:
  """Build the Ops.PROGRAM UOp for a registered microbench.

  Signature matches custom_* kernels so it plugs into Tensor.custom_kernel.
  If `mb.extra["lds_size"]` is set (bytes), an Ops.DEFINE_LOCAL is added to the
  sink so LDS is allocated for the microbench.
  """
  A = A.flatten()
  threads = UOp.special(A.size, "lidx0")
  k = Kernel(arch)
  (mb.prologue_builder or _standard_prologue)(k)
  mb.body(k)
  (mb.epilogue_builder or _standard_epilogue)(k)
  insts = k.finalize()
  lds_size = mb.extra.get("lds_size", 0)
  if lds_size:
    lds = UOp(Ops.DEFINE_LOCAL, dtypes.uint8.ptr(size=lds_size, addrspace=AddrSpace.LOCAL), (), "lds")
    sink = UOp.sink(A.base, lds, threads, arg=KernelInfo(mb.name))
  else:
    sink = UOp.sink(A.base, threads, arg=KernelInfo(mb.name))
  return UOp(Ops.PROGRAM, src=(
    sink,
    UOp(Ops.DEVICE, arg="AMD"),
    UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=x) for x in insts)),
  ))

def build_fxn(name: str) -> Callable[..., UOp]:
  """Return a custom_kernel-compatible fxn(A, arch) for a registered microbench."""
  if name not in MICROBENCHES:
    raise KeyError(f"microbench {name!r} not registered")
  mb = MICROBENCHES[name]
  def _fxn(A: UOp, arch: str) -> UOp: return _build_program_uop(mb, A, arch)
  _fxn.__name__ = f"microbench_{name}"
  return _fxn

# ── Runner for rigorous_hw_test.py ───────────────────────────────────────────

def _run_microbench(name: str) -> Callable[[], Any]:
  """Return a zero-arg run function suitable for rigorous_hw_test.KERNELS.

  The returned function:
    1. resolves the architecture from Device["AMD"].
    2. allocates Tensor.empty(mb.size, float32) as the I/O buffer.
    3. launches Tensor.custom_kernel with our generated fxn.
    4. realizes the result (triggers SQTT capture under PROFILE=1, SQTT=1).

  Lazy: `name` is re-looked-up at call time so demos registered after import
  still work.
  """
  def _run():
    if name not in MICROBENCHES:
      raise KeyError(f"microbench {name!r} not registered (did you import the demo module?)")
    mb = MICROBENCHES[name]

    # import here to avoid a circular import at module-load time
    from test.amd.helpers import TARGET_TO_ARCH
    from tinygrad.device import Compiled, ProfileProgramEvent, ProfileDeviceEvent

    arch = TARGET_TO_ARCH[Device["AMD"].arch]
    a = Tensor.empty(mb.size, dtype=dtypes.float32).contiguous().realize()
    Device[Device.DEFAULT].synchronize()
    # drop prior SQTT events so traces for this kernel stand alone
    Compiled.profile_events[:] = [e for e in Compiled.profile_events
                                  if isinstance(e, (ProfileProgramEvent, ProfileDeviceEvent))]
    fxn = functools.partial(build_fxn(name), arch=arch)
    return Tensor.custom_kernel(a, fxn=fxn)[0].realize()
  _run.__name__ = f"_run_microbench_{name}"
  return _run

# ── Introspection helpers ────────────────────────────────────────────────────

def list_microbenches(category: str | None = None) -> list[str]:
  if category is None: return list(MICROBENCHES)
  return [n for n, mb in MICROBENCHES.items() if mb.category == category]

def categories() -> list[str]:
  return sorted({mb.category for mb in MICROBENCHES.values()})

def describe(name: str) -> str:
  mb = MICROBENCHES[name]
  return f"{mb.name} [{mb.category}] size={mb.size}"

# When this module is imported standalone, there are zero microbenches
# registered. Import extra.sqtt.rgp.microbench_demo (or a taxonomy module) to
# populate MICROBENCHES.

__all__ = [
  "MicroBench", "MICROBENCHES",
  "microbench", "sweep", "register",
  "build_fxn", "_run_microbench",
  "list_microbenches", "categories", "describe",
  "Kernel",
]
