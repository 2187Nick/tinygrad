#!/usr/bin/env python3
"""SQTT Cycle-Accurate Emulator: Beam Search Proof-of-Concept

Demonstrates using the MOCKGPU SQTT emulator to predict kernel performance
and rank optimization variants — without touching real GPU hardware.

Usage (from repo root):
  PYTHONPATH=. DEV=AMD MOCKGPU=1 PYTHON_REMU=1 PROFILE=1 SQTT=1 .venv/bin/python extra/sqtt/beam_search_poc.py
"""
import os, sys, traceback, time
from dataclasses import replace

# ── sanity-check environment ────────────────────────────────────────────
for var in ("MOCKGPU", "PROFILE", "SQTT"):
  assert os.environ.get(var) == "1", f"Must run with {var}=1"

import numpy as np
from tinygrad import Tensor, Device
from tinygrad.helpers import Context, DEBUG, unwrap
from tinygrad.device import Buffer
from tinygrad.uop.ops import Ops
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.postrange import Scheduler, bufs_from_ast
from tinygrad.codegen.opt.search import get_kernel_actions, _ensure_buffer_alloc
from tinygrad.codegen import get_program
from tinygrad.engine.realize import CompiledRunner

# ── helpers ─────────────────────────────────────────────────────────────

def get_compute_ast():
  """Build a 128×128 matmul and return the kernel AST (Ops.SINK)."""
  # use numpy data so the inputs are already realised — schedule only contains the matmul
  a = Tensor(np.random.rand(128, 128).astype(np.float32)).realize()
  b = Tensor(np.random.rand(128, 128).astype(np.float32)).realize()
  c = (a @ b).contiguous()
  schedule = c.schedule()
  # pick the last compute kernel (SINK)
  for ei in reversed(schedule):
    if ei.ast.op == Ops.SINK:
      return ei.ast
  raise RuntimeError("no compute kernel found in schedule")


def compile_variant(s: Scheduler, compiler):
  """Compile one Scheduler variant.  Returns (ProgramSpec, lib, uop_count) or None."""
  try:
    p = get_program(s.copy().get_optimized_ast(name_override="test"), s.ren)
    assert p.uops is not None, "uop list not generated"
    lib = p.lib if p.lib is not None else compiler.compile(p.src)
    return p, lib, len(p.uops)
  except Exception:
    if DEBUG >= 3: traceback.print_exc()
    return None


def run_variant(p, lib, rawbufs, var_vals, device):
  """Run a compiled variant through the emulator; return predicted cycle count."""
  from test.mockgpu.amd.emu import sqtt_cycle_counts
  before = len(sqtt_cycle_counts)
  car = CompiledRunner(replace(p, lib=lib, device=device))
  input_bufs = [rawbufs[i] for i in car.p.globals]
  car(input_bufs, var_vals, wait=True)
  Device[device].synchronize()
  if len(sqtt_cycle_counts) > before:
    return sqtt_cycle_counts[-1]
  return None


def fmt_opts(opts):
  if not opts:
    return "(baseline — no optimizations)"
  return ", ".join(str(o) for o in opts)


# ── main ────────────────────────────────────────────────────────────────

def main():
  print("=" * 80)
  print("  SQTT Cycle-Accurate Emulator — Beam Search POC")
  print("=" * 80)
  print(f"  Device : {Device.DEFAULT}")
  print(f"  Mode   : MOCKGPU emulation with SQTT tracing")
  print()

  # 1. kernel AST ---------------------------------------------------------
  print("[1/5] Creating 128×128 matmul kernel …")
  ast = get_compute_ast()
  device = Device.DEFAULT
  renderer = Device[device].renderer
  compiler = Device[device].compiler
  print(f"       AST root op: {ast.op}")

  # 2. enumerate actions ---------------------------------------------------
  print("[2/5] Enumerating optimisation variants …")
  s = Scheduler(ast, renderer)
  s.convert_loop_to_global()
  actions = get_kernel_actions(s, include_0=True)
  print(f"       Found {len(actions)} valid variants (incl. baseline)")

  # 3. allocate emulated buffers -------------------------------------------
  print("[3/5] Allocating emulated buffers …")
  rawbufs = _ensure_buffer_alloc(bufs_from_ast(ast, device))
  var_vals: dict[str, int] = {k.expr: int(k.vmax + k.vmin) // 2 for k in ast.variables()}

  # 4. compile + emulate each variant -------------------------------------
  print("[4/5] Compiling & emulating each variant …")
  print()

  results: list[dict] = []
  compile_fails = 0
  run_fails = 0

  t0 = time.perf_counter()
  for idx, (action_id, variant_s) in enumerate(actions.items()):
    # compile
    compiled = compile_variant(variant_s, compiler)
    if compiled is None:
      compile_fails += 1
      continue
    p, lib, uop_count = compiled

    # run through emulator
    try:
      cycles = run_variant(p, lib, rawbufs, var_vals, device)
      if cycles is not None:
        results.append(dict(action_id=action_id, opts=variant_s.applied_opts,
                            cycles=cycles, uop_count=uop_count))
      else:
        run_fails += 1
    except Exception as e:
      run_fails += 1
      if DEBUG >= 1: print(f"       Run failed: {e}")

    # progress
    done = idx + 1
    if done % 5 == 0 or done == len(actions):
      elapsed = time.perf_counter() - t0
      print(f"\r       {done}/{len(actions)} tested  |  {len(results)} ok  "
            f"{compile_fails} compile-fail  {run_fails} run-fail  "
            f"[{elapsed:.1f}s]", end="", flush=True)

  elapsed = time.perf_counter() - t0
  print(f"\n       Done in {elapsed:.1f}s\n")

  if not results:
    print("ERROR: No variants completed successfully!")
    return 1

  # 5. rank & display ------------------------------------------------------
  print("[5/5] Ranking variants by predicted cycle count …\n")
  results.sort(key=lambda r: r["cycles"])

  hdr = f"{'Rank':>4s}  {'Pred Cycles':>12s}  {'UOps':>5s}  {'Optimisations'}"
  sep = "—" * 80

  def print_table(title, rows, start_rank):
    print(sep)
    print(f"  {title}")
    print(sep)
    print(hdr)
    print(sep)
    for i, r in enumerate(rows):
      rank = start_rank + i
      print(f"{rank:4d}  {r['cycles']:12,d}  {r['uop_count']:5d}  {fmt_opts(r['opts'])}")
    print()

  n_show = min(5, len(results))
  print_table("TOP-5 FASTEST (predicted)", results[:n_show], 1)
  if len(results) > n_show:
    print_table("BOTTOM-5 SLOWEST (predicted)", results[-n_show:], len(results) - n_show + 1)

  # summary
  best, worst = results[0], results[-1]
  print(sep)
  print("  SUMMARY")
  print(sep)
  print(f"  Total variants tested : {len(results)}")
  print(f"  Compile failures      : {compile_fails}")
  print(f"  Run failures          : {run_fails}")
  print(f"  Best  predicted cycles: {best['cycles']:>12,d}  {fmt_opts(best['opts'])}")
  print(f"  Worst predicted cycles: {worst['cycles']:>12,d}  {fmt_opts(worst['opts'])}")
  if best["cycles"] > 0:
    print(f"  Speedup range         : {worst['cycles'] / best['cycles']:.1f}×")
  print(sep)
  return 0


if __name__ == "__main__":
  sys.exit(main())
