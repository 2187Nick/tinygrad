"""Tests for BEAM_EMU=1 path in tinygrad/codegen/opt/search.py.

Run with:
  DEV=AMD MOCKGPU=1 PYTHON_REMU=1 PROFILE=1 SQTT=1 \
    PYTHONPATH=. .venv/bin/python -m pytest test/test_beam_emu.py -v
"""
import os, math, unittest

REQUIRED_ENV = {"DEV": "AMD", "MOCKGPU": "1", "PYTHON_REMU": "1", "PROFILE": "1", "SQTT": "1"}


@unittest.skipUnless(all(os.environ.get(k) == v for k,v in REQUIRED_ENV.items()),
                     f"requires env: {REQUIRED_ENV}")
class TestBeamEmu(unittest.TestCase):

  def test_emulate_program_cycles_returns_finite_seconds(self):
    """_emulate_program_cycles dispatches through MOCKGPU and returns predicted seconds."""
    from dataclasses import replace
    import numpy as np
    from tinygrad import Tensor, Device
    from tinygrad.codegen.opt.search import _emulate_program_cycles, _ensure_buffer_alloc
    from tinygrad.codegen.opt.postrange import Scheduler, bufs_from_ast
    from tinygrad.codegen import get_program
    from tinygrad.uop.ops import Ops

    a = Tensor(np.random.rand(32, 32).astype(np.float32)).realize()
    b = Tensor(np.random.rand(32, 32).astype(np.float32)).realize()
    sched = (a @ b).contiguous().schedule()
    ast = next(ei.ast for ei in reversed(sched) if ei.ast.op == Ops.SINK)

    s = Scheduler(ast, Device[Device.DEFAULT].renderer)
    s.convert_loop_to_global()
    p = get_program(s.copy().get_optimized_ast(name_override="test"), s.ren)
    lib = p.lib if p.lib is not None else Device[Device.DEFAULT].compiler.compile(p.src)

    rawbufs = _ensure_buffer_alloc(bufs_from_ast(ast, Device.DEFAULT))
    var_vals = {k.expr: int(k.vmax+k.vmin)//2 for k in ast.variables()}

    pred_seconds = _emulate_program_cycles(p, lib, var_vals, rawbufs)
    self.assertTrue(math.isfinite(pred_seconds))
    self.assertGreater(pred_seconds, 0)
    self.assertLess(pred_seconds, 1.0)  # 32x32 matmul should be sub-second predicted

  def test_time_program_routes_through_emu_when_beam_emu_set(self):
    """_time_program with BEAM_EMU=1 returns cnt copies of the same emu cycle estimate."""
    import tinygrad.codegen.opt.search as S
    from dataclasses import replace
    import numpy as np
    from tinygrad import Tensor, Device
    from tinygrad.codegen.opt.postrange import Scheduler, bufs_from_ast
    from tinygrad.codegen import get_program
    from tinygrad.uop.ops import Ops

    a = Tensor(np.random.rand(16, 16).astype(np.float32)).realize()
    b = Tensor(np.random.rand(16, 16).astype(np.float32)).realize()
    sched = (a @ b).contiguous().schedule()
    ast = next(ei.ast for ei in reversed(sched) if ei.ast.op == Ops.SINK)
    s = Scheduler(ast, Device[Device.DEFAULT].renderer); s.convert_loop_to_global()
    p = get_program(s.copy().get_optimized_ast(name_override="test"), s.ren)
    lib = p.lib if p.lib is not None else Device[Device.DEFAULT].compiler.compile(p.src)
    rawbufs = S._ensure_buffer_alloc(bufs_from_ast(ast, Device.DEFAULT))
    var_vals = {k.expr: int(k.vmax+k.vmin)//2 for k in ast.variables()}

    saved = S.BEAM_EMU
    try:
      S.BEAM_EMU = 1
      tms = S._time_program(p, lib, var_vals, rawbufs, cnt=3)
    finally:
      S.BEAM_EMU = saved

    self.assertEqual(len(tms), 3)
    self.assertEqual(tms[0], tms[1])
    self.assertEqual(tms[1], tms[2])
    self.assertTrue(math.isfinite(tms[0]))


if __name__ == "__main__":
  unittest.main()
