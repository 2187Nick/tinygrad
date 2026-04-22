# Stopping at 94% MODAL: what it means, and how we cash it in

Context: RDNA3 cycle-accurate emulator bounty, current standing 94.6% MODAL /
79.7% strict as of 2026-04-20. This note explains the scoring, makes the case
that 94% MODAL is a reasonable stopping point, and lays out how to turn it
into actual tinygrad speedup via BEAM search.

---

## 1. What "MODAL" means

See `extra/sqtt/rigorous_hw_test.py:398-424` for the authoritative definition.
Scoring runs in two modes:

- **Strict.** For each instruction position `j` in a wave, compare emu's
  cycle-delta to *that specific HW wave's* delta. Exact match required.

- **MODAL.** For each position `j`, collect *the set of deltas observed across
  ALL HW waves at that same position* (wave 0, wave 1, wave 2, …). Emu
  "matches" if its predicted delta equals *any* of those observed values.

The gap exists because real hardware is stochastic — wave 0 and wave 1 running
the exact same instruction at the same PC often differ by a few cycles (cache
state, issue-queue contention, SIMD scheduling jitter). Strict demands the
emulator reproduce one specific run exactly; MODAL asks the weaker (but more
meaningful) question: "did the emulator predict something the hardware actually
does?"

Today's numbers translated:

| Score       | Meaning                                                              |
|-------------|----------------------------------------------------------------------|
| MODAL 94.6% | In 94.6% of slots, emu's prediction is one of the values HW produced |
| Strict 79.7%| In 79.7% of slots, emu matches this specific HW wave exactly          |

The ~15 pt gap between them is approximately the **stochastic ceiling** —
natural cycle jitter between identical HW waves that no deterministic emulator
can collapse. See `extra/sqtt/rgp/MISMATCH_CATEGORIES.md` and
`extra/sqtt/STOCHASTIC_SCHEDULER_PLAN.md` for the full discussion.

## 2. Why 94% MODAL is a fine stopping point

Getting from 94 → 98 MODAL buys diminishing-returns accuracy against a
fundamentally noisy ground truth. What we *don't* need for downstream uses is
absolute accuracy — we need **correct ordering**. Two kernels scored by the
emulator only need to rank in the same order as HW would, not match HW to the
cycle. MODAL scoring is already distribution-matching, so emu predictions tend
to land near the HW mode, which is exactly where ranking stays stable.

The big remaining unlocks identified in `rgp_rga.md`:
- **(a)** Enable `SQ_TT_TOKEN_PERF` — projected ~+1500 strict tokens (~2%)
- **(b)** Static VGPR bank analysis — projected ~+250 strict tokens

Worth doing, but not required to turn the emulator into a useful cost model.

## 3. The cash-in: BEAM search cost model

The payoff isn't running inference on the emulator (too slow, wrong purpose).
It's **replacing the cost model inside BEAM autotuning**.

### Today: BEAM times candidates on real HW

See `tinygrad/codegen/opt/search.py:121-192`.

- `beam_search()` walks candidate kernel optimizations (UPCAST, UNROLL, LOCAL,
  TC, SWAP, THREAD, …).
- For each candidate it calls `_time_program()` (line 164), which:
  1. Compiles the candidate (`compiler.compile`)
  2. Uploads to the GPU
  3. Runs it 3× with L2 flushes between runs
  4. Returns the minimum time

For BEAM=8 over a 50-kernel model, that's hundreds-to-thousands of HW
roundtrips per compile. Every candidate pays a GPU sync cost.

### Tomorrow: BEAM scores candidates on the emulator

- Compile once to ISA (still needed for the emulator to see the instruction
  stream).
- Run the cycle emulator on the ISA → ~ms per candidate, no GPU sync, no L2
  flush.
- BEAM picks by predicted cycles.

### Why 94% MODAL is enough

BEAM only needs correct **ordering** of candidates. If emu predicts A=1000cy
and B=1200cy, it doesn't matter that HW actually runs A=1040 — we still pick
A. MODAL is already distribution-matching, and ordering stability is much
cheaper than absolute accuracy.

### Integration sketch

```python
# tinygrad/codegen/opt/search.py (modification)
def _time_program(p, lib, var_vals, rawbufs, ...,
                  use_emu=getenv("BEAM_EMU", 0)):
    if use_emu and p.device.startswith("AMD"):
        return [emulate_kernel_cycles(p, lib)] * cnt   # deterministic → 1 val
    # ...existing HW path unchanged
```

Then `BEAM=8 BEAM_EMU=1 python your_model.py` uses the emulator as scorer
while keeping the rest of BEAM's machinery (diskcache, early-stop,
`BEAM_UOPS_MAX`) untouched.

### What speedup shows up where

| Phase                           | Before                     | After                   | Who notices                                  |
|---------------------------------|----------------------------|-------------------------|----------------------------------------------|
| Cold-start / JIT compile        | Dominated by HW timing     | ~5–20× faster BEAM      | First inference latency; CI speedruns        |
| BEAM depth at fixed budget      | BEAM=8 × 5s                | BEAM=32 × 5s            | Better kernels → *steady-state* is faster    |
| Steady-state inference/training | N/A (no BEAM in hot path)  | Unchanged               | Nobody — emu isn't called at runtime         |

The interesting number for the bounty pitch isn't "we hit 94% accuracy" — it's
**"BEAM_EMU finds a kernel within X% of BEAM_HW's best, in 1/Nth the autotune
wall-clock."** That's a speedup users feel.

## 4. Proof plan — three experiments to run once BEAM_EMU is wired in

1. **Ranking accuracy.** For ~50 real kernels (resnet, GPT-2,
   stable-diffusion), compare emu-BEAM top-pick vs HW-BEAM top-pick.
   Measure: how often do they agree, and when they disagree, how much slower
   is the emu-pick on HW?
2. **Time-to-compile.** End-to-end `python examples/stable_diffusion.py`
   (compile-only) with and without `BEAM_EMU=1`. Expected: minutes → seconds.
3. **Deeper-search win.** Fix wall-clock budget (say 10 s). Let BEAM_HW use
   BEAM=8; let BEAM_EMU use BEAM=32. Measure final kernel runtime on HW.
   Hypothesis: deeper search beats shallower search, net.

## 5. Secondary uses (not the bounty hook, but flagged)

- **CI regression detection.** Perf regressions catchable in CI without a GPU.
- **Cross-device prediction.** With per-arch timing rules, predict 7900 XTX
  performance from a non-AMD dev machine.
- **Scheduler feedback.** Cost model inside fusion / tiling decisions, not just
  post-hoc BEAM.
- **Kernel explainability.** The viewer in `extra/sqtt/viewer/` renders the
  same cycle data the emulator uses — human-inspectable mismatches when
  something surprising happens.

## 6. Bottom line

94% MODAL is not the finish line for academic accuracy, but it's past the
threshold where the emulator stops being a study target and starts being a
**tool**. The next commit worth making isn't +2% accuracy — it's
`BEAM_EMU=1`.
