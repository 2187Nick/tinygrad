# Overnight team handoff — RDNA3 SQTT timing bounty

## Current state (as of 2026-04-19 20:30)

- **Total strict: 54830/69860 (78.5%)** — this is the bounty metric
- **Total MODAL: 65259/69860 (93.4%)**
- **Reference (10 custom kernels + 4 probes): 339/340 MODAL, 327/340 strict**
- **11/11 SQTT profiler + custom-kernel tests pass**
- 403 kernels across A/B/C/D/E/F/G in `extra/sqtt/rgp/batch_*.py`
- HW captures in `extra/sqtt/captures/rigorous/*.pkl`
- Capture logs preserved under `extra/sqtt/batch_logs/`

## How to validate + iterate (no GPU needed)

Compare emu vs HW captures with MOCKGPU — fully reproducible, runs in ~30s:

```bash
# MODAL (default)
DEV=AMD MOCKGPU=1 PYTHON_REMU=1 PROFILE=1 SQTT=1 MICROBENCH=1 PYTHONPATH=. \
  .venv/bin/python extra/sqtt/rigorous_hw_test.py --compare

# strict (the bounty target)
DEV=AMD MOCKGPU=1 PYTHON_REMU=1 PROFILE=1 SQTT=1 MICROBENCH=1 MODAL=0 PYTHONPATH=. \
  .venv/bin/python extra/sqtt/rigorous_hw_test.py --compare

# bounty tests (fast)
DEV=AMD MOCKGPU=1 PYTHON_REMU=1 PROFILE=1 SQTT=1 PYTHONPATH=. \
  .venv/bin/python -m pytest test/amd/test_sqtt_profiler.py test/amd/test_custom_kernel.py -q
```

When a rule is wrong, the CURRENT comparison in `--compare` mode prints the
offending `[idx] HW=X EMU=Y diff=Z` lines for the first 25 mismatches per wave.
That is the fastest debug loop.

## Overnight team options (pick any — all are independent)

### Option A — SIMD arbiter rewrite (biggest potential ROI, high risk)

**Why:** confirmed via MGPUSim scan that RDNA3 CUs have 4 SIMDs with
round-robin VALU issue arbitration. Our current heuristic rules (wave 0
bypass, depth≥4 stall, long-chain≥6 stall) *approximate* this but miss
the multi-chain queue-pressure cases (`mb_f2_raw_all_banks_n4`,
`mb_vmem_store_b32_chain_n4`, `mb_f2_vopd_then_raw`).

**Failed attempt today (see task #42 in commit history):** naive
wave→SIMD=`i%4` + 1cy/SIMD issue regressed 24K strict tokens — VOPD
chains pipeline at 1cy in HW but arbiter forced 4-way serialization.

**Correct model (what to build):**
1. Shadow-wire a `simd_valu_avail[4]` tracker (we have the stub in
   `emu.py` line 297).
2. Per-SIMD **oldest-wave priority** — the oldest wave on a SIMD
   monopolizes the VALU port while it has instructions ready. Other
   waves on the same SIMD wait until that wave's chain drains (defined
   as: it hits a non-VALU, or its `ready[i]` exceeds its peers by some
   threshold).
3. **VOPD is dual-issue at SIMD level** — one VOPD = one VALU+one VALU
   slot, same cycle (do NOT serialize VOPD chains to 4cy).
4. Validate against:
   - `mb_valu_add_n16` (wave 0 dt=1, waves 1-15 dt=5) — current 91% strict
   - `mb_f2_raw_all_banks_n4` (42.8% strict) — multi-chain
   - `mb_vopd_chain_n4_raw` (don't regress, currently 78%)
5. Size: probably 2-4 hours of careful work. Biggest ROI block left.

### Option B — VMEM store pipe contention (medium ROI, clean scope)

**Why:** HW shows VALU-after-VMEM_WR costs 8-9cy but our EMU produces
dt=1. Pattern: store → VALU → store → VALU… the VALU in between waits
for store to drain. `mb_vmem_store_b32_chain_n4` strict is 52% → could
gain 50-80 tokens.

**Evidence:** `extra/sqtt/captures/rigorous/mb_vmem_store_b32_chain_n4.pkl`,
`mb_f3_store_pair_then_pair.pkl`, `mb_f3_store_spaced_by_raw_chain.pkl`.

**Approach:** after a `vmem_wr` issue, set a `post_vmem_wr_ready[i]`
deadline. The next VALU on the same wave waits for it.

Estimated impact: +200-500 strict.

### Option C — SALU RAW +1cy chain (small, cheap)

**Why:** HW `mb_g4_s_add_u32_n8` shows SALU chain wave 0 dt=1, waves 1+
dt=2. Our EMU produces dt=1 uniformly.

**Blocker:** SALU decode path does NOT extract sgpr_r / sgpr_w — it
emits with `extra=None` at `test/mockgpu/amd/emu.py:1279`. Would need
to extend SALU decode to populate reg info (like VALU already does),
then add a stall rule analogous to the existing VALU wave-credit.

Estimated impact: +50-150 strict across all mb_g4_* and some existing
SALU chains.

### Option D — Trans chain wave ≥7 stagger for SHORT chains (n=2-4)

**Why:** HW `mb_f4_*_chain_n4` (rcp/log/sqrt/rsq/exp) shows waves 7+
paying +6cy (dt=10) on chain continuations, even for chains of length
4. Our current rule only fires for chains ≥ 6.

**Approach:** loosen the long-chain gate for trans chain. Extend from
`length≥6 AND wave≥4` to `length≥4 AND wave≥7`. See `emu.py:735-745`.

Estimated impact: +100-200 strict.

### Option E — More targeted HW batches (no advantage without GPU)

If a real 7900 XTX becomes available, a targeted **Batch H** could
probe: image ops (not yet tested), MFMA (if supported), atomic VMEM
(global_atomic_add), cross-lane with specific lane masks, wave32 vs
wave64 comparisons. ~30-40 kernels. Need GPU.

### Option F — Integrate the RGP oracle (once any team member has RGP)

We already have `extra/sqtt/rgptool.py` (347 lines) that reads/writes
`.rgp` files. If any team member has RGP installed on Windows/Mac, we
can:
1. Write a `rigorous_hw_test.py --export-rgp KERNEL_NAME` flag that
   takes one of our .pkl captures and emits a .rgp file.
2. Open in RGP; AMD's official "Instruction Timing" view breaks down
   per-instruction stall reasons (memory, scoreboard, bank conflicts,
   issue arbitration).
3. Use those stall reasons to model specific mismatches.

Not strictly needed but is the fastest way to decode the hard cases.

## What's known but unfixed (for reference)

1. `data_deps` wave 1 at [6] has HW=17 EMU=21 — 4cy too slow store.
   Pre-existing from session start; not introduced by any rule this
   session. Investigation punted.
2. `exp_chain` 111/112 strict (1 token off) — mentioned in prior
   SESSION_STATUS as `[56]` or `[57]` edge case on wave 0.
3. `mb_f4_*_chain_n2` short trans chains (all showing wave 7+ stall
   at [6] but not at [7]) — bimodal pattern, current rule doesn't
   fire for n<6 chains.

## Files the overnight team needs

All pushed to `origin/master` as of the last commit. Relevant paths:

- **emu:** `test/mockgpu/amd/emu.py` (single 3300+ line file with the
  entire timing model). State trackers in `test/mockgpu/amd/sq_timing/`.
- **microbench suite:** `extra/sqtt/rgp/batch_[a-g]*.py` + registry in
  `extra/sqtt/rgp/microbench.py`.
- **HW captures:** `extra/sqtt/captures/rigorous/*.pkl` (403 files).
  Everything gets compared via `rigorous_hw_test.py --compare`.
- **capture logs:** `extra/sqtt/batch_logs/batch_{f,g}_hw_capture.log`.
- **this handoff:** `extra/sqtt/HANDOFF_OVERNIGHT.md`.
- **session changelog:** `extra/sqtt/SESSION_STATUS.md` (numbers),
  `extra/sqtt/STOCHASTIC_SCHEDULER_PLAN.md` (wave-variance design).

## Running the compare for regressions before pushing

Always run BOTH `--compare` modes and the pytest before pushing a change:

```bash
for MODAL_ENV in "" "MODAL=0"; do
  echo "=== $MODAL_ENV ==="
  eval "$MODAL_ENV DEV=AMD MOCKGPU=1 PYTHON_REMU=1 PROFILE=1 SQTT=1 \
    MICROBENCH=1 PYTHONPATH=. .venv/bin/python \
    extra/sqtt/rigorous_hw_test.py --compare 2>&1" | grep "^  TOTAL"
done
DEV=AMD MOCKGPU=1 PYTHON_REMU=1 PROFILE=1 SQTT=1 PYTHONPATH=. \
  .venv/bin/python -m pytest test/amd/test_sqtt_profiler.py \
  test/amd/test_custom_kernel.py -q
```

Acceptance bar: strict ≥ 54830, MODAL ≥ 65259, reference held, 11 tests.

## Commit history (session highlights)

```
477d9d811 docs: SESSION_STATUS — Batch G totals + SIMD-arbiter lessons
1ec9dc13c Batch G: 40 gap-fill probes (SMEM/DS/hard-ops/SALU/f16)
32a201d04 emu: v_mul_lo_u32 / v_mul_hi_u32 are 4cy pipelined (+88 strict)
d4a800ad3 emu: s_nop(0) drain propagation (+528 strict)
e073fc350 emu: trans-chain per-wave stagger for long chains (+160 strict)
3a8e95315 emu: tune long-chain threshold 10→6 (+518 strict)
9a183f24d emu: FMAC/MAC accumulator implicit read (+389 strict)
b9e4da59f emu: look-ahead long-RAW-chain detection (+879 strict)
35ccd0438 Batch F: 97 deep-coverage probes (GPU-safe VMEM patterns)
91561e1cf emu: wave-credit RAW stall (depth≥4, wave≥1) + Batch E HW probes
```
