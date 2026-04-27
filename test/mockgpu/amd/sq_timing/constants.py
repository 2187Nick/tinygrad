"""RDNA3 SQ cycle-accurate timing constants.

Single source of truth for the per-wave timing parameters used by
`test/mockgpu/amd/emu.py::_simulate_sq_timing`. Per EMU_REWRITE_DESIGN.md §3,
these were previously scattered `_XXX` module-level constants in emu.py; they
now live in a single frozen `TimingConstants` dataclass exposed as `CONST`.

Behaviour is unchanged — every value equals the pre-refactor constant. Each
field's docstring comment links it to the microbenchmark that would calibrate
it (when such a probe exists; see extra/sqtt/rgp/PROBE_FINDINGS.md and
MICROBENCH_TAXONOMY.md).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class TimingConstants:
  # ── LDS ──────────────────────────────────────────────────────────────────
  LDS_RD_LATENCY: int = 31           # probe_ds_rd_latency
  LDS_WR_LATENCY: int = 33           # probe_ds_wr_latency
  LDS_SERVICE_COST: int = 6          # LDS unit busy per DS op
  LDS_B128_EXTRA: int = 5            # b128 loads (4 dwords) extra LDS latency (layernorm [18] diff=-5)
  LDS_B128_VGPR_STAGGER: int = 17    # upper 2 VGPRs of serialized b128 ready +17cy (layernorm [26])
  LDS_B128_RD_SERVICE: int = 19      # consecutive b128 reads serialized (layernorm pair1/pair2)
  VALU_DS_WR_FORWARD: int = 26       # VALU→DS_WR forwarding stall (reference PKL; HW=22)
  VALU_DS_RD_FORWARD: int = 22       # VALU→DS_RD forwarding stall

  # ── VMEM ─────────────────────────────────────────────────────────────────
  VMEM_LATENCY: int = 300            # global_load / global_store round-trip
  VMEM_DRAIN_CYCLES: int = 15        # SQ holds s_nop/s_endpgm until VMEM accepted (HW=15)
  VMEM_EXEC_MIN: int = 8             # min VMEM execution after forwarding overlap
  VALU_VMEM_WR_FORWARD: int = 21     # VALU→VMEM_WR forward (b32 base; b128 varies)
  VALU_VMEM_WR_BYPASS: int = 4       # inter-wave VMEM_WR overlap: 21→17cy (data_deps, probe_branch_cost)
  VALU_VMEM_ADDR_FORWARD: int = 27   # VALU→VMEM address VGPR forward (lds_sync=27)
  VALU_VMEM_RD_FORWARD: int = 22     # VALU→VMEM_RD forward
  VALU_VMEM_RD_BYPASS: int = 4       # inter-wave VMEM_RD bypass savings (22→18) — mirrors WR_BYPASS

  # ── Trans pipe ───────────────────────────────────────────────────────────
  TRANS_PIPE_CYCLES: int = 4         # trans→trans occupancy; trans→VALU=1
  TRANS_PIPELINE_LATENCY: int = 27   # v_exp/v_log/v_rcp — depctr waits for this
  TRANS_PIPELINE_LATENCY_SQRT: int = 31  # v_sqrt/v_rsq longer latency (depctr after v_sqrt = L-6=25)

  # ── SGPR / scoreboard ────────────────────────────────────────────────────
  SGPR_LATENCY: int = 4              # VALU SGPR write-to-read latency
  CNDMASK_SGPR_LATENCY: int = 4      # v_cndmask sgpr source standard SGPR latency
  CMP_LIT_WB_LATENCY: int = 5        # LIT-source v_cmp SGPR WB (answer.md completion buffer)
  SGPR_COMMIT_GAP: int = 2           # LIT v_cmp commit port serialization

  # ── VOPD ─────────────────────────────────────────────────────────────────
  VOPD_PIPE_CYCLES: int = 4          # probe_vopd_* — consecutive VOPDs +4cy (dep-aware gating TBD)

  # ── SMEM ─────────────────────────────────────────────────────────────────
  SMEM_LATENCY: int = 200            # s_load_b* round-trip

  # ── Barrier / wave launch ────────────────────────────────────────────────
  BARRIER_FROM_LAST: int = 6         # cycles from last wave's barrier issue to release
  WAVESTART_GAP: int = 1             # per-wave stagger at launch
  # NOTE: HW measurement (expansion capture, 29 kernels, stdev≈0) says 2cy/wave —
  # i.e. 30cy span across 16 waves. Bumping to 2 was tested and had neutral score
  # impact (Ref strict -1, MB strict +1) because the gap shifts only the WAVESTART
  # event, not the [5]-token LINEAR slope which needs a queue-propagation model.
  FIRST_INST_GAP: int = 2            # first instruction gap after wavestart

  # ── Exec write ───────────────────────────────────────────────────────────
  EXEC_WRITE_LATENCY: int = 24       # v_cmpx→EXEC→s_cbranch_execz propagation (layernorm)

  # ── VALU (no separate VALU_*_LATENCY scalars exist pre-refactor; kept
  #         as a named field for Step 5 extraction without changing any
  #         current behaviour) ─────────────────────────────────────────────
  # NOTE: VALU_PIPELINE_LATENCY is referenced in EMU_REWRITE_DESIGN §3 but is
  # NOT a module-level constant in emu.py today — its value lives inside
  # `_INSTID_BASE_STALLS`. Intentionally omitted here to preserve zero-diff.


CONST = TimingConstants()
