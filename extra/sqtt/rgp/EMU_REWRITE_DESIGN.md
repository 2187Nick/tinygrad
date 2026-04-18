# EMU Rewrite Design — Decoupled RDNA3 SQ Scheduler

**Target file:** `test/mockgpu/amd/emu.py` → replace `_simulate_sq_timing` (currently lines 195–850, ~655 LoC) and the module-level constants block (lines 126–157).

**Goal:** rearchitect the monolithic per-wave timing loop into a set of independent timing subsystems so that each outstanding mismatch class (the 28 tokens grouped into patterns A–F in `MISMATCH_ANALYSIS.md`) can be fit independently against its own microbenchmark batch, without one subsystem's fix re-balancing wrong elsewhere.

**Acceptance criterion:** regression harness never drops below 293/321 exact and 309/321 ±2 at any landed commit.

---

## 0. Motivation (what's wrong with the current loop)

`_simulate_sq_timing` at `test/mockgpu/amd/emu.py:195` threads ~30 shared per-wave state arrays through one loop. A single instruction handler (e.g. VMEM store at `emu.py:587–606`) reads:

- `valu_vmem_wr_deadline`, `valu_vmem_wr_set_time`, `valu_vmem_wr_slow_ext`, `vgpr_write_time`
- `consecutive_selffwd_vgprs`, `consecutive_vgprs_written`
- `_vmem_wr_bypass_active(i)` which itself closures over `wave_done`, `at_barrier`, `pc`, `ready`, four `valu_*_deadline` arrays, and `vmem_drain_deadline`.

And it writes: `vm_pend`, `vmem_drain_deadline`, `valu_vmem_wr_slow_ext`, implicitly through `_vmem_fwd_stall`.

Three failed attempts in the 2026-04-18 session (see `SESSION_STATUS.md` lines 234–238) each fixed one mismatch and broke 2–80 elsewhere — not because the hypotheses were wrong, but because the state crosstalk makes it impossible to isolate one rule without perturbing others. The s_nop handler at `emu.py:351–404` alone touches `last_drain_stamp`, `nop_start`, `nop_stamp`, `vmem_drain_deadline`, `sgpr_write_time`, `valu_vmem_wr_deadline`, `valu_vmem_rd_deadline`, `vgpr_write_time`, `burst_wave`, `burst_exclusive`.

Decoupling is the path forward per `PROBE_FINDINGS.md` §"Emu-fix status": *"the realistic path forward is a more structured rewrite … that separates the overlapping effects, not more constant-tuning."*

---

## 1. Subsystem decomposition

Eight subsystems, each owning its state and exposing a small interface. The scheduler becomes a thin orchestrator that asks each subsystem "when can this instruction issue?" and then notifies each subsystem "this instruction issued at cycle X".

All subsystems are **per-wave** unless explicitly marked CU-shared.

### 1.1 `TimingConstants` (dataclass, single global instance)

Module-level `CONST: TimingConstants` replaces the scattered `_LDS_RD_LATENCY`/etc. constants at `emu.py:126–157`. See §3.

### 1.2 `VAluPipe` (per-wave, non-trans VALU + VOPD)

**Owns:** last 4 non-trans VALU issue cycles (`issue_hist`), last VOPD issue cycle, VOPD pipe-available cycle, VOPD bank-write-times (4-bank array), consecutive-single-VALU run length, consecutive-selffwd/written VGPR chain counters, VGPR scoreboard, VGPR write times.

**Interface:**

```python
class VAluPipe:
    def valu_dep_stall(self, issue_cycle: int, instid: int) -> int: ...
    def vgpr_read_stall(self, read_regs: Iterable[int], issue_cycle: int) -> int: ...
    def vopd_ready(self) -> int: ...
    def vopd_bank_port_stall(self, issue_cycle: int, read_regs: Iterable[int], is_vopd_lit: bool) -> int: ...
    def slow_fresh_consume(self, issue_cycle: int, read_regs: Iterable[int]) -> bool: ...
    def on_valu_issue(self, cycle: int, *, is_vopd: bool, is_vopd_lit: bool,
                      write_regs: tuple[int, ...], read_regs: tuple[int, ...],
                      is_vopc: bool) -> None: ...
    def consecutive_single_run(self) -> int: ...
    def vgpr_write_time(self, reg: int) -> int: ...
```

### 1.3 `TransPipe` (per-wave, trans ALU)

**Owns:** trans-pipe-available cycle, trans-VGPR readiness map, scalar-after-trans-ready.

**Interface:**

```python
class TransPipe:
    def trans_ready(self) -> int: ...
    def trans_read_stall(self, read_regs: Iterable[int]) -> int: ...
    def scalar_ready(self) -> int: ...
    def depctr_drain(self) -> int: ...
    def on_trans_issue(self, cycle: int, *, trans_name: str, write_regs: tuple[int, ...]) -> None: ...
    def on_depctr(self, cycle: int) -> None: ...
```

### 1.4 `ScalarPipe` (per-wave, SALU + s_cbranch + s_nop)

**Owns:** scalar issue phase (new — 4-cycle beat per `MISMATCH_ANALYSIS.md` §E), last-scalar-was-nop flag, last-drain-stamp, cold-path cbranch memory.

**Interface:**

```python
class ScalarPipe:
    def scalar_issue_cycle(self, ready: int) -> int: ...
    def salu_cost(self, op_kind: ScalarKind, ready: int) -> tuple[int, int]: ...
    def cbranch_cost(self, taken: bool, *, scc_ready: int, tight: bool) -> tuple[int, int]: ...
        # HW-confirmed: tight=8, cold=13 for NOT-TAKEN (PROBE_FINDINGS.md §E)
    def snop_cost(self, n: int, *, ready: int, last_drain_stamp: int,
                  sgpr_drain: int, vmem_drain: int,
                  predecessor_kind: PredKind, is_last_in_chain: bool) -> tuple[int, int]: ...
        # post-vmem-drain  → 20 per nop (PROBE_FINDINGS §B)
        # middle of chain  → 16
        # last-before-valu → 20 (IB resume +4)
        # isolated post-vcmp → 15+1 + sgpr_drain
    def on_scalar_issue(self, cycle: int, *, kind: ScalarKind, writes_scc: bool) -> None: ...
    def on_nop_issue(self, stamp: int, n: int, *, in_chain: bool) -> None: ...
    def last_drain_stamp(self) -> int: ...
    def scc_write_time(self) -> int: ...
    def exec_write_time(self) -> int: ...
```

### 1.5 `SgprScoreboard` (per-wave)

**Owns:** `sgpr_write_time[reg] → cycle`, LIT v_cmp completion buffer, SMEM sgpr-ready.

**Interface:**

```python
class SgprWriterKind(Enum): VALU_STANDARD; VALU_CNDMASK_COND; VALU_LIT_VCMP; SMEM; SALU
class SgprReaderKind(Enum): STANDARD; CNDMASK_NONVCC; BRANCH_SCC

class SgprScoreboard:
    def read_stall(self, reg: int, issue_cycle: int, reader: SgprReaderKind) -> int: ...
    def write(self, reg: int, cycle: int, writer: SgprWriterKind) -> None: ...
    def pending_nonvcc_drain(self, now: int) -> int: ...
    def on_depctr(self, cycle: int) -> None: ...
    def on_non_litcmp_valu(self) -> None: ...
    def smem_return(self, regs: Iterable[int], ready_cycle: int) -> None: ...
    def prune(self, stall_until: int) -> None: ...
```

### 1.6 `VmemPipe` (per-wave; shared CU pressure checked via `WaveScheduler.peer_states`)

**Owns:** VALU→VMEM_WR/RD/addr forwarding deadlines, VMEM drain deadline, VMEM write arbiter "warm" state, vm_pend queue, VMEM scoreboard warm marker.

**Interface:**

```python
class VmemPipe:
    def store_issue_cycle(self, ready: int, *, store_vgprs: int,
                          addr_reg_time: int,
                          peer_states: PeerSnapshot) -> int: ...
    def load_issue_cycle(self, ready: int) -> int: ...
    def on_valu_write(self, cycle: int, *, write_regs: tuple[int, ...], read_regs: tuple[int, ...],
                      is_trans: bool, is_vopc: bool, slow_fresh: bool) -> None: ...
    def on_store_issue(self, cycle: int, *, fwd_stall_nonzero: bool) -> None: ...
    def on_load_issue(self, cycle: int) -> None: ...
    def on_nop_cap(self, stamp: int) -> None: ...
    def vmem_drain_deadline(self) -> int: ...
    def vm_pending(self) -> list[int]: ...
    def prune(self, stall_until: int) -> None: ...
```

### 1.7 `LdsPipe` (CU-shared + per-wave LDS deadlines)

**Interface:**

```python
class LdsPipe:
    def ds_write_issue(self, wave: int, ready: int) -> int: ...
    def ds_read_issue(self, wave: int, ready: int, *, b128: bool) -> int: ...
    def on_ds_write_issue(self, wave: int, cycle: int) -> None: ...
    def on_ds_read_issue(self, wave: int, cycle: int, *,
                         b128: bool, dest_base: int | None) -> None: ...
    def lgkm_pending(self, wave: int) -> list[int]: ...
    def prune(self, wave: int, stall_until: int) -> None: ...
```

### 1.8 `WaveScheduler` (orchestrator) + `IbFetch` (per-wave)

**`WaveScheduler` owns:** round-robin `rr` index, `prev_issue_cycle`, `prev_wave`, VALU burst state, global `clock`, `wave_done`, `at_barrier`, `barrier_issue`, `pc`.

**Exposes `peer_states(i) -> PeerSnapshot`** — a read-only snapshot of every other wave's state. Replaces in-place reads inside `_vmem_wr_bypass_active`.

**`IbFetch` per-wave:**

```python
class IbFetch:
    def set_drain(self, stamp: int) -> None: ...
    def mark_nop_in_chain(self) -> None: ...
    def resume_penalty(self) -> int: ...
    def last_nop_before_valu_extra(self) -> int: ...
    def on_non_nop_issue(self) -> None: ...
```

---

## 2. Per-instruction handlers

| Inst class | Queries (issue_cycle = max of …) | Updates (after issue) | Stamp rule |
|---|---|---|---|
| **VALU non-trans (non-VOPD)** | `VAluPipe.valu_dep_stall`, `VAluPipe.vgpr_read_stall`, `SgprScoreboard.read_stall` (each src SGPR), `TransPipe.trans_read_stall`, `IbFetch.resume_penalty` | `VAluPipe.on_valu_issue`, `SgprScoreboard.write` per SGPR dst, `VmemPipe.on_valu_write`, `IbFetch.on_non_nop_issue` | `stamp = issue_cycle` |
| **VALU trans (v_exp/log/sqrt/…)** | `TransPipe.trans_ready`, `VAluPipe.vgpr_read_stall`, `IbFetch.resume_penalty` | `TransPipe.on_trans_issue`, `VAluPipe.on_valu_issue(is_trans=True)`, `IbFetch.on_non_nop_issue` | `stamp = issue_cycle` |
| **VOPD (non-LIT)** | `VAluPipe.vopd_ready`, `VAluPipe.vopd_bank_port_stall`, `VAluPipe.vgpr_read_stall`, `SgprScoreboard.read_stall`, `IbFetch.resume_penalty` | `VAluPipe.on_valu_issue(is_vopd=True)` | `stamp = issue_cycle` |
| **VOPD_LIT** | Same as VOPD, `pipe_cycles=1` (warm LIT chain) | Same; `on_valu_issue(is_vopd_lit=True)` | `stamp = issue_cycle` |
| **SALU (s_mov, s_cmp, …)** | `ScalarPipe.scalar_issue_cycle`, `IbFetch.resume_penalty` | `ScalarPipe.on_scalar_issue(writes_scc=…)` | `stamp = issue_cycle` |
| **s_nop** | — *handled in drain loop*; queries `IbFetch.last_drain_stamp`, `VmemPipe.vmem_drain_deadline`, `SgprScoreboard.pending_nonvcc_drain`, `ScalarPipe` predecessor kind | `IbFetch.set_drain(nop_stamp)`, `IbFetch.mark_nop_in_chain()`, `VmemPipe.on_nop_cap(nop_stamp)`, `SgprScoreboard.prune` | `stamp = nop_stamp` per `ScalarPipe.snop_cost` |
| **s_waitcnt_vm/lgkm** | `VmemPipe.vm_pending`, `LdsPipe.lgkm_pending`, `TransPipe.scalar_ready` | `IbFetch.set_drain(stall_until)`, `VmemPipe.prune`, `LdsPipe.prune`, `SgprScoreboard.prune` | `stamp = stall_until` |
| **s_waitcnt_depctr** | `TransPipe.depctr_drain` | `TransPipe.on_depctr`, `SgprScoreboard.on_depctr`, `IbFetch.set_drain(stall_until)` | `stamp = stall_until` |
| **s_cbranch** (taken) | `ScalarPipe.cbranch_cost(taken=True, …)` | `IbFetch.on_non_nop_issue` | `stamp = issue_cycle` (cost=3) |
| **s_cbranch** (not-taken) | `ScalarPipe.cbranch_cost(taken=False, tight=…)`, `SgprScoreboard.read_stall(kind=BRANCH_SCC)` | `IbFetch.on_non_nop_issue`; `ready = issue + cost (8 or 13)` | `stamp = issue_cycle + 7` (late-stamp) |
| **LDS write (ds_wr)** | `LdsPipe.ds_write_issue`, VALU→DS forwarding | `LdsPipe.on_ds_write_issue`, adds to `lgkm_pend` | `stamp = issue_cycle` |
| **LDS read (ds_rd)** | `LdsPipe.ds_read_issue(b128=…)` | `LdsPipe.on_ds_read_issue` | `stamp = issue_cycle` |
| **VMEM global_load** | `VmemPipe.load_issue_cycle` | `VmemPipe.on_load_issue`, adds to `vm_pend` | `stamp = issue_cycle` |
| **VMEM global_store** | `VmemPipe.store_issue_cycle(peer_states=…)` | `VmemPipe.on_store_issue(fwd_stall_nonzero=…)`, adds to `vm_pend` | `stamp = issue_cycle` |
| **VMEM buffer** | `VmemPipe.store_issue_cycle` or `load_issue_cycle` | same | `stamp = issue_cycle` |
| **SMEM (s_load_b64)** | `ready` (no forwarding) | `SgprScoreboard.smem_return(regs, issue+SMEM_LATENCY)`, adds to `lgkm_pend` | `stamp = issue_cycle` |
| **barrier (s_barrier)** | `ready` | sets `at_barrier[i]=True`, `barrier_issue[i]=issue`; `ready = issue + 1` | `stamp = issue_cycle` |
| **wavestart** | fixed `1 + i * WAVESTART_GAP` | `pc=1`, `ready = ws_time + FIRST_INST_GAP` | special |
| **waveend** | `ready` | `wave_done[i]=True` | `stamp = issue_cycle` |

---

## 3. Timing constants registry

```python
@dataclass(frozen=True)
class TimingConstants:
    # ── VALU / VGPR pipe ──────────────────────────────────────────────────────
    VALU_LATENCY: int = 5
    VALU_SLOW_FRESH_LATENCY: int = 9
    VALU_NO_VGPR_READ_LATENCY: int = 1

    # ── Trans pipe ────────────────────────────────────────────────────────────
    TRANS_PIPE_CYCLES: int = 4
    TRANS_LATENCY_LOG: int = 27
    TRANS_LATENCY_SQRT: int = 31
    TRANS_TO_SCALAR_GAP: int = 3

    # ── VOPD ──────────────────────────────────────────────────────────────────
    VOPD_PIPE_CYCLES: int = 4
    VOPD_LIT_PIPE_CYCLES: int = 1
    VOPD_CHAIN_DEP_CYCLES: int = 3
    VOPD_INDEP_PIPE_CYCLES: int = 1

    # ── Scalar / branch ───────────────────────────────────────────────────────
    CBRANCH_TIGHT_CYCLES: int = 8      # NEW — probe_scalar_beat_p0 (PROBE_FINDINGS §E)
    CBRANCH_COLD_CYCLES: int = 13      # NEW — probe_scalar_beat_p{1,2,3}
    S_MOV_TO_S_NOP_BEAT: int = 3       # NEW — probe_scalar_beat_p1 HW dt=3
    SCC_READ_LATENCY: int = 2

    # ── s_nop cost variants ───────────────────────────────────────────────────
    NOP_BASE_CYCLES_PER: int = 16
    NOP_AFTER_VMCNT_DRAIN: int = 20    # NEW — probe_nop_chain (PROBE_FINDINGS §B)
    NOP_LAST_IN_CHAIN_EXTRA: int = 4   # NEW — MISMATCH_ANALYSIS §B.B1
    NOP_POST_VALU_RESUME: int = 1

    # ── SGPR scoreboard ───────────────────────────────────────────────────────
    SGPR_LATENCY: int = 4
    CNDMASK_SGPR_LATENCY: int = 4
    CMP_LIT_WB_LATENCY: int = 5
    SGPR_COMMIT_GAP: int = 2

    # ── VMEM ──────────────────────────────────────────────────────────────────
    VMEM_LATENCY: int = 300
    VMEM_DRAIN_CYCLES: int = 15
    VMEM_EXEC_MIN: int = 8
    VALU_VMEM_WR_FORWARD: int = 21
    VALU_VMEM_WR_BYPASS: int = 4
    VALU_VMEM_RD_FORWARD: int = 22
    VALU_VMEM_ADDR_FORWARD: int = 27

    # ── LDS ───────────────────────────────────────────────────────────────────
    LDS_RD_LATENCY: int = 31
    LDS_WR_LATENCY: int = 33
    LDS_SERVICE_COST: int = 6
    LDS_B128_EXTRA: int = 5
    LDS_B128_VGPR_STAGGER: int = 17
    LDS_B128_RD_SERVICE: int = 19
    VALU_DS_WR_FORWARD: int = 26
    VALU_DS_RD_FORWARD: int = 22

    # ── SMEM / barrier / wave launch ──────────────────────────────────────────
    SMEM_LATENCY: int = 200
    BARRIER_FROM_LAST: int = 6
    WAVESTART_GAP: int = 1
    FIRST_INST_GAP: int = 2
    EXEC_WRITE_LATENCY: int = 24

CONST = TimingConstants()
```

**New constants exposed by HW probes** (`PROBE_FINDINGS.md`):
- `NOP_AFTER_VMCNT_DRAIN=20` (§B)
- `NOP_LAST_IN_CHAIN_EXTRA=4` (§B)
- `CBRANCH_TIGHT_CYCLES=8`, `CBRANCH_COLD_CYCLES=13` (§E)
- `S_MOV_TO_S_NOP_BEAT=3` (§E secondary)
- `VOPD_INDEP_PIPE_CYCLES=1` (§C — gated on dep-aware detection, TBD)

---

## 4. Data flow through the scheduler

### 4.1 Keep single-wave-per-cycle picking

`PROBE_FINDINGS.md` confirms 2-wave WGs always land on different SIMDs. Single-wave-per-clock model remains correct. Wave-pair shared-issuer effects (MISMATCH_ANALYSIS §D, §F.F3) stay in `WaveScheduler`, not subsystems.

### 4.2 Main loop (pseudocode)

```python
def _simulate_sq_timing(wave_events):
    wave_ids = sorted(wave_events)
    n = len(wave_ids)
    if not n: return []

    sched  = WaveScheduler(n, wave_events)
    valu   = [VAluPipe(CONST) for _ in range(n)]
    trans  = [TransPipe(CONST) for _ in range(n)]
    scal   = [ScalarPipe(CONST) for _ in range(n)]
    sgpr   = [SgprScoreboard(CONST) for _ in range(n)]
    vmem   = [VmemPipe(CONST) for _ in range(n)]
    ib     = [IbFetch(CONST) for _ in range(n)]
    lds    = LdsPipe(CONST, n_waves=n)
    timed  = []

    sched.emit_wavestart(timed)

    for _ in range(sched.max_iters()):
        if sched.all_done(): break

        # (a) Drain zero-cost events per wave.
        for i in sched.active_waves():
            _drain_zero_cost(i, sched, valu[i], trans[i], scal[i], sgpr[i],
                             vmem[i], ib[i], lds, timed)

        # (b) Pick next wave.
        i = sched.pick_next(lambda j: _effective_ready(j, valu[j], vmem[j], lds,
                                                       sched.peer_states(j)))
        if i is None:
            if not sched.try_release_barrier(): break
            continue

        # (c) Compute issue_cycle.
        inst = sched.peek(i)
        issue_cycle = _issue_cycle(i, inst, valu[i], trans[i], scal[i], sgpr[i],
                                    vmem[i], ib[i], lds, sched.peer_states(i))

        # (d) Compute stamp_cycle.
        stamp_cycle = _stamp(inst, issue_cycle, scal[i])
        timed.append((stamp_cycle, wave_ids[i], inst.pkt_cls, inst.kwargs))

        # (e) Notify subsystems in a fixed order.
        _apply_issue(i, inst, issue_cycle,
                     valu[i], trans[i], scal[i], sgpr[i], vmem[i], ib[i], lds, sched)

    return timed
```

### 4.3 PeerSnapshot

```python
@dataclass(frozen=True)
class PeerSnapshot:
    waves: tuple[PeerWaveState, ...]

@dataclass(frozen=True)
class PeerWaveState:
    wave_idx: int
    done: bool
    at_barrier: bool
    next_cat: str | None
    ready: int
    vmem_drain: int
    ds_wr_forward: int
    ds_rd_forward: int
    vmem_wr_forward: int
    vmem_rd_forward: int
```

Built once per scheduler pick.

---

## 5. Regression strategy (multi-step migration)

Rule: **every step is a commit that runs `rigorous_hw_test.py --compare` and does not drop below 293/321 exact.**

### Step 1 — Constants dataclass, no logic change
Move `_LDS_RD_LATENCY` … into `TimingConstants`. Keep aliases: `_LDS_RD_LATENCY = CONST.LDS_RD_LATENCY`. **Expected 293/321.**

### Step 2 — Extract IbFetch
Pull `last_drain_stamp`, `had_drain_nop`, +1 resume into `IbFetch`. **Expected 293/321.**

### Step 3 — Extract LdsPipe
Move `cu_lds_*`, `lgkm_pend`, b128 stagger. **Expected 293/321.**

### Step 4 — Extract VmemPipe
Move `valu_vmem_*_deadline`, `vmem_drain_deadline`, `vm_pend`, `_vmem_wr_bypass_active`. **Expected 293/321.**

### Step 5 — Extract TransPipe + SgprScoreboard + VAluPipe
Three independent commits, each a pure refactor. **Expected 293/321 after each.**

### Step 6 — Extract ScalarPipe
s_nop / s_cbranch / SALU logic all live behind `ScalarPipe.*`. Still running with CBRANCH_TIGHT=9. **Expected 293/321.**

### Step 7 — Apply HW-confirmed fixes from PROBE_FINDINGS
Now subsystems are isolated; land one at a time:

- **7a** `ScalarPipe.cbranch_cost` → 8 tight / 13 cold. Passes `probe_scalar_beat_p0..p3`. Regression: probe_branch_cost w0[7] now 8 (was 9, HW=8) — **gain 1**.
- **7b** `ScalarPipe.snop_cost` → post-VMEM-drain returns 20 per nop. Passes `probe_nop_chain_n1/n3/n5`. Regression: probe_cmp_chain w1[21] 22→18 (HW=18) — **gain 1**. probe_sgpr_cmps s_nop(15)[23]=20 (was 16, HW=20) — **gain 2–3**.
- **7c** `ScalarPipe.snop_cost` → last-nop-in-chain +4. Must NOT double-count with vmcnt path. Passes `probe_nop_chain_n5` last-token tail.

### Step 8 — VOPD dep-aware spacing (Pattern C)
Gated on new microbench `probe_vopd_dep_raw`. Once calibrated, `VAluPipe.vopd_ready` returns `last_vopd_issue + VOPD_INDEP_PIPE_CYCLES` (no dep) vs `+ VOPD_CHAIN_DEP_CYCLES` (dep). **Expected +4 to +8.**

### Step 9 — Pattern A (VMEM store 17/21) via scoreboard-warm marker
Requires new probe `probe_store_bypass_mix_load`. **Expected +3 to +4.**

### Step 10 — Pattern F wave-slip (if feasible)
Conjectural; depends on wave-1 probe captures.

Every step passes through CI:
```
DEV=AMD MOCKGPU=1 PYTHON_REMU=1 PROFILE=1 SQTT=1 \
  .venv/bin/python extra/sqtt/rigorous_hw_test.py --compare
```

---

## 6. Hypotheses to test via microbenchmarks

1. **H1 — VOPD dep by reg-set intersection.** (C, Batch A) Confirm via `probe_vopd_spacing(chain | no_dep | split_depctr)`.
2. **H2 — VGPR bank conflict +1 beyond reg-set overlap.** (C, Batch A) Probe: VOPD writing `v0/v1`, next VOPD reading `v4/v5` (same bank).
3. **H3 — s_nop(15) cost is 20 only post-s_waitcnt_vmcnt.** (B, HW CONFIRMED — PROBE_FINDINGS §B)
4. **H4 — cbranch_scc1 cost 8 tight vs 13 cold.** (E, HW CONFIRMED — PROBE_FINDINGS §E)
5. **H5 — v_cmp LIT completion buffer 2cy commit serialization.** (Already implemented)
6. **H6 — VMEM store "warm" bypass gated on prior global_load.** (A, Batch B, FALSIFIED for current kernel shape)
7. **H7 — cndmask SGPR-read first-use spike.** (C.C2, Batch A)
8. **H8 — last-nop-before-VALU IB resume +4.** (B.B1, Batch B)
9. **H9 — Wave-pair TRANS pipe serialization.** (F.F3, needs wave-1 capture)
10. **H10 — Cold-start VALU issue port sharing.** (D, Batch B)
11. **H11 — Post-trans scalar stall 2cy.** (Already partially modeled)

Each hypothesis maps to **one subsystem method** — enables isolated fitting.

---

## 7. Risk / open questions

1. **`VOPD_PIPE_CYCLES=4` is the only value keeping `exp_chain` at 112/112 ±2.** Refactor alone doesn't fix C; makes fix *tryable* without collateral damage.
2. **Wave-slip patterns (D, F) need wave-1 traced captures.** HW-capture work item, gates last ~8 mismatches.
3. **`probe_sgpr_cmps [23]=20 vs [21]=16` in wave 0** — last-nop-in-chain path must not double-count with post-vcmp SGPR drain path.
4. **`PeerSnapshot` copy cost** is O(n) per pick. For n≤4 this is irrelevant. Don't optimize speculatively.

---

## 8. Out of scope

- Waves-per-SIMD > 2 (no evidence in scoring corpus)
- SIMD-residency signal (falsified)
- CDNA/RDNA4 (subclass constants class when needed)

---

## 9. Deliverables of the rewrite

- `test/mockgpu/amd/emu.py`: `_simulate_sq_timing` shrinks ~655 LoC → ~120 LoC orchestration.
- `test/mockgpu/amd/sq_timing/` (new package): `valu.py`, `trans.py`, `scalar.py`, `sgpr.py`, `vmem.py`, `lds.py`, `ib_fetch.py`, `scheduler.py`, `constants.py`.
- `test/amd/test_emulator_timing.py`: new per-subsystem unit tests.
- `extra/sqtt/rigorous_hw_test.py`: unchanged contract.

---

## Critical Files for Implementation

- `/home/admin653792/code/tinygrad/test/mockgpu/amd/emu.py` — target of rewrite.
- `/home/admin653792/code/tinygrad/extra/sqtt/rigorous_hw_test.py` — `--compare` harness.
- `/home/admin653792/code/tinygrad/extra/sqtt/rgp/MISMATCH_ANALYSIS.md` — 28-token table, Patterns A–F.
- `/home/admin653792/code/tinygrad/extra/sqtt/rgp/PROBE_FINDINGS.md` — HW-confirmed constants.
- `/home/admin653792/code/tinygrad/test/amd/test_emulator_timing.py` — unit-test harness.
- `/home/admin653792/code/tinygrad/extra/sqtt/rgp/MICROBENCH_TAXONOMY.md` — 302 kernels driving step 7–10 calibration.
