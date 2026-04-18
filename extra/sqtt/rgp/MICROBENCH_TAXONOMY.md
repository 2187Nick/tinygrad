# Microbenchmark Taxonomy — RDNA3 Cycle-Accurate Emulator

## Purpose

Systematic Chips-and-Cheese / NaviSim / GPGPU-Sim-style probe library to drive
the tinygrad RDNA3 emulator from 293/321 (91.3%) to 321/321 exact match on the
AMD 7900 XTX reference captures in `extra/sqtt/captures/rigorous/*.pkl`.

Each kernel isolates **one** HW behavior with **one** parameter knob so that
when it fails to match the emulator, the diagnosis is unambiguous. Every
kernel is a standalone `custom_mb_*(A:UOp, arch:str)` function following the
pattern in `test/amd/test_custom_kernel.py` using `Kernel(arch); k.emit(...)`
and dispatching `Tensor.empty(1024)` so the traced CU is guaranteed to receive
wave-1 coverage (16 WGs × 2 waves = 32 waves).

## Mismatch coverage reference

Running numbering used in this document maps to the 28 mismatched tokens from
`extra/sqtt/rgp/MISMATCH_ANALYSIS.md`:

| Code | Kernel              | W | Idx | HW | EMU | Notes                              |
|------|---------------------|---|-----|----|-----|------------------------------------|
| A1   | data_deps           | 1 | 6   | 17 | 21  | global_store_b32 bypass           |
| A2   | probe_cmp_chain     | 0 | 23  | 17 | 21  | global_store_b32 bypass           |
| A3   | probe_branch_cost   | 0 | 14  | 17 | 21  | global_store_b32 bypass           |
| B1   | probe_cmp_chain     | 1 | 21  | 18 | 22  | s_nop(15) after vcmp chain        |
| B2   | probe_sgpr_cmps     | 0 | 23  | 20 | 16  | last nop in chain +4              |
| B3   | probe_sgpr_cmps     | 1 | 21  | 20 | 16  | first nop post-trans-slip         |
| B4   | probe_sgpr_cmps     | 1 | 23  | 20 | 16  | last nop in chain +4              |
| C1   | exp_chain           | 0 | 26  | 1  | 3   | VOPD post-depctr                  |
| C2   | exp_chain           | 0 | 31  | 3  | 4   | vcmp_e64 s[1]                     |
| C3   | exp_chain           | 0 | 34  | 4  | 1   | cndmask first-cold SGPR           |
| C4   | exp_chain           | 0 | 35  | 1  | 3   | cndmask[6] shuffle                |
| C5   | exp_chain           | 0 | 36  | 1  | 2   | cndmask[7] shuffle                |
| C6   | exp_chain           | 0 | 37  | 3  | 1   | VOPD back-to-back spacing         |
| C7   | exp_chain           | 0 | 38  | 2  | 4   | VOPD pair spacing                 |
| C8   | exp_chain           | 0 | 40  | 2  | 1   | trans cold-start                  |
| C9   | exp_chain           | 0 | 54  | 1  | 4   | vcmp_e64 s[1]                     |
| C10  | exp_chain           | 0 | 57  | 3  | 1   | cndmask first-cold SGPR           |
| C11  | exp_chain           | 0 | 58  | 1  | 3   | cndmask shuffle                   |
| C12  | exp_chain           | 0 | 61  | 3  | 1   | VOPD back-to-back                 |
| D1   | probe_vmem_chain    | 1 | 2   | 4  | 1   | wave-1 VALU issue slip            |
| E1   | probe_branch_cost   | 0 | 7   | 8  | 9   | cbranch tight fast-path           |
| E2   | probe_branch_cost   | 1 | 5   | 2  | 1   | s_mov post-vmem stagger           |
| E3   | probe_branch_cost   | 1 | 7   | 10 | 9   | cbranch post-vmem                 |
| F1   | probe_sgpr_cmps     | 0 | 16  | 2  | 1   | cndmask tail slip                 |
| F2   | probe_sgpr_cmps     | 1 | 8   | 2  | 1   | v_mov fresh VGPR                  |
| F3   | probe_sgpr_cmps     | 1 | 16  | 5  | 1   | cndmask tail slip (cumulative)    |
| F4   | probe_sgpr_cmps     | 1 | 18  | 10 | 4   | v_log wave-pair TRANS             |
| F5   | probe_cmp_chain     | 1 | 6   | 2  | 1   | v_mov fresh VGPR                  |

---

## Summary

| Batch | Focus                            | Kernels | HW capture hours |
|-------|----------------------------------|---------|------------------|
| A     | Single-instruction microbenches  | 76      | ~4.5             |
| B     | Resource-conflict kernels        | 142     | ~8.5             |
| C     | Scheduler / occupancy            | 54      | ~3.2             |
| D     | Stress kernels (Seb-V style)     | 30      | ~1.8             |
| **Total** |                              | **302** | **~18 h**        |

Each capture is ~17 kernels per pkl batch at ~20 min/batch.
302 kernels ÷ 17 ≈ 18 batches ≈ ~6 overnight runs of ~3 batches each, or
one ~18-hour burn-in. Batch A should go first (needed for constant
calibration), then D (co-residence), then B (conflicts), then C (occupancy).

---

# Batch A — Single-instruction microbenchmarks (76 kernels, ~4.5 h)

Goal: establish the **baseline 1-cycle issue cost** and **drain-recovery cost**
for every ISA opcode we already generate. Every Batch-A kernel has the
canonical prologue:
```
s_load_b64 → s_waitcnt_lgkmcnt → v_lshlrev_b32_e32(v[0], 2) → global_load_b32
 → s_waitcnt_vmcnt → <probe body> → v_add_f32(v[1],1,v[1]) → global_store_b32
 → s_endpgm
```
The probe body holds the opcode(s) under test. dt tables target the `<probe body>`
tokens only.

## A.1 — v_add_f32 family (5 kernels)

| name                             | params                                    | HW expect          | diagnoses |
|----------------------------------|-------------------------------------------|--------------------|-----------|
| mb_valu_add_n1                   | 1 isolated `v_add_f32 v[1],1.0,v[1]`      | dt=1               | sanity    |
| mb_valu_add_n2                   | 2 back-to-back adds, RAW on v[1]          | dt=1,1             | sanity    |
| mb_valu_add_n4                   | 4 back-to-back adds, RAW on v[1]          | dt=1×4             | VALU burst issue |
| mb_valu_add_n8                   | 8 back-to-back adds, RAW on v[1]          | dt=1×8             | VALU burst, D1 |
| mb_valu_add_n16                  | 16 back-to-back adds, RAW on v[1]         | dt=1×16            | burst saturation |

## A.2 — v_mul / v_fmac / v_dual family (9 kernels)

| name                             | params                                    | HW expect          | diagnoses |
|----------------------------------|-------------------------------------------|--------------------|-----------|
| mb_valu_mul_n1                   | 1 `v_mul_f32 v[1],v[1],v[2]`              | dt=1               | sanity    |
| mb_valu_mul_n4                   | 4 back-to-back RAW v_mul                  | dt=1×4             | baseline  |
| mb_valu_fmac_n1                  | 1 `v_fmac_f32 v[1],v[2],v[3]`             | dt=1               | sanity    |
| mb_valu_fmac_n4                  | 4 back-to-back RAW v_fmac                 | dt=1×4             | baseline  |
| mb_valu_fmac_n8                  | 8 back-to-back RAW v_fmac                 | dt=1×8             | baseline  |
| mb_vopd_fmac_mul_n2              | 2 VOPD v_dual_fmac_f32 + v_dual_mul_f32   | dt=1,1             | C6/C12    |
| mb_vopd_fmac_mul_n4              | 4 VOPDs same pair, no RAW                 | dt=1,1,1,1         | C6/C12    |
| mb_vopd_cndmask_n2               | 2 VOPD v_dual_cndmask_b32                 | dt=1,1             | C-family  |
| mb_vopd_mixed_n4                 | 4 VOPDs alternating add_mul/mul_sub       | dt=1×4 or 1,3,1,3  | C6/C12    |

## A.3 — Transcendental (v_exp / v_log / v_sqrt / v_rcp / v_rsq) (10 kernels)

| name                             | params                                    | HW expect          | diagnoses |
|----------------------------------|-------------------------------------------|--------------------|-----------|
| mb_trans_exp_n1                  | 1 `v_exp_f32 v[1],v[1]`                   | dt=1               | sanity    |
| mb_trans_log_n1                  | 1 `v_log_f32 v[1],v[1]`                   | dt=1               | C8        |
| mb_trans_sqrt_n1                 | 1 `v_sqrt_f32 v[1],v[1]`                  | dt=1               | sanity    |
| mb_trans_rcp_n1                  | 1 `v_rcp_f32 v[1],v[1]`                   | dt=1               | sanity    |
| mb_trans_rsq_n1                  | 1 `v_rsq_f32 v[1],v[1]`                   | dt=1               | sanity    |
| mb_trans_exp_n4                  | 4 back-to-back v_exp, RAW                 | dt=1,4,4,4         | trans pipe occupancy |
| mb_trans_log_n4                  | 4 back-to-back v_log, RAW                 | dt=1,4,4,4         | trans pipe occupancy |
| mb_trans_mixed_exp_log           | `v_exp → v_log` RAW on v[1]               | dt=1,4             | trans pipeline |
| mb_trans_exp_valu_exp            | `v_exp v[1] → v_add v[2] → v_exp v[1]`    | dt=1,1,4           | trans+VALU parallel |
| mb_trans_cold_vs_warm            | `v_add; v_exp; v_add; v_add; v_exp`       | dt=1,2,1,1,1       | C8 (cold-warm gap) |

## A.4 — v_cmp family (8 kernels)

| name                             | params                                    | HW expect          | diagnoses |
|----------------------------------|-------------------------------------------|--------------------|-----------|
| mb_vcmp_vcc_n1                   | 1 `v_cmp_gt_f32_e32` → VCC                | dt=1               | sanity    |
| mb_vcmp_vcc_n4                   | 4 `v_cmp_gt_f32_e32` chain                | dt=1×4             | VCC write burst |
| mb_vcmp_sgpr_n1                  | 1 `v_cmp_gt_f32_e64 s[0]`                 | dt=1               | sanity    |
| mb_vcmp_sgpr_n4                  | 4 `v_cmp_gt_f32_e64 s[0..3]`              | dt=1×4             | C2/C9     |
| mb_vcmp_sgpr_n8                  | 8 `v_cmp_gt_f32_e64 s[0..7]`              | dt=1×4,?           | SGPR queue depth |
| mb_vcmp_vcc_then_sgpr            | vcc vcmp then e64 vcmp to s[0]            | dt=1,1             | mixed SGPR/VCC |
| mb_vcmp_literal                  | `v_cmp_gt_f32_e32 v[0], 1.0`              | dt=1               | LIT encoding |
| mb_vcmp_chain_different_regs     | 4 vcmps reading v[0..3]                   | dt=1×4             | port distinction |

## A.5 — v_cndmask family (8 kernels)

| name                             | params                                    | HW expect          | diagnoses |
|----------------------------------|-------------------------------------------|--------------------|-----------|
| mb_cndmask_vcc_n1                | `v_cndmask_b32 v[4]`, VCC                 | dt=1               | sanity    |
| mb_cndmask_vcc_n4                | 4 vcc cndmasks, distinct dst              | dt=1×4             | C/F       |
| mb_cndmask_sgpr_n1               | `v_cndmask_b32_e64 v[4], s[0]`            | dt=1 or 2          | C3/C10    |
| mb_cndmask_sgpr_fresh_n4         | 4 cndmasks reading s[0..3] fresh          | dt=2,1,1,1         | C3/C10    |
| mb_cndmask_sgpr_stale_n4         | vcmp→nop(8)→cndmasks reading same s[0..3] | dt=1,1,1,1         | stale bypass |
| mb_cndmask_sgpr_followed_by_vcmp | 4 cndmasks then vcmp                      | ; after: +1 or +2  | F2 tail-slip |
| mb_cndmask_tail_gt4              | 5 cndmasks on s[0..4]                     | dt=2,1,1,1,2       | F1/F3 tail slip |
| mb_cndmask_dst_fresh_vgpr_high   | cndmasks to v[10..15] never touched       | dt=2,1,1,1,1,1     | F2 fresh-VGPR hypothesis |

## A.6 — SALU opcodes (10 kernels)

| name                             | params                                    | HW expect          | diagnoses |
|----------------------------------|-------------------------------------------|--------------------|-----------|
| mb_salu_smov_n1                  | `s_mov_b32 s[4],0`                        | dt=1               | sanity    |
| mb_salu_smov_n4                  | 4 s_mov distinct sgprs                    | dt=1×4             | baseline  |
| mb_salu_smov_followed_by_nop0    | `s_mov; s_nop(0)`                         | s_nop dt=3         | E-family scalar warmup |
| mb_salu_scmp_tight               | `s_mov; s_cmp_eq_i32; s_cbranch_scc1`     | dts=1,1,8          | E1        |
| mb_salu_scmp_spaced_nop0         | `s_mov; s_nop(0); s_cmp; s_cbranch`       | dts=1,3,1,13       | E1 (slow path) |
| mb_salu_scmp_spaced_nop0x2       | `s_mov; s_nop(0)×2; s_cmp; s_cbranch`     | 1,3,1,1,13         | E1        |
| mb_salu_scmp_spaced_nop0x3       | `s_mov; s_nop(0)×3; s_cmp; s_cbranch`     | 1,3,1,1,1,13       | E1        |
| mb_salu_scbranch_taken           | forced-taken s_cbranch                    | dt=13-16           | branch taken path |
| mb_salu_sbitcmp_branch           | `s_bitcmp1_b32; s_cbranch_scc1`           | dt=?,?             | alt SCC writer |
| mb_salu_sand_scmp_branch         | `s_and_b32 (sets SCC); s_cbranch_scc1`    | dt=?,?             | SCC path from s_and |

## A.7 — s_nop cost by N and by predecessor (16 kernels)

| name                             | params                                    | HW expect          | diagnoses |
|----------------------------------|-------------------------------------------|--------------------|-----------|
| mb_snop_0_after_valu             | `v_add; s_nop(0)`                         | dt=1               | baseline  |
| mb_snop_5_after_valu             | `v_add; s_nop(5)`                         | dt=6 or 7          | formula check |
| mb_snop_10_after_valu            | `v_add; s_nop(10)`                        | dt=11 or 13        | per SESSION: 13 |
| mb_snop_15_after_valu            | `v_add; s_nop(15)`                        | dt=16              | confirm middle |
| mb_snop_0_after_scalar           | `s_mov; s_nop(0)`                         | dt=3               | E-family scalar warmup |
| mb_snop_15_after_scalar          | `s_mov; s_nop(15)`                        | dt=16              | scalar predecessor |
| mb_snop_15_after_waitcnt_vmcnt   | `s_waitcnt_vmcnt; s_nop(15); v_add`       | nop_dt=20          | PROBE_FINDINGS B |
| mb_snop_15_after_waitcnt_lgkmcnt | `s_waitcnt_lgkmcnt; s_nop(15); v_add`     | nop_dt=?           | unknown   |
| mb_snop_15_after_waitcnt_empty   | `s_waitcnt(0); s_nop(15); v_add`          | dt=16 or 20        | B2/B3     |
| mb_snop_15_after_depctr          | `s_waitcnt_depctr(4095); s_nop(15)`       | dt=?               | depctr reset |
| mb_snop_15_chain_n2              | 2 nops then v_add                         | dts=16,20          | B2/B4 (last-in-chain) |
| mb_snop_15_chain_n3              | 3 nops then v_add                         | dts=16,16,20       | B2/B4     |
| mb_snop_15_chain_n5              | 5 nops then v_add                         | dts=16,16,16,16,20 | B2/B4     |
| mb_snop_15_chain_n8              | 8 nops then v_add                         | 16×7, 20           | B2/B4     |
| mb_snop_15_chain_after_vmcnt_n3  | vmcnt → 3 nops → v_add                    | 20,16,20           | B2/B3/B4  |
| mb_snop_mixed_values             | `s_nop(0); s_nop(15); s_nop(5); v_add`    | 1,16,6,1           | predecessor dependence |

## A.8 — s_waitcnt variants (6 kernels)

| name                             | params                                    | HW expect          | diagnoses |
|----------------------------------|-------------------------------------------|--------------------|-----------|
| mb_waitcnt_vmcnt_null            | after DRAM load                           | dt=~600            | sanity    |
| mb_waitcnt_vmcnt_nonzero         | `s_waitcnt_vmcnt(7)`                      | dt=small           | partial drain |
| mb_waitcnt_lgkmcnt_null_smem     | after `s_load_b64`                        | dt=~200            | sanity    |
| mb_waitcnt_lgkmcnt_null_lds      | after `ds_load_b32`                       | dt=~31             | LDS       |
| mb_waitcnt_depctr_4095           | `v_add; s_waitcnt_depctr(4095); v_add`    | dt=? (~25)         | C1 (depctr reset) |
| mb_waitcnt_empty_barrier         | `s_waitcnt(0)` no pending                 | dt=1               | baseline  |

## A.9 — LDS opcodes (ds_load/store) (8 kernels)

| name                             | params                                    | HW expect          | diagnoses |
|----------------------------------|-------------------------------------------|--------------------|-----------|
| mb_lds_store_b32_n1              | 1 `ds_store_b32`                          | dt=1               | sanity    |
| mb_lds_store_b64_n1              | 1 `ds_store_b64`                          | dt=1               | sanity    |
| mb_lds_store_b128_n1             | 1 `ds_store_b128`                         | dt=1               | sanity    |
| mb_lds_load_b32_n1               | 1 `ds_load_b32 → s_waitcnt → v_add`       | waitcnt=~31        | sanity    |
| mb_lds_load_b64_n1               | 1 `ds_load_b64 → s_waitcnt → v_add`       | waitcnt=~31        | LDS_b64   |
| mb_lds_load_b128_n1              | 1 `ds_load_b128 → s_waitcnt → v_add`      | waitcnt=~36        | LDS_b128 extra |
| mb_lds_store_then_valu_forward   | `ds_store_b32; v_add(stored reg)`         | dt_add=26          | LDS_WR_FORWARD |
| mb_lds_load_then_valu_forward    | `ds_load_b32; wait; v_add(loaded reg)`    | dt_add=22          | LDS_RD_FORWARD |

## A.10 — global_load / global_store (6 kernels)

| name                             | params                                    | HW expect          | diagnoses |
|----------------------------------|-------------------------------------------|--------------------|-----------|
| mb_vmem_store_b32_isolated       | `v_add(v[1]); global_store_b32 v[1]`      | dt=21              | A-family (no warm) |
| mb_vmem_store_b32_after_load     | `global_load; wait; v_add; store`         | dt=17 or 21        | A1/A2/A3  |
| mb_vmem_store_b32_chain_n4       | 4 `global_store` to v[1..4], paired VALU  | dts=21,1,1,1 ?     | A chain hypothesis |
| mb_vmem_store_b32_spaced_nopK    | K nops between producer VALU and store    | dt=21-K (clamp 4)  | A spacing curve |
| mb_vmem_store_b64_isolated       | `global_store_b64`                        | dt=?               | b64 baseline |
| mb_vmem_store_b128_isolated      | `global_store_b128 (v[1:4])`              | dt=20 or 24        | b128 extra |

---

# Batch B — Resource-conflict kernels (142 kernels, ~8.5 h)

## B.1 — LDS bank-conflict sweep (48 kernels)

16 LDS banks on RDNA3 (32 at chip level, but 16 for compute-shader group
allocation). Each kernel dispatches 64-thread WGs where each thread reads/writes
a pattern that hits K distinct banks in parallel.

For each of `{b32, b64, b128}`:

| name                             | params                                                | HW expect    | diagnoses |
|----------------------------------|-------------------------------------------------------|--------------|-----------|
| mb_lds_rd_b32_bank1              | thread i reads lds[0]        — full broadcast         | dt=~31       | baseline  |
| mb_lds_rd_b32_bank2              | thread i reads lds[(i%2)*4]  — 2 banks                | dt=~32       | 2-way conflict |
| mb_lds_rd_b32_bank4              | thread i reads lds[(i%4)*4]  — 4 banks                | dt=~33       | 4-way     |
| mb_lds_rd_b32_bank8              | thread i reads lds[(i%8)*4]  — 8 banks                | dt=~34       | 8-way     |
| mb_lds_rd_b32_bank16             | thread i reads lds[(i%16)*4] — no conflict            | dt=~31       | no conflict baseline |
| mb_lds_rd_b32_stride4            | thread i reads lds[i*16]     — max conflict           | dt=high      | stride-4 pathological |
| mb_lds_rd_b32_stride8            | thread i reads lds[i*32]                              | dt=high      | stride-8 |
| mb_lds_rd_b32_random             | shuffled addresses                                    | dt=~31       | random distribution |
| …and 8 identical for `ds_store_b32` (write) variants                              |              |          |
| …and 16 for b64, 16 for b128 (same pattern)                                       |              |          |

Total: 8 rd + 8 wr each × {b32, b64, b128} = **48 kernels.**

All diagnose future LDS bank-conflict model. Not required to reach 100% on
current 28 mismatches but calibrates LDS_b128 stagger constants.

## B.2 — VGPR bank-port conflict probe (24 kernels)

RDNA3 has 4 VGPR banks (v[n] %4). Two VALUs reading the same bank in adjacent
cycles conflict. Tests walk every combination of {src_bank, dst_bank}.

| name                             | params                                    | HW expect          | diagnoses |
|----------------------------------|-------------------------------------------|--------------------|-----------|
| mb_vgpr_bank_same_src_n4         | `v_add v[2],v[0],v[0]` ×4 (bank 0 only)   | dt=1,2,1,2 ?       | bank0 conflict |
| mb_vgpr_bank_diff_src_n4         | `v_add v[2],v[1],v[5]`×4 (b1/b1→b2, b3/b3→b0, …) | dt=1×4      | no conflict baseline |
| mb_vgpr_bank_src2_same_n4        | `v_fmac v[4],v[0],v[0]` (src0=src1 b0)    | dt=?               | self-bank read |
| mb_vgpr_bank_3src_b0_b0_b0       | `v_fma v[4],v[0],v[0],v[0]`               | dt=?               | 3 reads, same bank |
| mb_vgpr_bank_3src_b0_b1_b2       | `v_fma v[4],v[0],v[1],v[2]`               | dt=1               | 3 distinct banks baseline |
| mb_vgpr_bank_3src_b0_b0_b1       | `v_fma v[4],v[0],v[0],v[1]`               | dt=?               | 2-in-bank0      |
| mb_vgpr_bank_pair_b0_b0          | 2 VALU: `v_add v[2],v[0]; v_add v[6],v[0]`| dt=1,? (bank-cross)| adjacent-cycle conflict |
| mb_vgpr_bank_pair_b0_b1          | `v_add v[2],v[0]; v_add v[6],v[1]`        | dt=1,1             | no conflict |
| mb_vgpr_bank_pair_b0_b2          | `v_add v[2],v[0]; v_add v[6],v[2]`        | dt=1,1             | no conflict |
| mb_vgpr_bank_pair_b0_b3          | `v_add v[2],v[0]; v_add v[6],v[3]`        | dt=1,1             | no conflict |
| mb_vgpr_bank_chain_b0_aaaa       | 8 VALUs all reading v[0]                  | dts=1,?,?,?…       | bank-0 saturation |
| …16 total pattern variants covering dst-bank × src-bank × 3-src ops               |          |          |

## B.3 — VOPD eligibility matrix (40 kernels)

VOPD requires: opx ∈ {add,mul,fmac,cndmask,mov,max,min,sub}, opy ∈ same set,
non-overlapping dst parity (v[even]/v[odd]), disjoint src banks.

Each cell tests **one pairing** for dt=1 (eligible/fused) vs dt≥1 (split):

### B.3.a — Opcode pair grid (20 kernels)

| name                             | opx / opy                               | HW expect (dt)        | diagnoses |
|----------------------------------|-----------------------------------------|-----------------------|-----------|
| mb_vopd_add_add                  | v_dual_add_f32 + v_dual_add_f32         | fused, dt=1           | sanity     |
| mb_vopd_mul_mul                  | v_dual_mul_f32 + v_dual_mul_f32         | fused, dt=1           | C6/C12     |
| mb_vopd_fmac_fmac                | v_dual_fmac_f32 + v_dual_fmac_f32       | fused, dt=1           | matmul core |
| mb_vopd_cndmask_cndmask          | v_dual_cndmask + v_dual_cndmask         | fused, dt=1           | cndmask pair |
| mb_vopd_add_mul                  | add + mul                               | fused, dt=1           | baseline   |
| mb_vopd_mul_add                  | mul + add                               | fused, dt=1           | baseline   |
| mb_vopd_fmac_add                 | fmac + add                              | fused, dt=1           | baseline   |
| mb_vopd_fmac_mul                 | fmac + mul                              | fused, dt=1           | baseline   |
| mb_vopd_cndmask_add              | cndmask + add                           | fused, dt=1           | C-family   |
| mb_vopd_cndmask_mul              | cndmask + mul                           | fused, dt=1           | C-family   |
| mb_vopd_sub_sub                  | sub + sub                               | fused, dt=1           | sanity     |
| mb_vopd_min_max                  | v_dual_min_f32 + v_dual_max_f32         | fused, dt=1           | sanity     |
| mb_vopd_mov_add                  | v_dual_mov_b32 + v_dual_add_f32         | fused, dt=1           | mov fuse   |
| mb_vopd_mov_mov                  | v_dual_mov + v_dual_mov                 | fused, dt=1           | mov-mov    |
| mb_vopd_dot2_add                 | v_dual_dot2_f16 + v_dual_add            | fused, dt=1 (if exists)| dot2 eligibility |
| mb_vopd_add_int                  | v_dual_add_u32 + v_dual_add_u32         | fused, dt=1           | int VOPD   |
| mb_vopd_mixed_f32_f16            | f32 op + f16 op                         | not fused, dt≥1        | incompat   |
| mb_vopd_mul_fmac_bank_same       | bank collision between mul and fmac     | not fused, split       | bank split |
| mb_vopd_lit_add                  | v_dual_add(lit) + v_dual_mul            | fused, dt=1 (VOPD_LIT) | LIT path   |
| mb_vopd_lit_lit                  | both operands with literal              | not fused              | double-LIT reject |

### B.3.b — Dependency / bank / VCC matrix (20 kernels)

| name                                | params                                            | HW expect (dt sequence) | diagnoses |
|-------------------------------------|---------------------------------------------------|-------------------------|-----------|
| mb_vopd_pair_no_raw                 | 2 VOPDs, disjoint regs                            | 1,1                     | C6 (baseline) |
| mb_vopd_pair_raw_x                  | 2 VOPDs, VOPD2.srcx = VOPD1.dstx                  | 1,3                     | C6/C12     |
| mb_vopd_pair_raw_y                  | 2 VOPDs, VOPD2.srcy = VOPD1.dsty                  | 1,3                     | C6/C12     |
| mb_vopd_pair_raw_xy                 | Both src lanes RAW                                | 1,3                     | C6/C12     |
| mb_vopd_pair_war                    | Write-after-read                                  | 1,1                     | WAR ok     |
| mb_vopd_chain_n4_raw                | 4 VOPDs all RAW                                   | 1,3,3,3                 | C6/C12     |
| mb_vopd_chain_n4_no_raw             | 4 VOPDs no dep                                    | 1,1,1,1                 | C1 baseline |
| mb_vopd_vcc_producer_then_cndmask   | `v_cmp→VOPD(cndmask_x,cndmask_y)`                 | 1,1                     | VCC propagation |
| mb_vopd_vcc_within_pair             | VOPD writes VCC + next VOPD reads                 | 1,3                     | VCC raw cross-VOPD |
| mb_vopd_vcc_war                     | cndmask reads VCC, later VOPD writes VCC          | 1,1                     | VCC WAR |
| mb_vopd_bank_conflict_src           | both opx and opy read v[0] (same bank)            | not fused or 1,3        | bank conflict |
| mb_vopd_bank_conflict_dst           | dst parity violation (v[0]/v[2])                  | not fused (rejected)    | compiler usually prevents |
| mb_vopd_post_depctr                 | VOPD, s_waitcnt_depctr(4095), VOPD                | 1,25,1                  | C1        |
| mb_vopd_post_waitcnt_vmcnt          | VMEM return → VOPD                                | wait,1                  | sanity    |
| mb_vopd_post_trans                  | v_exp → VOPD                                      | 1,? (2?)                | trans→VOPD |
| mb_vopd_pre_trans                   | VOPD → v_exp on VOPD-dst                          | 1,4                     | trans raw |
| mb_vopd_sandwich_trans              | VOPD → v_exp → VOPD                               | 1,1,?                   | trans mid-pair |
| mb_vopd_lit_chain_n4                | 4 VOPD_LIT back-to-back                           | 1,1,1,1                 | LIT fast path |
| mb_vopd_lit_then_nonlit             | VOPD_LIT → VOPD                                   | 1,1                     | mixed |
| mb_vopd_nonlit_then_lit             | VOPD → VOPD_LIT                                   | 1,1 or 1,3              | which direction triggers? |

## B.4 — v_cmp / v_cndmask / SGPR forwarding (15 kernels)

| name                                | params                                             | HW expect       | diagnoses |
|-------------------------------------|----------------------------------------------------|-----------------|-----------|
| mb_vcmp_cndmask_k1                  | 1 vcmp → 1 cndmask                                 | 1,1 or 1,2      | C3/C10 cold SGPR |
| mb_vcmp_cndmask_k2                  | 2 vcmps → 2 cndmasks                               | 1,1,2,1         | C3/C10    |
| mb_vcmp_cndmask_k4                  | 4 vcmps → 4 cndmasks                               | 1×4,2,1,1,1     | C-family  |
| mb_vcmp_cndmask_k8                  | 8 vcmps → 8 cndmasks                               | 1×8,2,1×7       | SGPR queue |
| mb_vcmp_spaced_cndmask              | vcmp → nop(4) → cndmask                            | 1,5,1           | warm/cold boundary |
| mb_vcmp_spaced_cndmask_nop8         | vcmp → nop(8) → cndmask                            | 1,9,1           | warm boundary |
| mb_vcmp_spaced_cndmask_nop12        | vcmp → nop(12) → cndmask                           | 1,13,1          | full drain |
| mb_cndmask_tail_retire              | 4 cndmasks then extra cndmask same sgpr            | 1×4,2 or 1      | F1/F3 tail-retire |
| mb_cndmask_tail_new_sgpr            | 4 cndmasks then cndmask reads NEW s[4]             | 1×4,2           | F1/F3     |
| mb_cndmask_new_vgpr_tail            | last cndmask writes fresh v[20]                    | ,1 or ,2        | F2 fresh-VGPR |
| mb_vcmp_after_cndmask_chain         | 4 cndmasks then vcmp                               | 1×4,1 or 2      | F closing stall |
| mb_vcmp_interleave_cndmask          | vcmp,cndmask,vcmp,cndmask,…                        | 1×8             | interleaved |
| mb_vcmp_e32_vs_e64                  | e32 writes VCC, e64 writes s[k]                    | baseline        | encoding path |
| mb_cndmask_read_vcc_then_sgpr       | cndmask(VCC), cndmask(s[0])                        | 1,2 or 1,1      | src route |
| mb_cndmask_sgpr_k_sweep             | 1..8 vcmps then 1 cndmask reading last             | ,k,1            | SGPR drain curve |

## B.5 — Trans pipe interactions (15 kernels)

| name                                | params                                          | HW expect         | diagnoses |
|-------------------------------------|--------------------------------------------------|-------------------|-----------|
| mb_trans_after_trans_0              | v_exp; v_log                                     | 1,4               | F4 wave-pair? (single wave baseline) |
| mb_trans_after_trans_1              | v_exp; v_add; v_log                              | 1,1,4             | partial overlap |
| mb_trans_after_trans_4              | v_exp; v_add×4; v_log                            | 1,1,1,1,1,1       | trans-idle |
| mb_trans_after_trans_8              | v_exp; v_add×8; v_log                            | 1×9,1             | no conflict |
| mb_trans_raw_exp_log                | v_exp v[1]; v_log v[1] RAW                       | 1,31 (latency)    | raw stall |
| mb_trans_raw_valu                   | v_exp v[1]; v_add v[2],v[1]                      | 1,27              | trans→VALU raw |
| mb_trans_raw_with_depctr            | v_exp; s_waitcnt_depctr(4095); v_add v[1]        | 1,?,1             | depctr trans drain |
| mb_trans_then_salu                  | v_sqrt; s_waitcnt(); s_mov                       | 1,3,1             | C8/PROBE B |
| mb_trans_then_snop                  | v_sqrt; s_waitcnt; s_nop(15); v_add              | 1,3,16 or 20,1    | B3 (first nop post-trans) |
| mb_trans_chain4_then_snop           | 4 trans chain, waitcnt, s_nop(15)                | …,20 (first-nop)  | B3        |
| mb_trans_pair_same_op               | v_exp; v_exp                                     | 1,4               | trans occupancy |
| mb_trans_pair_diff_op               | v_exp; v_log (no RAW)                            | 1,4               | diff trans type |
| mb_trans_nested_valu                | v_exp; v_fmac; v_log; v_fmac                     | 1,1,4,1           | interleaved trans |
| mb_trans_cold_warm_alternating      | v_exp; v_exp×4; v_add; v_exp; v_add; v_exp       | dts vary          | C8 warm detection |
| mb_trans_vopd_cross                 | v_exp; VOPD(mul,mul); v_log                      | 1,1,4             | VOPD mid-trans |

## B.6 — Wave co-residence forcing (WG-size × reg-pressure) (15 kernels)

Objective: defeat the "waves on different SIMDs" bias found via RGP. Combine
large WGs (128/256 thread) with register-pressure tricks to push 2+ waves
onto the same SIMD.

| name                                | WG size | VGPRs alloc  | HW expect                         | diagnoses |
|-------------------------------------|---------|--------------|------------------------------------|-----------|
| mb_cores_wg64_lowreg                | 64      | 8            | 1 wave, different SIMD             | D1 baseline |
| mb_cores_wg128_lowreg               | 128     | 8            | 2 waves same SIMD?                 | D1 contention |
| mb_cores_wg128_highreg              | 128     | 64           | 2 waves, reg pressure              | occupancy |
| mb_cores_wg256_lowreg               | 256     | 8            | 4 waves                            | occupancy |
| mb_cores_wg256_highreg              | 256     | 128          | spill or low occupancy             | occupancy |
| mb_cores_wg128_valu_burst           | 128     | 8, 16 v_adds | wave slip visible?                 | D1        |
| mb_cores_wg128_trans_burst          | 128     | 8, v_exp×8   | F4 wave-pair TRANS                 | F4        |
| mb_cores_wg128_store_burst          | 128     | 8, 4 stores  | A wave bimodality                  | A         |
| mb_cores_wg128_cndmask_burst        | 128     | 8, cndmask×8| F1/F3 tail slip                    | F1/F3     |
| mb_cores_wg128_snop15_chain         | 128     | 8, nop(15)×3| B wave slip on nop                 | B-family  |
| mb_cores_wg64_dual_issue            | 64      | 8, VOPD×8   | VOPD across waves                  | C-family  |
| mb_cores_wg64_scalar_intense        | 64      | 8, 10 s_mov | E-family wave behavior             | E         |
| mb_cores_wg128_lds_rd_burst         | 128     | 8, ds_load×4| LDS arbiter contention             | LDS       |
| mb_cores_wg128_mixed_vmem_lds       | 128     | 8, mix      | multi-queue                        | mixed     |
| mb_cores_wg128_pure_salu            | 128     | 4, s_mov×8  | scalar contention across waves     | scalar    |

---

# Batch C — Scheduler / occupancy (54 kernels, ~3.2 h)

## C.1 — Wave-launch rate by concurrency (10 kernels)

Sweep WG × register-pressure to control the number of concurrent waves on
the traced CU. Measure wavestart spacing and first-VALU gap.

| name                                | waves target | mechanism                     | HW expect            | diagnoses |
|-------------------------------------|--------------|-------------------------------|----------------------|-----------|
| mb_occ_waves_1                      | 1            | WG=32, reg=8                  | wavestart_gap = N/A  | D1 cold   |
| mb_occ_waves_2                      | 2            | WG=64, reg=8                  | wavestart ~10 cy     | D1        |
| mb_occ_waves_4                      | 4            | WG=128, reg=8                 | wavestart ~5-10 cy   | occupancy |
| mb_occ_waves_6                      | 6            | WG=192, reg=32                | wavestart spread     | occupancy |
| mb_occ_waves_8                      | 8            | WG=256, reg=64                | wavestart tight      | occupancy |
| mb_occ_waves_12                     | 12           | WG=384, reg=48                | wavestart tight      | occupancy |
| mb_occ_waves_16                     | 16           | WG=512, reg=24                | saturated launch     | occupancy |
| mb_occ_waves_highreg_2              | 2 (starved)  | WG=64, reg=192                | 1 wave/SIMD          | reg spill |
| mb_occ_waves_highreg_1              | 1            | WG=64, reg=252                | min occupancy        | reg spill |
| mb_occ_waves_lds_limited            | 4            | WG=128 + 48 KB LDS alloc      | LDS-limited occupancy| LDS occupancy |

## C.2 — Barrier cost at various wave counts (8 kernels)

Measure s_barrier completion time as function of wavefront count in WG.

| name                                | WG size | barriers    | HW expect               | diagnoses |
|-------------------------------------|---------|-------------|--------------------------|-----------|
| mb_barrier_wg64_b1                  | 64      | 1 barrier   | ~6 cy (_BARRIER_FROM_LAST)| baseline |
| mb_barrier_wg128_b1                 | 128     | 1 barrier   | ~8 cy                    | barrier scaling |
| mb_barrier_wg256_b1                 | 256     | 1 barrier   | ~10 cy                   | barrier scaling |
| mb_barrier_wg64_b4                  | 64      | 4 barriers  | 4× base                   | barrier burst |
| mb_barrier_wg128_b4                 | 128     | 4 barriers  | 4× scaled                 | barrier burst |
| mb_barrier_wg64_b8_interleave_valu  | 64      | 8 w/ VALU   | valu interleave           | overlap |
| mb_barrier_wg64_b1_after_lds        | 64      | after ds_store | waits for dscnt        | LDS-barrier |
| mb_barrier_wg64_b1_after_vmem       | 64      | after global_store | big stall           | VMEM-barrier |

## C.3 — s_sendmsg / s_sleep / s_barrier variants (8 kernels)

| name                                | params                           | HW expect         | diagnoses |
|-------------------------------------|----------------------------------|-------------------|-----------|
| mb_sendmsg_halt                     | s_sendmsghalt                    | dt=0 (skipped)    | skip sanity |
| mb_sendmsg_debug                    | s_sendmsg(MSG_DEBUG)             | dt=0              | skip sanity |
| mb_sleep_0                          | s_sleep(0)                       | dt=1-ish          | baseline  |
| mb_sleep_4                          | s_sleep(4)                       | dt=4*64 cy        | sleep unit |
| mb_sleep_8                          | s_sleep(8)                       | dt=8*64 cy        | sleep unit |
| mb_barrier_tight_chain              | barrier,nop,barrier              | 2×base            | barrier spacing |
| mb_barrier_after_ssleep             | s_sleep(4); s_barrier            | sleep done        | sleep+barrier |
| mb_sbarrier_rdna4_signal_wait       | s_barrier_signal; s_barrier_wait | RDNA4 only        | cross-arch |

## C.4 — Wave-pair slip measurement (10 kernels)

Designed to measure how waves interact on the same WG, across shared
resources. Each kernel pairs two waves via a 64-thread WG and tests a
specific slip trigger (closest in spirit to the D1, F-family effects).

| name                                | trigger                                | HW expect           | diagnoses |
|-------------------------------------|----------------------------------------|---------------------|-----------|
| mb_wp_cold_start_n1                 | no vmem, 1 v_mov                       | w1_dt[2]=1          | D1 baseline |
| mb_wp_cold_start_n2                 | no vmem, 2 v_mov                       | w1_dt[2]=2          | D1        |
| mb_wp_cold_start_n4                 | no vmem, 4 v_mov                       | w1_dt[2]=4          | D1 (matches HW) |
| mb_wp_cold_start_n8                 | no vmem, 8 v_mov                       | w1_dt[2]=? (≤8?)    | D1 saturation |
| mb_wp_warm_start_n4                 | vmem-delayed, 4 v_mov                  | w1_dt[2]=1          | D1 no-slip control |
| mb_wp_cold_start_staggered          | 4 v_mov on w0 + stagger nop on w1      | w1_dt[2]=1          | D1 stagger fix |
| mb_wp_trans_both                    | 4 v_exp on both waves                  | w1_v_log slips      | F4        |
| mb_wp_trans_one_only                | v_exp on w0 only                       | w1 baseline         | F4 control |
| mb_wp_store_both                    | store on both waves                    | A bimodal           | A         |
| mb_wp_cndmask_both                  | 8 cndmasks on both                     | F tail slips        | F1/F3/F5  |

## C.5 — Scheduler ordering and SIMD placement (10 kernels)

Probes that dump wavestart packet order to reveal SQ scheduler decisions.
Parse result via RGP `wave→SIMD` map.

| name                                | params                              | HW expect           | diagnoses |
|-------------------------------------|-------------------------------------|---------------------|-----------|
| mb_sched_order_wg64                 | WG=64                               | 2 waves, round-robin| scheduler order |
| mb_sched_order_wg128                | WG=128                              | 4 waves, striped    | scheduler order |
| mb_sched_order_wg256                | WG=256                              | 8 waves             | scheduler order |
| mb_sched_order_interleave_vmem_valu | 2 waves mix                         | placement stable?   | vmem-placement bias |
| mb_sched_same_wgp                   | small grid, few WGs                 | WG→WGP placement    | WGP mapping |
| mb_sched_cross_wgp                  | >4 WGs                              | spread across WGPs  | cross-WGP scheduling |
| mb_sched_wg_boundary_barrier        | WG=64, barrier                      | barrier placement   | WG-barrier |
| mb_sched_wg128_lds_heavy            | LDS heavy                           | co-loc on CU        | LDS affinity |
| mb_sched_wg128_vmem_heavy           | VMEM heavy                          | spread              | VMEM bandwidth |
| mb_sched_wg64_mixed                 | 50% valu, 50% smem                  | mixed               | baseline |

## C.6 — Scalar pipe beat/phase (8 kernels)

Resolve E1/E2/E3 scalar-phase effect.

| name                                | params                                   | HW expect           | diagnoses |
|-------------------------------------|------------------------------------------|---------------------|-----------|
| mb_scalar_beat_phase0               | s_mov,s_cmp,s_cbranch (tight)            | 1,1,8               | E1        |
| mb_scalar_beat_phase1               | s_mov,s_nop(0),s_cmp,s_cbranch           | 1,3,1,13            | E1        |
| mb_scalar_beat_phase2               | s_mov,s_nop(0)×2,s_cmp,s_cbranch         | 1,3,1,1,13          | E1        |
| mb_scalar_beat_phase3               | s_mov,s_nop(0)×3,s_cmp,s_cbranch         | 1,3,1,1,1,13        | E1        |
| mb_scalar_after_vmem_k0             | waitcnt_vmcnt,s_mov,s_cmp,s_cbranch      | 1,1,9-10            | E2/E3     |
| mb_scalar_after_vmem_k4             | +4 s_nops before s_mov                   | 1,1,8               | E2/E3     |
| mb_scalar_after_vmem_k8             | +8 s_nops before s_mov                   | 1,1,8               | E2/E3 drained |
| mb_scalar_after_vmem_k16            | +16 s_nops                               | 1,1,8               | E2/E3 fully drained |

---

# Batch D — Stress kernels (30 kernels, ~1.8 h)

Adaptations of Seb-v's Kernel 6 (dual-FMA double-buffered matmul). Validates
emulator behavior on complex, real-workload code paths once the micro-bench
constants are correct.

## D.1 — Tile size sweep (9 kernels)

For each of `{8×8, 16×16, 32×32}` tile × `{unroll=1, unroll=2, unroll=4}`:

| name                                | tile | unroll | HW expect           | diagnoses |
|-------------------------------------|------|--------|---------------------|-----------|
| mb_ss_matmul_t8_u1                  | 8×8  | 1      | baseline low FLOP   | small-tile |
| mb_ss_matmul_t8_u2                  | 8×8  | 2      |                     | unroll effect |
| mb_ss_matmul_t8_u4                  | 8×8  | 4      | bound by LDS        | unroll saturation |
| mb_ss_matmul_t16_u1                 | 16×16| 1      |                     | mid tile   |
| mb_ss_matmul_t16_u2                 | 16×16| 2      | VOPD dual-fmac      | VOPD sustained |
| mb_ss_matmul_t16_u4                 | 16×16| 4      | close to peak       | matmul peak |
| mb_ss_matmul_t32_u1                 | 32×32| 1      | VGPR pressure       | occupancy  |
| mb_ss_matmul_t32_u2                 | 32×32| 2      | reg-bound           | occupancy  |
| mb_ss_matmul_t32_u4                 | 32×32| 4      | max tile            | extreme pressure |

## D.2 — LDS padding on/off (6 kernels)

| name                                | tile | pad  | HW expect                | diagnoses |
|-------------------------------------|------|------|--------------------------|-----------|
| mb_ss_lds_t8_pad0                   | 8×8  | no   | bank conflict cost       | conflict  |
| mb_ss_lds_t8_pad1                   | 8×8  | yes  | no conflict              | padding   |
| mb_ss_lds_t16_pad0                  | 16×16| no   | 2× conflict              | conflict  |
| mb_ss_lds_t16_pad1                  | 16×16| yes  | baseline                 | padding   |
| mb_ss_lds_t32_pad0                  | 32×32| no   | heavy conflict           | conflict  |
| mb_ss_lds_t32_pad1                  | 32×32| yes  | baseline                 | padding   |

## D.3 — Double-buffer stages (5 kernels)

| name                                | buffers | HW expect                 | diagnoses |
|-------------------------------------|---------|---------------------------|-----------|
| mb_ss_db_stages_1                   | 1       | sequential, no overlap    | baseline  |
| mb_ss_db_stages_2                   | 2       | ping-pong, classic DB     | full DB   |
| mb_ss_db_stages_3                   | 3       | triple-buffer             | TB        |
| mb_ss_db_stages_4                   | 4       | quad-buffer               | 4-buf     |
| mb_ss_db_overlap_metric             | 2 + metric| measure overlap ratio   | DB efficiency |

## D.4 — VOPD saturation workloads (5 kernels)

| name                                | body                               | HW expect           | diagnoses |
|-------------------------------------|------------------------------------|---------------------|-----------|
| mb_ss_vopd_fmac_16                  | 16 VOPD fmac_fmac                  | 1×16, matmul-ish    | VOPD sustained |
| mb_ss_vopd_fmac_32                  | 32 VOPD fmac_fmac                  | 1×32                | VOPD sustained |
| mb_ss_vopd_mul_add_16               | 16 VOPD mul+add                    | 1×16                | mixed VOPD |
| mb_ss_vopd_chain_with_lds           | VOPD + LDS interleave              | LDS forwarding hits | forwarding |
| mb_ss_vopd_with_vmem_prefetch       | VOPD + global_load prefetch        | overlap observed    | prefetch overlap |

## D.5 — Real-workload mini-benchmarks (5 kernels)

| name                                | workload                           | HW expect                 | diagnoses |
|-------------------------------------|------------------------------------|---------------------------|-----------|
| mb_ss_softmax_64                    | softmax over 64                    | validates exp+reduce      | softmax   |
| mb_ss_layernorm_128                 | layernorm over 128                 | validates sqrt/rcp        | layernorm |
| mb_ss_gemm_8x8                      | 8×8 GEMM stripe                    | validates VOPD            | GEMM      |
| mb_ss_reduce256                     | 256-elt tree reduce                | validates barrier+LDS     | reduce    |
| mb_ss_reduce_large                  | 4096-elt reduce                    | validates multi-pass      | reduce-big |

---

# Capture-run sequencing

1. **Run Batch A first** (~4.5 h). Outputs calibrate single-op constants
   (`VALU_PIPELINE_LATENCY`, `TRANS_PIPELINE_LATENCY*`, `SGPR_LATENCY`,
   `CNDMASK_SGPR_LATENCY`, `_VMEM_DRAIN_CYCLES`, `_VOPD_PIPE_CYCLES`,
   `s_nop` predecessor formulas, `s_cbranch` fast/slow path). Expected
   to close B2/B3/B4 (s_nop formula), C8 (trans cold-warm), E1 (cbranch
   fast path), and validate A-family stores (A1/A2/A3) as 17/21 bimodal
   behavior of `global_store_b32`.

2. **Run Batch D before B** (~1.8 h). Wave-pair and co-residence tests expose
   whether shared-issuer slip (D1, F4) actually exists on specific WG shapes
   that the current probe set doesn't reach. This gates whether to build the
   wave-pair-arbiter model at all.

3. **Run Batch B** (~8.5 h). Resolves bank-conflict + VOPD eligibility +
   cndmask/trans interaction. Closes C-family mismatches (12 tokens) and
   solidifies VOPD_LIT / VOPD-chain / depctr-reset behavior. Also settles
   F1/F3/F5 fresh-VGPR vs SGPR-retire hypothesis.

4. **Run Batch C** (~3.2 h) last, after B constants are known. Occupancy
   and scheduler probes confirm end-to-end behavior and reveal any remaining
   inter-wave timing issues that single-op probes can't pin down.

Overnight capture: ~18 batches × 20 min ≈ 6 hours for the main pass, then
re-runs for probes that didn't land wave-1 (typically 3-5 reruns at
~1 hour each). Full campaign: two overnight runs.

## Mismatch → kernel coverage summary

| Mismatch group        | Primary probes                                          | Secondary confirmation        |
|-----------------------|---------------------------------------------------------|-------------------------------|
| A1/A2/A3              | A.10 mb_vmem_store_b32_{isolated,after_load,chain_n4,spaced_nopK} | B.6 mb_cores_wg128_store_burst, C.4 mb_wp_store_both |
| B1/B2/B3/B4           | A.7 mb_snop_15_chain_n{2,3,5,8}, mb_snop_15_after_* | B.5 mb_trans_then_snop, mb_trans_chain4_then_snop |
| C1                    | B.3.b mb_vopd_post_depctr, A.8 mb_waitcnt_depctr_4095 | B.3.b mb_vopd_chain_n4_{no_raw,raw} |
| C2/C9                 | B.4 mb_vcmp_cndmask_k{1,2,4,8}, mb_vcmp_sgpr_k_sweep  | B.4 mb_cndmask_sgpr_k_sweep |
| C3/C10                | A.5 mb_cndmask_sgpr_fresh_n4, B.4 mb_vcmp_cndmask_k{2,4} | B.4 mb_vcmp_spaced_cndmask_* |
| C4/C5/C11             | B.4 mb_vcmp_cndmask_k4, B.4 mb_vcmp_interleave_cndmask | A.5 mb_cndmask_tail_gt4 |
| C6/C7/C12             | B.3.b mb_vopd_{chain_n4_raw, pair_raw_x, pair_no_raw} | B.3.b mb_vopd_post_depctr |
| C8                    | A.3 mb_trans_cold_vs_warm, B.5 mb_trans_cold_warm_alternating | B.5 mb_trans_nested_valu |
| D1                    | C.4 mb_wp_cold_start_n{1,2,4,8}, mb_wp_cold_start_staggered | C.4 mb_wp_warm_start_n4 |
| E1/E2/E3              | C.6 mb_scalar_beat_phase{0,1,2,3}, mb_scalar_after_vmem_k{0,4,8,16} | A.6 mb_salu_scmp_* |
| F1/F3                 | B.4 mb_cndmask_tail_retire, mb_cndmask_tail_new_sgpr | A.5 mb_cndmask_tail_gt4 |
| F2/F5                 | A.5 mb_cndmask_dst_fresh_vgpr_high, B.4 mb_cndmask_new_vgpr_tail | B.6 mb_cores_wg128_cndmask_burst |
| F4                    | C.4 mb_wp_trans_both, mb_wp_trans_one_only           | B.5 mb_trans_pair_{same,diff}_op |

Every one of the 28 mismatched tokens is covered by at least two probes, at
least one of which sweeps the suspected trigger parameter continuously. That
turns each cluster of mismatches into a **1-D sensitivity curve** instead of
a single data point, which is what the emulator rewrite needs to converge.
