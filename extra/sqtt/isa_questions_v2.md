# ISA Questions for RDNA3 Team

Current status: **286/347 exact (82.4%), 324/347 ±2 (93.4%)** on rigorous suite incl. probes.

Three specific unknowns where the emulator's per-packet distribution still mismatches HW SQTT. Cumulative wave time is usually correct; it's WHICH packet absorbs the stall that we can't explain.

---

## Q1: v_cndmask_b32_e64 with non-VCC SGPR — position-dependent delta

When a chain of `v_cmp_*_e64` + `v_cndmask_b32_e64` reads a non-VCC SGPR as its condition, we measure DIFFERENT per-instruction deltas at different positions in the chain even though the micro-op sequence is identical.

**probe_sgpr_cmps** pattern (same block repeated twice with drain between):
- `v_cmp_gt_f32_e64` writes `s[4]`
- `v_cmp_lt_f32_e64` writes `s[5]`
- `v_cmp_gt_f32_e64` writes `s[6]`
- `v_cmp_lt_f32_e64` writes `s[7]`
- `v_cndmask_b32_e64` reads `s[4]` → HW delta varies **1 or 3 cy** (Block A vs Block B)
- `v_cndmask_b32_e64` reads `s[5]` → HW delta ≈ 2 cy consistently
- `v_cndmask_b32_e64` reads `s[6]` → HW delta varies **1 or 2 cy**
- `v_cndmask_b32_e64` reads `s[7]` → HW delta ≈ 1 cy

**Question:** Is the SGPR "not-VCC" read stall position-dependent inside a chain? E.g., does the scalar register read-port arbitrate round-robin, so a v_cndmask reading s[4] may hit a free port or a busy port depending on whether s[5]/s[6]/s[7] readers are still pending?

Or: is there a **scalar operand collector** with limited bandwidth that changes the first-packet cost after a drain (barrier/waitcnt)?

---

## Q2: s_nop(N) SQTT token timing — confirmation

Measurement from probe_vmem_chain:
- `s_nop(10)` (11 stall cycles): HW SQTT token records delta = **13** from prev inst's stamp.

Empirical model we're using now: `stamp = nop_issue + nop_cycles + 1`.

**Question:** Is there an official cycle relationship between `s_nop(N)` issue and its SQTT token emission? Is it `N + 2` cycles post-issue? What does the token represent — wave exit from IB wait, wave re-enter at next fetch, or something else?

---

## Q3: Scalar branch NOT-TAKEN on scc — cost variance

probe_branch_cost: two back-to-back `s_cbranch_scc1(not-taken)` show HW deltas of **8 and 9** on the branch's own SQTT token, with next instruction having delta **3**.

**Question:** What is the documented cycle cost of `s_cbranch_scc1` when NOT-TAKEN on RDNA3? Is the observed 8-10cy variance due to:
- Scalar pipeline redirect latency
- SCC read hazard depth when immediately after `s_cmp`
- Branch predictor state

Specifically: does an `s_cmp` → `s_cbranch_scc1` pair have a documented minimum gap when the CC is consumed before writeback?

---

## Bonus — VALU→VMEM forwarding after s_nop

probe_vmem_chain [7]→[8]: `s_nop(10)` then `global_store_b32`. HW shows store delta = **5** after nop. Our emu model says VALU→VMEM_WR forwarding deadline = VALU_issue + 21cy, which puts the first store ~22cy after nop (because the VALU feeding the store was issued before the nop).

**Question:** Does `s_nop(N)` count toward the VALU writeback→VMEM_WR forwarding window? If the VALU write committed during the nop stall, the forwarding deadline should effectively be shortened by N+1 cycles.
