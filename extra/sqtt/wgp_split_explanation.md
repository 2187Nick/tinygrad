# Why 16-wave workgroups split even/odd across two WGPs

**Discovery:** HW_ID probe data showed that 16-wave workgroups place waves
0,2,4,…,14 on one WGP's SIMD 0 and waves 1,3,5,…,15 on a *different* WGP's
SIMD 0 of the same Shader Engine. The two WGPs are always 8 apart in the
global WGP numbering (WGP 0 ↔ WGP 8, WGP 4 ↔ WGP 12, …) — i.e. they are
the "first WGP of SA0" and the "first WGP of SA1" of the same SE.

**Mechanism** (grounded in the Linux `amdgpu` driver, not speculation):

## 1. CU mask register layout puts SA0 and SA1 in one register

From `drivers/gpu/drm/amd/include/asic_reg/gc/gc_11_0_0_sh_mask.h:14725–14728`:

```
COMPUTE_STATIC_THREAD_MGMT_SE0__SA0_CU_EN__SHIFT   0x0      # bits [15:0]  = SA0 CUs
COMPUTE_STATIC_THREAD_MGMT_SE0__SA1_CU_EN__SHIFT   0x10     # bits [31:16] = SA1 CUs
COMPUTE_STATIC_THREAD_MGMT_SE0__SA0_CU_EN_MASK     0x0000FFFF
COMPUTE_STATIC_THREAD_MGMT_SE0__SA1_CU_EN_MASK     0xFFFF0000
```

A single register enables CUs on *both* Shader Arrays of one Shader Engine
simultaneously. When the CP activates a queue, whichever CUs are enabled
(SA0 *and* SA1) will compete for waves from that SE's dispatch queue.

## 2. amdgpu encodes the mask with SE as the inner loop

From `drivers/gpu/drm/amd/amdgpu/amdgpu_gfx.c:506–553` (`amdgpu_gfx_mqd_symmetrically_map_cu_mask`):

```c
bool wgp_mode_req = amdgpu_ip_version(adev, GC_HWIP, 0) >= IP_VERSION(10, 0, 0);
int cu_inc = wgp_mode_req ? 2 : 1;
uint32_t en_mask = wgp_mode_req ? 0x3 : 0x1;
...
for (sh = 0; sh < max_sh_per_se; sh++)       // OUTER: SA
  for (se = 0; se < max_shader_engines; se++) // INNER: SE
    ...
    se_mask[se] |= en_mask << (cu + sh * 16);  // SA1 in bits 16..31
```

Combined, the MQD ends up enabling both SAs of each SE for compute, which
wires **two independent SPIs (one per SA) onto the same dispatch SE**.

## 3. Two SPIs pulling the same dispatch queue → alternating allocation

Each Shader Array has its own SPI (ROCm HIP docs: *"The workgroup manager,
also called the shader processor input (SPI), bridges the command processor
and compute units"*). When the CP hands a 16-wave workgroup to an SE, both
SPIs are live consumers of that SE's wave-dispatch stream. As waves come
off the queue, they alternate — SA0 → SA1 → SA0 → SA1 — because whichever
SPI is ready in a given cycle grabs the next wave. That yields the observed
even/odd split:

- wave 0 → SA0 (your "WGP 0"), slot 0
- wave 1 → SA1 (your "WGP 8"), slot 0
- wave 2 → SA0, slot 1
- wave 3 → SA1, slot 1
- …
- wave 14 → SA0, slot 7
- wave 15 → SA1, slot 7

Both SA0 and SA1 always use SIMD 0 within their WGP (the earlier 1270/1270
SIMD_ID=0 observation holds per-SA; the total is really 1270 split as 635
waves on SA0's SIMD 0 and 635 waves on SA1's SIMD 0).

## 4. Why wave 1 doesn't always win at first VALU

Even though wave 1 is on a **different** SA from wave 0, it can still stall
at its first VALU post-waitcnt. 50×100 capture data (100 runs per kernel):

| kernel | wave-1 first-VALU dt=1cy | next modes |
|---|---|---|
| `mb_valu_add_n8`  | 96/100 | rest ≤5 |
| `mb_valu_add_n16` | 47/100 | spread 3, 6, 9, 15… |
| `mb_e1_valu_add_n24` | 21/100 | 43cy (18), 39cy (6), 46cy (5) |
| `mb_f1_valu_add_n32` | 14/100 | 101cy (17), 98cy (9), 100cy (7) |

Wave 1 is **stochastically** slow — dominantly fast on short chains, dominantly
slow on long chains. Hypothesis: SA1's SPI is sometimes busy servicing
some other workload or finishing a prior SE dispatch, and the *probability*
that it's busy when wave 1 arrives grows with upstream queue depth (which
correlates with the post-waitcnt latency of the compiled program). This is
exactly the kind of arbitration variance that a stochastic scheduler would
need to model — a single deterministic pass over emu cannot reproduce it.

## 5. Implications for the emu

- `SimdArbiter.simd_for_wave(w) = w & 0x1` (2 peer groups) is mechanically
  correct; `N_SIMDS=2` is the right port count.
- A pure peer-gate on the wave-credit RAW rule regresses strict by 22
  (`EMU_WGP_PEER_GATE=1` test): waves 1, 3, 5, … still queue in practice
  because of the stochastic SA1-busy effect described in §4.
- The path to closing this stochastically is **not** a single deterministic
  rule — it's a probability model over wave 1, 3, 5… (matches the BIMODAL
  100-run classifier category). See `STOCHASTIC_SCHEDULER_PLAN.md` §1, §3.
- The path to closing this **deterministically in strict mode** is:
  (a) make reference captures consistent (each kernel's reference pkl
  represents the *modal* HW outcome, not a random single run), or
  (b) simulate both SPI arbitration outcomes and have the comparator accept
  a match against either outcome per wave (a strict-mode MODAL-adjacent).

## 6. Single-CU / single-WGP pinning for future probes

`gc_11_0_0_sh_mask.h` bit layout (`COMPUTE_STATIC_THREAD_MGMT_SEn`):

- bits `[0]`..`[15]` = 16 bits → 8 WGPs × 2 CUs of SA0
  (bit 0 = CU0 of WGP0, bit 1 = CU1 of WGP0, bit 2 = CU0 of WGP1, …)
- bits `[16]`..`[31]` = SA1 equivalent.

To pin all compute to **one CU only** (e.g. SE0 / SA0 / WGP0 / CU0):

```
COMPUTE_STATIC_THREAD_MGMT_SE0 = 0x00000001   # bit 0 alone
COMPUTE_STATIC_THREAD_MGMT_SE1..SE5 = 0x0     # all other SEs disabled
```

Note the kernel's default writer (`amdgpu_gfx.c:540–545`) uses `en_mask=0x3`
(pairs of bits, WGP-granular). Sub-WGP CU pinning needs the MQD field
written directly (bypassing the helper). That's what `cu_reservation_probe.py`
in mes_notes §5.2 needs.

## References

- `drivers/gpu/drm/amd/amdgpu/gfx_v11_0.c` L4255–4298 (MQD CU mask writer)
- `drivers/gpu/drm/amd/amdgpu/amdgpu_gfx.c` L506–553 (mask iteration logic)
- `drivers/gpu/drm/amd/include/asic_reg/gc/gc_11_0_0_sh_mask.h` L14725–14728
- `drivers/gpu/drm/amd/include/asic_reg/gc/gc_11_0_0_offset.h` L4022–4025 (SE4/SE5 offsets)
- [ROCm HIP: Hardware implementation](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html)
- RDNA3 ISA Reference Guide (Feb 2023) — AMD
- RDNA Architecture white paper — GPUOpen
- Local empirical evidence: `extra/sqtt/wave_probe/captures/hw_id_20260420_102105.json`,
  `extra/sqtt/wave_probe/captures/hwid_per_kernel/hwid_n0512.json`
