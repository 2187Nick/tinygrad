#!/usr/bin/env python3
# Run all ALU and memory instructions in the ISA
import functools, inspect
from enum import Enum
from tinygrad import Tensor, Device, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AddrSpace
from tinygrad.renderer.amd.dsl import Inst, Reg, OPERANDS, SrcField, VGPRField, SGPRField, SSrcField, SBaseField, AlignedSGPRField, BitField
from tinygrad.renderer.amd.dsl import FixedBitField, EnumBitField, s, v, NULL, VCC_LO
from extra.gemm.amd_asm_matmul import Kernel

# skip instructions that mutate wave state (PC, EXEC, allocations, signals) or hit emulator limitations
SKIP = {"S_SETPC_B64", "S_SWAPPC_B64", "S_RFE_B64", "S_BARRIER_SIGNAL_ISFIRST", "S_GET_BARRIER_STATE", "S_ALLOC_VGPR", "S_SLEEP_VAR", "S_GETPC_B64",
        "S_SENDMSG_RTN_B32", "S_SENDMSG_RTN_B64", "S_CALL_B64", "S_CMOV_B64",
        "S_CVT_PK_RTZ_F16_F32", "S_QUADMASK_B64", "S_SETREG_B32", "S_SETREG_IMM32_B32", "S_WQM_B64",
        "V_DIV_FIXUP_F16", "V_EXP_F16_E32", "V_EXP_F16_E64", "V_FMAC_DX9_ZERO_F32_E32",
        "V_FREXP_EXP_I16_F16_E32", "V_FREXP_EXP_I16_F16_E64", "V_FREXP_MANT_F16_E32", "V_FREXP_MANT_F16_E64",
        "V_MULLIT_F32", "V_SWAPREL_B32_E32", "V_WMMA_I32_16X16X16_IU4", "V_WMMA_I32_16X16X16_IU8"}
# skip barriers, s_waits, DS atomics, ray tracing (bvh), and other unsupported ops
SKIP_SUBSTR = ["SAVEEXEC", "CMPX", "WREXEC", "MOVREL", "ATOMIC", "S_BUFFER_", "S_ATC_PROBE", "BARRIER", "S_WAITCNT", "BVH",
               "DS_CMPSTORE", "DS_WRAP_RTN_B32", "DS_ORDERED_COUNT", "DS_GWS", "GS_REG", "GLOBAL_LOAD_LDS", "GLOBAL_STORE_BLOCK",
               "DS_ADD_", "DS_SUB_", "DS_RSUB_", "DS_MIN_", "DS_MAX_", "DS_AND_", "DS_OR_", "DS_XOR_", "DS_MSKOR_",
               "DS_INC_", "DS_DEC_", "DS_STOREXCHG_", "DS_CONDXCHG", "DS_APPEND", "DS_CONSUME", "DS_SWIZZLE_",
               "ADDTID", "V_INTERP_P"]

ALU_FORMATS = {"VOP1", "VOP1_LIT", "VOP1_SDST", "VOP2", "VOP2_LIT", "VOP3", "VOP3_SDST", "VOP3SD", "VOP3P", "VOP3P_MFMA", "VOP3PX2",
               "VOPC", "SOP1", "SOP1_LIT", "SOP2", "SOP2_LIT", "SOPC", "SOPC_LIT", "SOPK", "SOPK_LIT", "VINTERP"}
# intentionally not testing scratch memory ops
MEM_FORMATS = {"VGLOBAL", "GLOBAL", "SMEM", "DS"}

def should_skip(op):
  return (name:=op.name) in SKIP or any(sub in name for sub in SKIP_SUBSTR)

ALU_VGPR_STRIDE = 16
ALU_SGPR_STRIDE = 4
S_KERNARG_PTR = (0, 1)
S_BUF_PTR = (2, 3)
V_VADDR = (0, 1)
V_DS_ADDR = 0
MEM_VGPR_BASE = 32
MEM_VGPR_STRIDE = 16
MEM_SGPR_BASE = 8
MEM_SGPR_STRIDE = 2

def create_alu_inst(op, builder):
  inst_cls, operands, slot = builder.func, OPERANDS[op], 0
  kwargs = {}
  for name, field in inst_cls._fields:
    if isinstance(field, (FixedBitField, EnumBitField)): continue
    nregs = max(1, operands[name][1] // 32) if name in operands else 1
    is_sreg = name in operands and "SREG" in str(operands[name][2])
    base_v, base_s = slot * ALU_VGPR_STRIDE, slot * ALU_SGPR_STRIDE
    if name == "sdst" and isinstance(field, SGPRField): reg = VCC_LO
    elif is_sreg and not isinstance(field, VGPRField): reg = VCC_LO
    elif isinstance(field, VGPRField): reg = v[base_v:base_v+nregs-1] if nregs > 1 else v[base_v]
    elif isinstance(field, SSrcField): reg = VCC_LO if nregs <= 2 else s[base_s:base_s+nregs-1] if nregs > 1 else s[base_s]
    elif isinstance(field, SGPRField): reg = s[base_s:base_s+nregs-1] if nregs > 1 else s[base_s]
    elif isinstance(field, SrcField): reg = v[base_v:base_v+nregs-1] if nregs > 1 else v[base_v]
    else: reg = None
    if reg is not None: kwargs[name] = reg; slot += 1
    elif isinstance(field, BitField): kwargs[name] = field.default
  return builder(**kwargs)

MEM_PRESET_REGS = {
  "VGLOBAL":{"saddr":s[S_BUF_PTR[0]:S_BUF_PTR[1]], "vaddr":v[V_VADDR[0]:V_VADDR[1]]},
  "GLOBAL":{"saddr":s[S_BUF_PTR[0]:S_BUF_PTR[1]], "addr":v[V_DS_ADDR]},
  "DS":{"addr":v[V_DS_ADDR]},
  "SMEM":{"sbase":s[S_KERNARG_PTR[0]:S_KERNARG_PTR[1]], "soffset":NULL},
}

def create_mem_inst(op, builder):
  inst_cls, operands, field_map = builder.func, OPERANDS.get(op, {}), MEM_PRESET_REGS.get(builder.func.__name__, {})
  kwargs = {}
  vslot, sslot = 0, 0
  for name, field in inst_cls._fields:
    if isinstance(field, (FixedBitField, EnumBitField)): continue
    if name in field_map: kwargs[name] = field_map[name]; continue
    nregs = max(1, operands[name][1] // 32) if name in operands else 1
    if isinstance(field, VGPRField):
      vi = MEM_VGPR_BASE + vslot * MEM_VGPR_STRIDE
      kwargs[name] = v[vi:vi+nregs-1] if nregs > 1 else v[vi]; vslot += 1
    elif isinstance(field, (SGPRField, AlignedSGPRField, SBaseField)):
      si = MEM_SGPR_BASE + sslot * MEM_SGPR_STRIDE
      kwargs[name] = s[si:si+nregs-1] if nregs > 1 else s[si]; sslot += 1
    elif isinstance(field, BitField): kwargs[name] = field.default
  return builder(**kwargs)

def collect_instructions():
  op_map = {}
  for name, obj in inspect.getmembers(all_insts):
    if isinstance(obj, functools.partial) and len(obj.args) == 1: op_map[obj.args[0]] = obj
  alu_insts, mem_insts, skipped = [], [], []
  for op_enum, builder in op_map.items():
    if should_skip(op_enum) or op_enum not in OPERANDS: skipped.append(op_enum.name); continue
    fmt = builder.func.__name__
    if fmt in ALU_FORMATS: alu_insts.append(create_alu_inst(op_enum, builder))
    elif fmt in MEM_FORMATS: mem_insts.append(create_mem_inst(op_enum, builder))
  return alu_insts, mem_insts, skipped

def exec_insts(insts):
  k = Kernel(arch)
  k.emit(s_load_b64(sdata=s[S_BUF_PTR[0]:S_BUF_PTR[1]], sbase=s[S_KERNARG_PTR[0]:S_KERNARG_PTR[1]], soffset=NULL))
  k.waitcnt(lgkm=0)
  k.emit(v_mov_b32_e32(v[V_VADDR[0]], 0))
  k.emit(v_mov_b32_e32(v[V_VADDR[1]], 0))
  for inst in insts: k.emit(inst)
  k.emit(s_endpgm())
  NUM_THREADS, NUM_GRIDS, BUF_SIZE = 32, 1, 1024*1024
  def fxn(A, B, C):
    lidx, gidx = UOp.special(NUM_THREADS, "lidx0"), UOp.special(NUM_GRIDS, "gidx0")
    lds = UOp(Ops.DEFINE_LOCAL, dtypes.uint8.ptr(size=BUF_SIZE, addrspace=AddrSpace.LOCAL), (), "lds")
    sink = UOp.sink(A.base, B.base, C.base, lds, lidx, gidx, arg=KernelInfo(name="discover_ops"))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD"), UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=x) for x in k.finalize()))))
  A = Tensor.empty(BUF_SIZE, dtype=dtypes.uint8)
  B = Tensor.empty(1, dtype=dtypes.uint8)
  C = Tensor.empty(1, dtype=dtypes.uint8)
  Tensor.custom_kernel(A, B, C, fxn=fxn)[0].realize()

if __name__ == "__main__":
  import sys
  arch = Device[Device.DEFAULT].renderer.target.arch
  if arch.startswith("gfx12"):
    from tinygrad.runtime.autogen.amd.rdna4.ins import *
    import tinygrad.runtime.autogen.amd.rdna4.ins as all_insts
  elif arch.startswith("gfx11"):
    from tinygrad.runtime.autogen.amd.rdna3.ins import *
    import tinygrad.runtime.autogen.amd.rdna3.ins as all_insts
    SKIP.update(["S_FMAAK_F32", "S_FMAMK_F32"])
  else:
    print(f"{arch} not supported yet"); sys.exit(0)
  alu_insts, mem_insts, skipped = collect_instructions()
  print(f"collected {len(alu_insts)} ALU + {len(mem_insts)} memory instructions ({len(skipped)} skipped)")
  exec_insts(mem_insts+alu_insts)
