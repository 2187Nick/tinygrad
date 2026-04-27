import ctypes, hashlib, tempfile, subprocess, pathlib, shutil
from tinygrad.helpers import system, getenv, ContextVar
from tinygrad.runtime.autogen import comgr
try:
  comgr.amd_comgr_get_version(ctypes.byref(major:=ctypes.c_uint64()), ctypes.byref(minor:=ctypes.c_uint64()))
  if major.value >= 3:
    # in comgr 3 the values of enums in headers were changed: https://github.com/ROCm/llvm-project/issues/272
    import tinygrad.runtime.autogen.comgr_3 as comgr # type: ignore[no-redef]
    assert comgr.AMD_COMGR_LANGUAGE_HIP == 3
except AttributeError: pass  # ignore if ROCm isn't installed
from tinygrad.device import Compiler, CompileError
from tinygrad.runtime.support.compiler_cpu import LLVMCompiler
from tinygrad.runtime.support import c
from tinygrad.helpers import OSX, to_char_p_p

# Opt-in flag: when set, AMD compilers will also produce/preserve the unstripped
# AMDGPU HSACO (with .symtab + .note.AMDGPU.metadata) via a full comgr pipeline,
# making the artifact consumable by RGA's --livereg / --isa modes. Default 0 to
# guarantee byte-identical `lib` outputs for existing callers.
KEEP_FULL_HSACO = ContextVar("KEEP_FULL_HSACO", 0)

# Side-channel: maps md5(stripped_lib) -> full unstripped HSACO bytes. Populated
# by the AMD compilers when KEEP_FULL_HSACO=1 and read by AMDProgram.__init__.
# Module-level so it survives the diskcache (which only persists `lib`).
FULL_HSACO_STASH: dict[bytes, bytes] = {}

def stash_full_hsaco(lib:bytes, full:bytes) -> None: FULL_HSACO_STASH[hashlib.md5(lib).digest()] = full
def get_full_hsaco(lib:bytes) -> bytes|None: return FULL_HSACO_STASH.get(hashlib.md5(lib).digest())

def _find_llvm_objdump():
  if OSX: return '/opt/homebrew/opt/llvm/bin/llvm-objdump'
  # Try ROCm path first, then versioned, then unversioned
  for p in ['/opt/rocm/llvm/bin/llvm-objdump', 'llvm-objdump-21', 'llvm-objdump-20', 'llvm-objdump']:
    if shutil.which(p): return p
  raise FileNotFoundError("llvm-objdump not found")

def amdgpu_disassemble(lib:bytes):
  asm = system(f"{_find_llvm_objdump()} -d -", input=lib).splitlines()
  while asm and ("s_nop 0" in asm[-1] or "s_code_end" in asm[-1]): asm.pop()
  print("\n".join(asm))

def check(status):
  if status != 0:
    comgr.amd_comgr_status_string(status, ctypes.byref(status_str := ctypes.POINTER(ctypes.c_char)()))
    raise RuntimeError(f"comgr fail {status}, {ctypes.string_at(status_str).decode()}")

def _get_comgr_data(data_set, data_type):
  check(comgr.amd_comgr_action_data_get_data(data_set, data_type, 0, ctypes.byref(data_exec := comgr.amd_comgr_data_t())))
  check(comgr.amd_comgr_get_data(data_exec, ctypes.byref(sz := ctypes.c_uint64()), None))
  check(comgr.amd_comgr_get_data(data_exec, ctypes.byref(sz), (dat := ctypes.create_string_buffer(sz.value))))
  check(comgr.amd_comgr_release_data(data_exec))
  return bytes(dat)

# amd_comgr_action_info_set_options was deprecated
def set_options(action_info, options:bytes):
  # TODO: this type should be correct in the autogen stub
  @comgr.dll.bind(comgr.amd_comgr_status_t, comgr.amd_comgr_action_info_t, c.POINTER[c.POINTER[ctypes.c_char]], comgr.size_t)
  def amd_comgr_action_info_set_option_list(ai, o, c) -> comgr.amd_comgr_status_t: pass # type: ignore[empty-body]
  return amd_comgr_action_info_set_option_list(action_info, to_char_p_p(options_list:=options.split(b' ')), len(options_list))

# AMD_COMGR_SAVE_TEMPS=1 AMD_COMGR_REDIRECT_LOGS=stdout AMD_COMGR_EMIT_VERBOSE_LOGS=1
def compile_hip(prg:str, arch="gfx1100", asm=False, keep_full:bool=False) -> bytes|tuple[bytes, bytes]:
  """Compile a HIP source string to an AMDGPU executable via comgr.

  When `keep_full` is True, returns (lib, full_hsaco) where `full_hsaco` is the
  pre-link relocatable object — it still has `.symtab`, `.note.AMDGPU.metadata`,
  and per-kernel symbols, which RGA's `--livereg` / `--isa` modes require. The
  default-mode `lib` (the linked executable) is returned unchanged.
  """
  check(comgr.amd_comgr_create_action_info(ctypes.byref(action_info := comgr.amd_comgr_action_info_t())))
  check(comgr.amd_comgr_action_info_set_language(action_info, comgr.AMD_COMGR_LANGUAGE_HIP))
  check(comgr.amd_comgr_action_info_set_isa_name(action_info, b"amdgcn-amd-amdhsa--" + arch.encode()))
  check(comgr.amd_comgr_action_info_set_logging(action_info, True))

  check(comgr.amd_comgr_create_data_set(ctypes.byref(data_set_src := comgr.amd_comgr_data_set_t())))
  check(comgr.amd_comgr_create_data_set(ctypes.byref(data_set_bc := comgr.amd_comgr_data_set_t())))
  check(comgr.amd_comgr_create_data_set(ctypes.byref(data_set_reloc := comgr.amd_comgr_data_set_t())))
  check(comgr.amd_comgr_create_data_set(ctypes.byref(data_set_exec := comgr.amd_comgr_data_set_t())))

  check(comgr.amd_comgr_create_data(comgr.AMD_COMGR_DATA_KIND_SOURCE, ctypes.byref(data_src := comgr.amd_comgr_data_t())))
  check(comgr.amd_comgr_set_data(data_src, len(rprg := prg.encode()), rprg))

  if asm:
    check(comgr.amd_comgr_set_data_name(data_src, b"<null>.s"))
    check(comgr.amd_comgr_data_set_add(data_set_src, data_src))
    status = comgr.amd_comgr_do_action(comgr.AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE, action_info, data_set_src, data_set_reloc)
    if status != 0:
      print(_get_comgr_data(data_set_reloc, comgr.AMD_COMGR_DATA_KIND_LOG).decode())
      raise RuntimeError("assemble failed")
  else:
    check(comgr.amd_comgr_set_data_name(data_src, b"<null>"))
    check(comgr.amd_comgr_data_set_add(data_set_src, data_src))
    # -include hiprtc_runtime.h was removed
    options = [
      "-O3", "-mcumode", "--hip-version=6.0.32830", "-DHIP_VERSION_MAJOR=6", "-DHIP_VERSION_MINOR=0", "-DHIP_VERSION_PATCH=32830",
      "-D__HIPCC_RTC__", "-std=c++14", "-nogpuinc", "-Wno-gnu-line-marker", "-Wno-missing-prototypes", f"--offload-arch={arch}",
      "-I/opt/rocm/include", "-Xclang -disable-llvm-passes", "-Xclang -aux-triple", "-Xclang x86_64-unknown-linux-gnu"]
    check(set_options(action_info, ' '.join(options).encode()))
    status = comgr.amd_comgr_do_action(comgr.AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC, action_info, data_set_src, data_set_bc)
    if status != 0:
      print(_get_comgr_data(data_set_bc, comgr.AMD_COMGR_DATA_KIND_LOG).decode())
      raise RuntimeError("compile failed")
    # Same codegen options regardless of keep_full — the kernel entry point keeps external linkage,
    # only HIP helpers get internalized, and the relocatable still has a populated .symtab either way.
    # This guarantees the linked executable (returned as `lib`) is byte-identical to the default path.
    check(set_options(action_info, b"-O3 -mllvm -amdgpu-internalize-symbols"))
    check(comgr.amd_comgr_do_action(comgr.AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE, action_info, data_set_bc, data_set_reloc))

  # Snapshot the relocatable BEFORE linking. The linked executable comgr produces is
  # heavily stripped (no .symtab, no .note.AMDGPU.metadata) on stock ROCm 6 setups —
  # whereas the relocatable still carries symbols + AMDGPU metadata that RGA needs.
  full_hsaco = _get_comgr_data(data_set_reloc, comgr.AMD_COMGR_DATA_KIND_RELOCATABLE) if keep_full else None

  check(set_options(action_info, b""))
  check(comgr.amd_comgr_do_action(comgr.AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, action_info, data_set_reloc, data_set_exec))
  ret = _get_comgr_data(data_set_exec, comgr.AMD_COMGR_DATA_KIND_EXECUTABLE)
  check(comgr.amd_comgr_release_data(data_src))
  for x in [data_set_src, data_set_bc, data_set_reloc, data_set_exec]: check(comgr.amd_comgr_destroy_data_set(x))
  check(comgr.amd_comgr_destroy_action_info(action_info))
  if keep_full: return ret, full_hsaco  # type: ignore[return-value]
  return ret

class HIPCompiler(Compiler):
  def __init__(self, arch:str):
    self.arch = arch
    # KEEP_FULL_HSACO uses a suffixed cachekey so the cache stores (lib, full_hsaco) pairs and the
    # default (KEEP=0) cache stays untouched / byte-identical. compile_cached is overridden below to
    # populate FULL_HSACO_STASH from the cache hit too — otherwise a hit bypasses compile() entirely.
    super().__init__(f"compile_hip_{self.arch}{'_full' if KEEP_FULL_HSACO.value else ''}")
  def compile(self, src:str) -> bytes:
    try:
      asm = src.split('\n', 1)[0].strip() == '.text'
      if KEEP_FULL_HSACO.value and not asm:
        lib, full = compile_hip(src, self.arch, asm, keep_full=True)  # type: ignore[misc]
        stash_full_hsaco(lib, full)
        return lib
      return compile_hip(src, self.arch, asm)  # type: ignore[return-value]
    except RuntimeError as e: raise CompileError(e) from e
  def compile_cached(self, src:str) -> bytes:
    if not KEEP_FULL_HSACO.value: return super().compile_cached(src)
    return _amd_compile_cached_with_full(self, src, lambda: self.compile(src))
  def disassemble(self, lib:bytes): amdgpu_disassemble(lib)

class HIPCCCompiler(Compiler):
  def __init__(self, arch:str, extra_options:list[str]=[]):
    self.arch, self.extra_options = arch, extra_options
    super().__init__(f"compile_hipcc_{self.arch}_{hashlib.sha256(' '.join(extra_options).encode()).hexdigest()[:8]}")
  def compile(self, src:str) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".cpp") as srcf, tempfile.NamedTemporaryFile(suffix=".bc") as bcf:
      with tempfile.NamedTemporaryFile(suffix=".hsaco") as libf:
        srcf.write(src.encode())
        srcf.flush()

        rocm_path = getenv("ROCM_PATH", "/opt/rocm")
        subprocess.run(["hipcc", "-c", "-emit-llvm", "--cuda-device-only", "-O3", "-mcumode",
                        f"--offload-arch={self.arch}", f"-I{rocm_path}/include/hip", "-o", bcf.name, srcf.name] + self.extra_options, check=True)
        subprocess.run(["hipcc", "-target", "amdgcn-amd-amdhsa", f"-mcpu={self.arch}",
                        "-O3", "-mllvm", "-amdgpu-internalize-symbols", "-c", "-o", libf.name, bcf.name] + self.extra_options, check=True)

        return pathlib.Path(libf.name).read_bytes()
  def disassemble(self, lib:bytes): amdgpu_disassemble(lib)

class AMDLLVMCompiler(LLVMCompiler):
  jit = False
  target_arch = "AMDGPU"
  def __init__(self, arch: str):
    self.arch = arch
    # Suffix the cachekey when KEEP_FULL_HSACO=1 so the default cache stays byte-identical.
    super().__init__(self.arch, "+cumode",
                     cache_key=f"compile_llvm_{arch}_+cumode_full" if KEEP_FULL_HSACO.value else None)
  def __reduce__(self): return (AMDLLVMCompiler, (self.arch,))
  def compile(self, src:str) -> bytes:
    try:
      lib = super().compile(src)
      # The LLVM-emitted AMDGPU object goes straight to runtime as `lib`. When
      # KEEP_FULL_HSACO=1, also stash the same bytes — for some LLVM/ROCm setups
      # this object already carries .symtab + .note.AMDGPU.metadata, so it's a
      # valid input to RGA. If not, callers should switch to the HIPRenderer
      # (DEV=AMD with HIPCompiler enabled) for a fully-symbolic HSACO.
      if KEEP_FULL_HSACO.value: stash_full_hsaco(lib, lib)
      return lib
    except RuntimeError as e:
      if "undefined value '@llvm.amdgcn." in str(e): raise CompileError(str(e) + "AMD with LLVM backend requires LLVM >= 18") from e
      raise CompileError(e) from e
  def compile_cached(self, src:str) -> bytes:
    if not KEEP_FULL_HSACO.value: return super().compile_cached(src)
    return _amd_compile_cached_with_full(self, src, lambda: self.compile(src))
  def disassemble(self, lib:bytes): amdgpu_disassemble(lib)


def _amd_compile_cached_with_full(self_:Compiler, src:str, do_compile) -> bytes:
  """Like Compiler.compile_cached, but the cache value is `(lib, full_hsaco)` so cache hits also
  re-populate FULL_HSACO_STASH for AMDProgram.__init__. Used only when KEEP_FULL_HSACO=1, with a
  separately suffixed cachekey to avoid corrupting the default-mode cache."""
  from tinygrad.helpers import diskcache_get, diskcache_put, getenv
  if self_.cachekey is None or (cached := diskcache_get(self_.cachekey, src)) is None:
    assert not getenv("ASSERT_COMPILE"), f"tried to compile with ASSERT_COMPILE set\n{src}"
    lib = do_compile()
    full = get_full_hsaco(lib)  # populated by compile() when KEEP_FULL_HSACO=1
    if self_.cachekey is not None: diskcache_put(self_.cachekey, src, (lib, full))
    return lib
  if isinstance(cached, tuple) and len(cached) == 2:
    lib, full = cached
    if full is not None: stash_full_hsaco(lib, full)
    return lib
  # Defensive: legacy/raw bytes in this cache slot — fall back to recompile so the stash gets filled.
  lib = do_compile()
  full = get_full_hsaco(lib)
  if self_.cachekey is not None: diskcache_put(self_.cachekey, src, (lib, full))
  return lib
