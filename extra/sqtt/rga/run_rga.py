#!/usr/bin/env python3
"""Drive RGA against a tinygrad capture pickle.

Reads `full_hsaco` (populated when capture ran with KEEP_FULL_HSACO=1) from a
capture .pkl, writes it to a temp file, then invokes RGA in `bin` mode with
`--isa --livereg --livereg-sgpr -a` and prints a one-line summary.

Usage:
  python extra/sqtt/rga/run_rga.py <capture.pkl> [output_dir]

Env vars:
  RGA_BIN    Path to RGA binary (default: ~/tools/rga-2.14.1.3/rga).
"""
import os, sys, pickle, pathlib, subprocess, tempfile, re, shutil


def _rga_path() -> str:
  default = pathlib.Path.home() / "tools" / "rga-2.14.1.3" / "rga"
  return os.environ.get("RGA_BIN", str(default))


def _arch_to_rga_asic(target: str) -> str:
  # RGA accepts the bare AMDGPU arch token for `bin` mode.
  return target.split(":")[0]


def _summarize(out_dir: pathlib.Path, kernel_name: str) -> tuple[int, int]:
  """Best-effort: return (instruction_count, max_vgpr_live) by scanning RGA output files.

  RGA filenames end in `_isa.amdisa`, `_livereg.txt`, `_livereg_sgpr.txt`, `_analysis.csv`.
  Returns (-1, -1) if a file is missing or unparseable — diagnostics, not load-bearing.
  """
  inst_count = max_live = -1
  for p in out_dir.iterdir():
    n = p.name.lower()
    if n.endswith("_isa.amdisa"):
      try:
        # Count `// ADDR: ENC` ISA lines (hex-prefixed comment markers RGA emits per instruction).
        inst_count = sum(1 for ln in p.read_text(errors="ignore").splitlines() if re.search(r"//\s*[0-9A-F]{6,}\s*:", ln))
      except OSError: pass
    if n.endswith("_livereg.txt"):
      try:
        # Extract the "Maximum # VGPR used  N" footer RGA prints.
        m = re.search(r"Maximum # VGPR used\s+(\d+)", p.read_text(errors="ignore"))
        if m: max_live = int(m.group(1))
      except OSError: pass
  return inst_count, max_live


def main() -> int:
  if len(sys.argv) < 2:
    print(__doc__, file=sys.stderr)
    return 2
  pkl_path = pathlib.Path(sys.argv[1]).resolve()
  out_dir = pathlib.Path(sys.argv[2]).resolve() if len(sys.argv) >= 3 else pkl_path.parent / f"{pkl_path.stem}_rga"
  out_dir.mkdir(parents=True, exist_ok=True)

  rga = _rga_path()
  if not pathlib.Path(rga).exists() and not shutil.which(rga):
    print(f"error: RGA not found at {rga} (set RGA_BIN to override)", file=sys.stderr)
    return 3

  with open(pkl_path, "rb") as f: cap = pickle.load(f)
  full = cap.get("full_hsaco")
  if not full:
    print(f"error: {pkl_path} has no `full_hsaco` field. Re-run the capture with KEEP_FULL_HSACO=1.", file=sys.stderr)
    return 4

  target = cap.get("target", "gfx1100")
  kernel = cap.get("kernel", pkl_path.stem)
  asic = _arch_to_rga_asic(target)

  with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=False) as tmp:
    tmp.write(full)
    hsaco_path = tmp.name

  try:
    base = out_dir / f"{asic}_{kernel}"
    # `bin` mode infers the asic from the code object's e_flags; it does NOT accept -c.
    # The HSACO path goes through --co (NOT positional).
    cmd = [rga, "-s", "bin",
           "--isa",          str(base) + "_isa.amdisa",
           "--livereg",      str(base) + "_livereg.txt",
           "--livereg-sgpr", str(base) + "_livereg_sgpr.txt",
           "-a",             str(base) + "_analysis.csv",
           "--co",           hsaco_path]
    print("running:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
      print("stdout:", proc.stdout)
      print("stderr:", proc.stderr, file=sys.stderr)
      return proc.returncode
    insts, max_live = _summarize(out_dir, kernel)
    print(f"output dir: {out_dir}")
    print(f"summary: kernel={kernel} asic={asic} insts={insts} max_vgpr_live={max_live}")
    return 0
  finally:
    try: os.unlink(hsaco_path)
    except OSError: pass


if __name__ == "__main__":
  sys.exit(main())
