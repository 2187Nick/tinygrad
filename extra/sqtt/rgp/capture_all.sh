#!/usr/bin/env bash
# Capture one .rgp file per probe shader via MESA_VK_TRACE.
#
# Output: extra/sqtt/rgp/captures/<probe>.rgp
# Requires: out/vkrun + out/*.spv (run ./build.sh first) + Mesa RADV driver.
set -euo pipefail
cd "$(dirname "$0")"
OUT=captures
mkdir -p "$OUT"

# 7900 XTX has 48 WGPs; SQTT traces 1 CU per SE (1 WGP's worth). To catch enough
# workgroups on the traced units we dispatch many — GX=256 gives ~32 WGs/SE.
GX="${GX:-256}"
for spv in out/probe_*.spv; do
  name=$(basename "$spv" .spv)
  echo "=== ${name} (gx=${GX}) ==="
  rm -f /tmp/vkrun_*.rgp
  MESA_VK_TRACE=rgp MESA_VK_TRACE_PER_SUBMIT=1 \
    RADV_THREAD_TRACE_INSTRUCTION_TIMING=true \
    RADV_THREAD_TRACE_QUEUE_EVENTS=true \
    ./out/vkrun "$spv" "$GX" 2>&1 | sed 's/^/  /'
  latest=$(ls -t /tmp/vkrun_*.rgp 2>/dev/null | head -1 || true)
  if [[ -z "$latest" ]]; then
    echo "  WARNING: no .rgp produced for ${name}" >&2
    continue
  fi
  mv "$latest" "${OUT}/${name}.rgp"
  echo "  -> ${OUT}/${name}.rgp ($(du -h "${OUT}/${name}.rgp" | cut -f1))"
done
echo "Done. See ${OUT}/"
