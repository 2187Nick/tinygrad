#!/usr/bin/env bash
# Build vkrun harness + compile all GLSL compute shaders to SPIR-V.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out
echo "[1/2] Compiling shaders…"
for g in shaders/*.comp; do
  n=$(basename "$g" .comp)
  glslangValidator -V "$g" -o "out/${n}.spv" -S comp >/dev/null
  echo "  built out/${n}.spv"
done
echo "[2/2] Compiling vkrun…"
gcc -O2 -Wall -Wextra vkrun.c -lvulkan -o out/vkrun
echo "Done. Artifacts in $(pwd)/out/"
