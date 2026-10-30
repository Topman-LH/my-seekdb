#!/usr/bin/env bash
# Builds the hipVS HNSW bulk-build worker against the ROCm / hipVS toolchain.
#
# Prerequisites (AMD Radeon Cloud notebook images already ship these):
#   - ROCm >= 6.2 (hipcc, HIP runtime)
#   - hipVS: the ROCm distribution of the cuVS C API
#     (headers: cuvs/core/c_api.h, cuvs/neighbors/cagra.h; lib: libcuvs.so / -lcuvs)
#   - DLPack headers (shipped with cuVS)
#
# Usage:  ./build_worker.sh [output binary]
set -euo pipefail

OUT="${1:-./hipvs_hnsw_worker}"
HERE="$(cd "$(dirname "$0")" && pwd)"

CANDIDATE_INCLUDES=(
  "/opt/rocm/include"
  "/usr/local/include"
)

EXTRA_INC=()
for d in "${CANDIDATE_INCLUDES[@]}"; do
  [ -d "$d" ] && EXTRA_INC+=("-I$d")
done

# Locate the cuVS headers; hipVS installs them under /opt/roken/ or /opt/hipvs
# on some Radeon Cloud images.
for d in /opt/hipvs/include /opt/roken/include /usr/include; do
  if [ -f "$d/cuvs/core/c_api.h" ]; then
    EXTRA_INC+=("-I$d")
    break
  fi
done

echo "[build_worker] compiling with hipcc ..."
hipcc -O3 -std=c++17 \
  "${EXTRA_INC[@]}" \
  -I"$HERE" \
  "$HERE/hipvs_hnsw_worker.cpp" \
  -o "$OUT" \
  -lhip_hcc -lcuvs \
  -pthread

echo "[build_worker] done: $OUT"
"$OUT" 2>&1 | head -1 || true   # prints the usage line and exits 2, which is expected
exit 0
