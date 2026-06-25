#!/usr/bin/env bash
# Build torch_memory_saver for Intel XPU (Level Zero backend).
#
# Unlike the CUDA path (which ships prebuilt manylinux wheels for fixed CUDA
# majors), the XPU build is done from source against the LOCAL oneAPI + torch
# XPU runtime. This is because the produced .so links libsycl.so.<N>, whose
# major MUST match the intel-sycl-rt bundled with your torch+xpu wheel
# (e.g. torch 2.11.0+xpu -> intel-sycl-rt 2025.3.x -> libsycl.so.8). Building
# against a mismatched oneAPI makes the .so fail to dlopen with
# "undefined symbol: urDeviceWaitExp ... LIBUR_LOADER".
#
# Requirements: Intel oneAPI (icpx) + Level Zero headers (ze_api.h, zes_api.h).
#
# Usage:
#   # auto-detect icpx from PATH / oneAPI install:
#   bash scripts/build_xpu.sh
#   # or pin a specific compiler that matches your torch runtime:
#   ICPX=/opt/intel/oneapi/compiler/2025.3/bin/icpx bash scripts/build_xpu.sh
#
# After building, verify the SONAME matches your torch runtime:
#   objdump -p torch_memory_saver_hook_mode_torch*.so | grep libsycl
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

PYTHON="${PYTHON:-python3}"

# Resolve icpx: $ICPX, then PATH, then oneAPI setvars.
if [ -z "${ICPX:-}" ]; then
  if command -v icpx >/dev/null 2>&1; then
    ICPX="$(command -v icpx)"
  elif [ -f /opt/intel/oneapi/setvars.sh ]; then
    set +u
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
    set -u
    ICPX="$(command -v icpx || true)"
  fi
fi
if [ -z "${ICPX:-}" ] || ! command -v "$ICPX" >/dev/null 2>&1; then
  echo "ERROR: icpx not found. Install Intel oneAPI, source setvars.sh, or set" \
       "ICPX=/opt/intel/oneapi/compiler/<ver>/bin/icpx" >&2
  exit 1
fi
export ICPX
echo "Using icpx: $ICPX"
"$ICPX" --version | head -1

# TMS_PLATFORM=xpu forces the XPU branch in setup.py even if nvcc/hipcc exist.
TMS_PLATFORM=xpu "$PYTHON" setup.py build_ext --inplace

echo ""
echo "Built. SONAME check (must match your torch+xpu libsycl):"
for so in torch_memory_saver_hook_mode_torch*.so; do
  [ -e "$so" ] || continue
  echo -n "  $so -> "
  objdump -p "$so" 2>/dev/null | awk '/NEEDED.*libsycl/{print $2}'
done
