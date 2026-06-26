#!/usr/bin/env bash
# Build torch_memory_saver for Intel XPU (Level Zero backend).
#
# Unlike the CUDA path (which ships prebuilt manylinux wheels for fixed CUDA
# majors), the XPU build is done from source against the LOCAL oneAPI + torch
# XPU runtime. The produced .so links libsycl.so.<N>, whose major MUST match the
# intel-sycl-rt bundled with your torch+xpu wheel (e.g. torch 2.12.0+xpu ->
# intel-sycl-rt 2025.3.x -> libsycl.so.8; a newer wheel may need libsycl.so.9).
# Building against a mismatched oneAPI yields a .so that links one libsycl ABI
# while running against another, corrupting the SYCL runtime (garbage device
# counts / runaway prewarm / segfault on load, or an "undefined symbol:
# urDeviceWaitExp ... LIBUR_LOADER" dlopen failure).
#
# setup.py automatically selects an installed oneAPI compiler whose libsycl
# major matches your torch, so you normally do NOT need to source a specific
# oneAPI. It only fails if no matching compiler exists on the system.
#
# Requirements: Intel oneAPI (icpx) + Level Zero headers (ze_api.h, zes_api.h).
#
# Usage:
#   # let setup.py auto-pick a compiler matching your torch's libsycl:
#   bash scripts/build_xpu.sh
#   # or pin a specific compiler (used as-is, no auto-switch):
#   ICPX=/opt/intel/oneapi/compiler/2025.3/bin/icpx bash scripts/build_xpu.sh
#
# After building, verify the SONAME matches your torch runtime:
#   objdump -p torch_memory_saver_hook_mode_torch*.so | grep libsycl
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

PYTHON="${PYTHON:-python3}"

# Make some icpx discoverable for setup.py's auto-selection. We do NOT pin ICPX
# here (that would force one specific compiler and defeat ABI auto-matching);
# we only source oneAPI if no icpx is visible at all. If the user set ICPX
# explicitly it is respected (used as-is) by setup.py.
if [ -z "${ICPX:-}" ] && ! command -v icpx >/dev/null 2>&1; then
  if [ -f /opt/intel/oneapi/setvars.sh ]; then
    set +u
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
    set -u
  fi
fi
if [ -z "${ICPX:-}" ] && ! command -v icpx >/dev/null 2>&1 \
   && [ ! -d /opt/intel/oneapi/compiler ]; then
  echo "ERROR: no Intel oneAPI compiler found. Install Intel oneAPI, source" \
       "setvars.sh, or set ICPX=/opt/intel/oneapi/compiler/<ver>/bin/icpx" >&2
  exit 1
fi

# TMS_PLATFORM=xpu forces the XPU branch in setup.py even if nvcc/hipcc exist.
TMS_PLATFORM=xpu "$PYTHON" setup.py build_ext --inplace

echo ""
echo "Built. SONAME check (must match your torch+xpu libsycl):"
for so in torch_memory_saver_hook_mode_torch*.so \
          torch_memory_saver/torch_memory_saver_hook_mode_torch*.so; do
  [ -e "$so" ] || continue
  echo -n "  $so -> "
  objdump -p "$so" 2>/dev/null | awk '/NEEDED.*libsycl/{print $2}'
done