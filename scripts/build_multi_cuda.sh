#!/usr/bin/env bash
# Build a single wheel that ships both cu12 and cu13 .so variants.
#
# Flow: run scripts/build.sh once per CUDA version (each produces a
# per-CUDA `+cuXXX`-tagged wheel in dist/), then merge_cuda_wheels.py
# combines them into a single CUDA-agnostic wheel. Pass PYTHON_VERSION
# and ARCH through the environment; CUDA_VERSION is controlled here.
set -euxo pipefail

: "${PYTHON_VERSION:=3.10}"

# Build each CUDA variant. Each invocation tags its wheel with a different
# +cuXXX local-version (see scripts/rename_wheels.sh), so the two intermediates
# coexist in dist/ instead of clobbering each other.
CUDA_VERSION=12.8 PYTHON_VERSION="${PYTHON_VERSION}" bash scripts/build.sh
CUDA_VERSION=13.0 PYTHON_VERSION="${PYTHON_VERSION}" bash scripts/build.sh

# Merge the two per-CUDA wheels into one. File globbing relies on the
# local-version tags applied by rename_wheels.sh.
shopt -s nullglob
cu12_matches=( dist/*+cu128-*.whl )
cu13_matches=( dist/*+cu130-*.whl )
shopt -u nullglob

if (( ${#cu12_matches[@]} != 1 )); then
    echo "ERROR: expected exactly 1 cu12 wheel in dist/, found ${#cu12_matches[@]}: ${cu12_matches[*]:-<none>}" >&2
    echo "Hint: run 'make clean' first if dist/ has stale wheels from a previous build." >&2
    exit 1
fi
if (( ${#cu13_matches[@]} != 1 )); then
    echo "ERROR: expected exactly 1 cu13 wheel in dist/, found ${#cu13_matches[@]}: ${cu13_matches[*]:-<none>}" >&2
    exit 1
fi
CU12_WHEEL="${cu12_matches[0]}"
CU13_WHEEL="${cu13_matches[0]}"

python3 scripts/merge_cuda_wheels.py "${CU12_WHEEL}" "${CU13_WHEEL}" --out-dir dist

# Remove the per-CUDA intermediates so dist/ contains only the merged wheel.
rm -f "${CU12_WHEEL}" "${CU13_WHEEL}"

ls -la dist/
