#!/usr/bin/env bash
set -euxo pipefail

# NOTE MODIFIED FROM https://github.com/sgl-project/sglang/blob/main/sgl-kernel/build.sh

WHEEL_DIR="dist"

wheel_files=($WHEEL_DIR/*.whl)
for wheel in "${wheel_files[@]}"; do
    # Skip wheels already renamed by a prior invocation (needed when multiple
    # per-CUDA builds accumulate into dist/ before merging).
    if [[ "$wheel" == *manylinux2014* ]]; then
        echo "Skipping already-renamed wheel: $wheel"
        continue
    fi

    intermediate_wheel="${wheel/linux/manylinux2014}"

    case "${CUDA_VERSION:-}" in
        13.0) new_wheel="${intermediate_wheel/-cp39/+cu130-cp39}" ;;
        12.8) new_wheel="${intermediate_wheel/-cp39/+cu128-cp39}" ;;
        *)    new_wheel="$intermediate_wheel" ;;
    esac

    if [[ "$wheel" != "$new_wheel" ]]; then
        echo "Renaming $wheel to $new_wheel"
        mv -- "$wheel" "$new_wheel"
    fi
done
echo "Wheel renaming completed."
