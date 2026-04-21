#!/usr/bin/env bash
set -euxo pipefail

# NOTE MODIFIED FROM https://github.com/sgl-project/sglang/blob/main/sgl-kernel/build.sh

# tms's C++ sources don't use torch or nvcc, so only setuptools/wheel are
# needed at build time. No torch pre-install, no TORCH_CUDA_ARCH_LIST.
${PYTHON_ROOT_PATH}/bin/pip install --no-cache-dir setuptools==75.0.0 wheel==0.41.0
export CUDA_VERSION=${CUDA_VERSION}
mkdir -p /usr/lib/${ARCH}-linux-gnu/
ln -s /usr/local/cuda-${CUDA_VERSION}/targets/${LIBCUDA_ARCH}-linux/lib/stubs/libcuda.so /usr/lib/${ARCH}-linux-gnu/libcuda.so

cd /app
export TMS_CUDA_MAJOR="${CUDA_VERSION%%.*}"
PYTHONPATH=${PYTHON_ROOT_PATH}/lib/python${PYTHON_VERSION}/site-packages ${PYTHON_ROOT_PATH}/bin/python setup.py bdist_wheel --py-limited-api cp39
bash /app/scripts/rename_wheels.sh
