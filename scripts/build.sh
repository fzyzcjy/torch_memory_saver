#!/bin/bash
set -euxo pipefail
# NOTE MODIFIED FROM https://github.com/sgl-project/sglang/blob/main/sgl-kernel/build.sh
echo "python=${PYTHON_VERSION} cuda=${CUDA_VERSION}" # args

PYTHON_ROOT_PATH=/opt/python/cp${PYTHON_VERSION//.}-cp${PYTHON_VERSION//.}
# Respect caller-provided ARCH (e.g. cross-image builds from a different host arch,
# or macOS where `uname -i` is not supported). Fall back to `uname -m` (portable).
ARCH="${ARCH:-$(uname -m)}"

echo "ARCH:  $ARCH"
if [ ${ARCH} = "aarch64" ]; then
   LIBCUDA_ARCH="sbsa"
   BUILDER_NAME="pytorch/manylinuxaarch64-builder"
else
   LIBCUDA_ARCH=${ARCH}
   if [ ${CUDA_VERSION} = "12.8" ] || [ ${CUDA_VERSION} = "13.0" ]; then
      BUILDER_NAME="pytorch/manylinux2_28-builder"
   else
      BUILDER_NAME="pytorch/manylinux-builder"
   fi
fi

DOCKER_IMAGE="${BUILDER_NAME}:cuda${CUDA_VERSION}"
docker run --rm \
   -v $(pwd):/app \
   ${DOCKER_IMAGE} \
   bash -c "
   PYTHON_ROOT_PATH=${PYTHON_ROOT_PATH} \
   PYTHON_VERSION=${PYTHON_VERSION} \
   CUDA_VERSION=${CUDA_VERSION} \
   ARCH=${ARCH} \
   LIBCUDA_ARCH=${LIBCUDA_ARCH} \
   bash /app/scripts/build_in_docker.sh
   "