# This file is for development usage

SHELL=/bin/bash

.PHONY:reinstall
reinstall:
	rm -rf ./*.so ./build
	pip uninstall torch_memory_saver -y
	pip install --no-cache-dir .

# Release
# sudo make clean
# sudo make build-wheel
# sudo make build-sdist
# sudo make upload

.PHONY:clean
clean:
	rm -rf dist build torch_memory_saver.egg-info ./*.so

.PHONY:build-wheel
build-wheel:
	PYTHON_VERSION=3.9 CUDA_VERSION=12.4 bash scripts/build.sh

.PHONY:build-wheel-cu13
build-wheel-cu13:
	PYTHON_VERSION=3.10 CUDA_VERSION=13.0 bash scripts/build.sh

.PHONY:build-wheel-cu13-aarch64
build-wheel-cu13-aarch64:
	ARCH=aarch64 PYTHON_VERSION=3.10 CUDA_VERSION=13.0 bash scripts/build.sh

.PHONY:build-wheel-multi-cuda
build-wheel-multi-cuda:
	PYTHON_VERSION=3.10 bash scripts/build_multi_cuda.sh

.PHONY:build-wheel-multi-cuda-aarch64
build-wheel-multi-cuda-aarch64:
	ARCH=aarch64 PYTHON_VERSION=3.10 bash scripts/build_multi_cuda.sh

.PHONY:build-xpu
build-xpu:
	TMS_PLATFORM=xpu pip install --no-build-isolation .

.PHONY:build-sdist
build-sdist:
	docker run --rm \
	  -e TMS_HOST_UID=$$(id -u) \
	  -e TMS_HOST_GID=$$(id -g) \
	  -v $(CURDIR):/app \
	  $${TMS_PYTHON_BUILD_IMAGE:-python:3.11} \
	  /bin/bash -c "pip install --no-cache-dir setuptools==75.0.0 \
	  && cd /app \
	  && TMS_CUDA_MAJOR=12 python setup.py sdist --dist-dir=dist \
	  && chown -R \"\$${TMS_HOST_UID}:\$${TMS_HOST_GID}\" /app/dist /app/torch_memory_saver.egg-info"

.PHONY:upload
upload:
	ls -alh dist
	docker run -it --rm -v $(shell pwd):/app python:3.11 \
	  /bin/bash -c "pip install twine && python3 -m twine upload /app/dist/*"
