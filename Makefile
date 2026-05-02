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
	rm -rf dist/*

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

.PHONY:build-sdist
build-sdist:
	# python3 -m build --no-isolation
	python3 setup.py sdist --dist-dir=dist

.PHONY:upload
upload:
	ls -alh dist
	docker run -it --rm -v $(shell pwd):/app python:3.11 \
	  /bin/bash -c "pip install twine && python3 -m twine upload /app/dist/*"
