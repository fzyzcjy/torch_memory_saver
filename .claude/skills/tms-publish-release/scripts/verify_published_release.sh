#!/usr/bin/env bash
set -euxo pipefail

readonly RELEASE_VERSION="${1:?Usage: verify_published_release.sh VERSION ARTIFACT_MANIFEST}"
readonly ARTIFACT_MANIFEST="${2:?Usage: verify_published_release.sh VERSION ARTIFACT_MANIFEST}"
readonly REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
readonly PROXY_URL="${TMS_PROXY_URL:-http://127.0.0.1:7890}"
readonly PYPI_JSON="$(mktemp)"
readonly EXPECTED_PYTEST_SKIPS='{
  "test/test_examples.py::test_cleanup_failure_injection_xpu": "XPU-specific path",
  "test/test_examples.py::test_cpu_backup_multi_device_mmap_restore[preload]": "Multi-device test requires at least two devices",
  "test/test_examples.py::test_cpu_backup_multi_device_mmap_restore[torch]": "Multi-device test requires at least two devices",
  "test/test_examples.py::test_disable_unsupported_xpu": "XPU-specific path",
  "test/test_examples.py::test_free_failure_injection_xpu": "XPU-specific path",
  "test/test_examples.py::test_memory_margin_unsupported_xpu": "XPU-specific path",
  "test/test_examples.py::test_multi_device[preload]": "Multi-device test requires at least two devices",
  "test/test_examples.py::test_multi_device[torch]": "Multi-device test requires at least two devices",
  "test/test_examples.py::test_multi_device_sync_xpu": "XPU-specific path",
  "test/test_examples.py::test_multi_device_torch_mode": "Multi-device test requires at least two devices",
  "test/test_examples.py::test_resume_failure_injection_xpu": "XPU-specific path"
}'

trap 'status=$?; rm -f "$PYPI_JSON"; echo "RESULT: returncode=$status"; exit "$status"' EXIT

test "$(uname -m)" = "x86_64"
for executable in curl jq docker; do
  command -v "$executable"
done
test -f "$ARTIFACT_MANIFEST"
test "$(wc -l < "$ARTIFACT_MANIFEST")" -eq 3

curl --fail --silent --show-error --location --proxy "$PROXY_URL" \
  "https://pypi.org/pypi/torch-memory-saver/${RELEASE_VERSION}/json" \
  --output "$PYPI_JSON"
test "$(jq '.urls | length' "$PYPI_JSON")" -eq 3
while read -r expected_digest filename; do
  test "$(jq --raw-output --arg filename "$filename" '.urls[] | select(.filename == $filename) | .digests.sha256' "$PYPI_JSON")" = "$expected_digest"
done < "$ARTIFACT_MANIFEST"

for runtime in \
  '12|docker.io/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime' \
  '13|docker.io/pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime'; do
  IFS='|' read -r cuda_major image <<< "$runtime"
  docker run --rm --pull missing \
    --gpus device=0 \
    --network host \
    -e "http_proxy=$PROXY_URL" \
    -e "https_proxy=$PROXY_URL" \
    -e "TMS_RELEASE_VERSION=$RELEASE_VERSION" \
    -e "TMS_CUDA_MAJOR=$cuda_major" \
    -e "TMS_EXPECTED_PYTEST_SKIPS=$EXPECTED_PYTEST_SKIPS" \
    -v "$REPOSITORY_ROOT/test:/release-tests:ro" \
    -v "$REPOSITORY_ROOT/.claude/skills/tms-publish-release/scripts/pytest_skip_gate.py:/validation/pytest_skip_gate.py:ro" \
    "$image" \
    bash -lc '
set -euxo pipefail
python -c '\''import importlib.util; assert importlib.util.find_spec("torch_memory_saver") is None'\''
python -m pip install --no-cache-dir pytest==8.3.5 nvidia-ml-py==12.570.86
python -m pip install --no-cache-dir --pre --no-deps "torch-memory-saver==${TMS_RELEASE_VERSION}"
python - <<'\''PY'\''
from pathlib import Path
import os
import torch
import torch_memory_saver

cuda = torch.version.cuda
name = torch.cuda.get_device_name(0)
package_path = Path(torch_memory_saver.__file__)
print(torch.__version__, cuda, torch.cuda.is_available(), name, package_path)
assert cuda is not None and cuda.split(".", 1)[0] == os.environ["TMS_CUDA_MAJOR"]
assert torch.cuda.is_available()
assert "4090 D" in name or "4090D" in name
assert "site-packages" in package_path.parts
PY
CUDA_RUNTIME_LIB="$(python -c '\''from pathlib import Path; import os, site; major=os.environ["TMS_CUDA_MAJOR"]; matches=[path for root in site.getsitepackages() for path in (Path(root) / "nvidia").glob(f"**/libcudart.so.{major}")]; assert matches, matches; print(":".join(sorted({str(path.parent) for path in matches})))'\'')"
export LD_LIBRARY_PATH="${CUDA_RUNTIME_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
mkdir -p /validation/test
cp -a /release-tests/examples /validation/test/examples
cp /release-tests/test_configure_subprocess.py /release-tests/test_examples.py /release-tests/test_utils.py /validation/test/
cd /validation
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=30s 3600s python -m pytest -p pytest_skip_gate test -vv -ra
'
done
