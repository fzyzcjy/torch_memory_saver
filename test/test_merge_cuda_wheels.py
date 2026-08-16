import importlib.util
import sys
import zipfile
from pathlib import Path
from types import ModuleType


def _load_merge_cuda_wheels_module() -> ModuleType:
    module_path = Path(__file__).parents[1] / "scripts" / "merge_cuda_wheels.py"
    spec = importlib.util.spec_from_file_location("merge_cuda_wheels", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


merge_cuda_wheels = _load_merge_cuda_wheels_module()


def test_merge_rewrites_internal_wheel_tag_to_match_output_filename(
    tmp_path: Path,
) -> None:
    """Merged wheel metadata uses the compatibility tag from its final filename."""

    cu12_wheel = tmp_path / (
        "torch_memory_saver-0.0.10b1+cu128-cp39-abi3-manylinux2014_x86_64.whl"
    )
    cu13_wheel = tmp_path / (
        "torch_memory_saver-0.0.10b1+cu130-cp39-abi3-manylinux2014_x86_64.whl"
    )
    _write_wheel(wheel_path=cu12_wheel, cuda_major="12")
    _write_wheel(wheel_path=cu13_wheel, cuda_major="13")
    output_path = tmp_path / (
        "torch_memory_saver-0.0.10b1-cp39-abi3-manylinux2014_x86_64.whl"
    )

    merge_cuda_wheels.merge(
        input_wheels=[cu12_wheel, cu13_wheel],
        out_path=output_path,
    )

    with zipfile.ZipFile(output_path) as wheel:
        wheel_metadata = wheel.read(
            "torch_memory_saver-0.0.10b1.dist-info/WHEEL"
        ).decode()
    assert "Tag: cp39-abi3-manylinux2014_x86_64\n" in wheel_metadata
    assert "Tag: cp39-abi3-linux_x86_64" not in wheel_metadata


def test_merge_duplicates_cuda12_binaries_under_compatibility_names(
    tmp_path: Path,
) -> None:
    """Unsuffixed legacy binaries are byte-identical CUDA 12 aliases."""

    cu12_wheel = tmp_path / (
        "torch_memory_saver-0.0.10b1+cu128-cp39-abi3-manylinux2014_x86_64.whl"
    )
    cu13_wheel = tmp_path / (
        "torch_memory_saver-0.0.10b1+cu130-cp39-abi3-manylinux2014_x86_64.whl"
    )
    _write_wheel(wheel_path=cu12_wheel, cuda_major="12")
    _write_wheel(wheel_path=cu13_wheel, cuda_major="13")
    output_path = tmp_path / (
        "torch_memory_saver-0.0.10b1-cp39-abi3-manylinux2014_x86_64.whl"
    )

    merge_cuda_wheels.merge(
        input_wheels=[cu12_wheel, cu13_wheel],
        out_path=output_path,
    )

    with zipfile.ZipFile(output_path) as wheel:
        for hook_mode in ("preload", "torch"):
            compatibility = wheel.read(
                f"torch_memory_saver_hook_mode_{hook_mode}.abi3.so"
            )
            cuda12 = wheel.read(
                f"torch_memory_saver_hook_mode_{hook_mode}_cu12.abi3.so"
            )
            assert compatibility == cuda12 == b"12"


def _write_wheel(*, wheel_path: Path, cuda_major: str) -> None:
    metadata_root = "torch_memory_saver-0.0.10b1.dist-info"
    with zipfile.ZipFile(wheel_path, mode="w") as wheel:
        wheel.writestr(
            f"{metadata_root}/WHEEL",
            "Wheel-Version: 1.0\nTag: cp39-abi3-linux_x86_64\n",
        )
        for hook_mode in ("preload", "torch"):
            wheel.writestr(
                f"torch_memory_saver_hook_mode_{hook_mode}_cu{cuda_major}.abi3.so",
                cuda_major.encode(),
            )
        wheel.writestr(f"{metadata_root}/RECORD", "")
