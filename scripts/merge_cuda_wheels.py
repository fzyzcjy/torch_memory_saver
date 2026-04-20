"""Merge per-CUDA-version wheels into a single multi-CUDA wheel.

Usage:
    python merge_cuda_wheels.py <wheel1> <wheel2> [<wheelN> ...] [--out-dir DIR]

Each input wheel must have the same package version and compatibility tag
(cp39-abi3-manylinux2014_<arch>); they differ only in the CUDA-specific .so
files they carry (torch_memory_saver_hook_mode_*_cu{12,13}.abi3.so). The
output wheel contains the union of all .so files, plus the shared Python
package, with a rebuilt RECORD. The output filename drops any `+cuXXX`
local-version tag.
"""

from __future__ import annotations

import argparse
import base64
import difflib
import hashlib
import re
import zipfile
from pathlib import Path


def _urlsafe_b64(digest: bytes) -> str:
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _hash_and_size(data: bytes) -> tuple[str, int]:
    return f"sha256={_urlsafe_b64(hashlib.sha256(data).digest())}", len(data)


def _strip_local_version(filename: str) -> str:
    # torch_memory_saver-0.0.10+cu130-cp39-abi3-...whl -> torch_memory_saver-0.0.10-cp39-abi3-...whl
    return re.sub(r"\+cu\d+", "", filename)


def merge(input_wheels: list[Path], out_path: Path) -> None:
    if len(input_wheels) < 2:
        raise SystemExit("Need at least two input wheels to merge.")

    merged: dict[str, bytes] = {}
    record_name: str | None = None

    for wheel in input_wheels:
        with zipfile.ZipFile(wheel) as zf:
            for name in zf.namelist():
                if name.endswith("/RECORD"):
                    record_name = name
                    continue  # regenerated at the end
                data = zf.read(name)
                if name in merged and merged[name] != data:
                    if name.endswith("/top_level.txt"):
                        # Each per-CUDA wheel lists only its own top-level
                        # extension modules; take the union so the merged wheel
                        # covers all .so files it ships.
                        merged_lines = {
                            line.strip()
                            for buf in (merged[name], data)
                            for line in buf.decode("utf-8").splitlines()
                            if line.strip()
                        }
                        merged[name] = (
                            "\n".join(sorted(merged_lines)) + "\n"
                        ).encode("utf-8")
                        continue
                    try:
                        old = merged[name].decode("utf-8").splitlines(keepends=True)
                        new = data.decode("utf-8").splitlines(keepends=True)
                        diff = "".join(difflib.unified_diff(old, new, fromfile="earlier", tofile=str(wheel)))
                    except UnicodeDecodeError:
                        diff = "(binary file; cannot show diff)"
                    raise SystemExit(
                        f"File {name!r} differs between {wheel} and an earlier wheel.\n"
                        f"Inputs must share identical Python package contents.\n"
                        f"Diff:\n{diff}"
                    )
                merged[name] = data

    if record_name is None:
        raise SystemExit("No RECORD file found in any input wheel.")

    # Rebuild RECORD: one line per file, then the RECORD's own entry with empty hash/size.
    record_lines: list[str] = []
    for name, data in sorted(merged.items()):
        digest, size = _hash_and_size(data)
        record_lines.append(f"{name},{digest},{size}")
    record_lines.append(f"{record_name},,")
    record_bytes = ("\n".join(record_lines) + "\n").encode("utf-8")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in merged.items():
            zf.writestr(name, data)
        zf.writestr(record_name, record_bytes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheels", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("dist"))
    args = parser.parse_args()

    out_name = _strip_local_version(args.wheels[0].name)
    out_path = args.out_dir / out_name
    merge(args.wheels, out_path)
    print(f"Merged {len(args.wheels)} wheels -> {out_path}")


if __name__ == "__main__":
    main()
