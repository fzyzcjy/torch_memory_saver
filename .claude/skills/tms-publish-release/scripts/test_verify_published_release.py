from __future__ import annotations

import re
import subprocess
from pathlib import Path


def test_published_release_script_is_strict_valid_shell() -> None:
    """The public post-release workflow is strict shell with valid syntax."""

    script_path = Path(__file__).with_name("verify_published_release.sh")
    source = script_path.read_text(encoding="utf-8")

    subprocess.run(["bash", "-n", str(script_path)], check=True)
    assert source.startswith("#!/usr/bin/env bash\nset -euxo pipefail\n")
    assert "bash -lc '\nset -euxo pipefail\n" in source


def test_published_release_script_uses_fresh_pypi_installs_and_runtime_tests() -> None:
    """Both CUDA majors reject preinstallation before exercising PyPI runtime code."""

    source = (
        Path(__file__)
        .with_name("verify_published_release.sh")
        .read_text(encoding="utf-8")
    )

    assert source.count("docker run --rm --pull missing") == 1
    assert "cuda12.4-cudnn9-runtime" in source
    assert "cuda13.0-cudnn9-runtime" in source
    assert source.index('find_spec("torch_memory_saver") is None') < source.index(
        '"torch-memory-saver==${TMS_RELEASE_VERSION}"'
    )
    assert "site-packages" in source
    assert "pytest -p pytest_skip_gate test -vv -ra" in source
    assert '"$REPOSITORY_ROOT/test:/release-tests:ro"' in source
    assert "/dist/" not in source


def test_published_release_script_matches_approved_remote_artifacts() -> None:
    """Post-release checks bind PyPI and GitHub state to the approved manifest."""

    source = (
        Path(__file__)
        .with_name("verify_published_release.sh")
        .read_text(encoding="utf-8")
    )

    assert 'test "$(wc -l < "$ARTIFACT_MANIFEST")" -eq 3' in source
    assert 'test "$(jq ' in source
    assert ".digests.sha256" in source
    assert "api.github.com/repos/fzyzcjy/torch_memory_saver/releases/tags" in source
    assert "'.tag_name'" in source
    assert "'.prerelease == true'" in source
    assert "'.prerelease == false'" in source
    assert 'echo "RESULT: returncode=$status"' in source


def test_published_release_script_checks_each_host_dependency() -> None:
    """A missing non-final host command fails during verifier preflight."""

    source = (
        Path(__file__)
        .with_name("verify_published_release.sh")
        .read_text(encoding="utf-8")
    )

    assert "for executable in curl jq docker; do" in source
    assert 'command -v "$executable"' in source
    assert "command -v curl jq docker" not in source


def test_every_repository_shell_script_enables_strict_mode_first() -> None:
    """Every shell script enables strict tracing before its first command."""

    repository_root = Path(__file__).parents[4]
    script_paths = sorted((repository_root / "scripts").glob("*.sh")) + sorted(
        Path(__file__).parent.glob("*.sh")
    )

    assert script_paths
    for script_path in script_paths:
        lines = script_path.read_text(encoding="utf-8").splitlines()
        first_command = next(
            line for line in lines[1:] if line.strip() and not line.startswith("#")
        )
        assert first_command == "set -euxo pipefail", script_path


def test_every_documented_shell_context_enables_strict_mode() -> None:
    """Every runnable doc block and nested remote shell enables strict tracing."""

    skill_source = (
        Path(__file__).parents[1].joinpath("SKILL.md").read_text(encoding="utf-8")
    )
    bash_blocks = [
        section.split("```", 1)[0] for section in skill_source.split("```bash\n")[1:]
    ]

    assert bash_blocks
    assert all(block.startswith("set -euxo pipefail\n") for block in bash_blocks)
    assert len(
        re.findall(r"ssh tom-workstation [\"']set -euxo pipefail", skill_source)
    ) == skill_source.count("ssh tom-workstation")
    assert "zsh -lc 'set -euxo pipefail;" in skill_source
