from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import pytest


@dataclass(eq=False)
class SkipGate:
    expected: dict[str, str]
    actual: dict[str, str] = field(default_factory=dict)
    duplicates: list[str] = field(default_factory=list)

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        self._record(report=report)

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        self._record(report=report)

    def pytest_sessionfinish(
        self,
        session: pytest.Session,
        exitstatus: pytest.ExitCode,
    ) -> None:
        mismatch = _skip_mismatch(
            expected=self.expected,
            actual=self.actual,
            duplicates=self.duplicates,
        )
        print(f"SKIP_GATE_ACTUAL={json.dumps(self.actual, sort_keys=True)}", flush=True)
        if mismatch is None:
            print("SKIP_GATE_RESULT=pass", flush=True)
            return

        print(f"SKIP_GATE_RESULT=fail {mismatch}", flush=True)
        session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def _record(self, *, report: Any) -> None:
        if not report.skipped:
            return
        reason = _skip_reason(longrepr=report.longrepr)
        if report.nodeid in self.actual:
            self.duplicates.append(report.nodeid)
        self.actual[report.nodeid] = reason


def pytest_configure(config: pytest.Config) -> None:
    raw_expected = os.environ.get("TMS_EXPECTED_PYTEST_SKIPS")
    if raw_expected is None:
        raise pytest.UsageError("TMS_EXPECTED_PYTEST_SKIPS is required")
    try:
        expected = json.loads(raw_expected)
    except json.JSONDecodeError as error:
        raise pytest.UsageError(
            "TMS_EXPECTED_PYTEST_SKIPS must be valid JSON"
        ) from error
    if not isinstance(expected, dict) or not all(
        isinstance(nodeid, str) and isinstance(reason, str)
        for nodeid, reason in expected.items()
    ):
        raise pytest.UsageError("TMS_EXPECTED_PYTEST_SKIPS must be a JSON string map")
    config.pluginmanager.register(SkipGate(expected=expected), "tms-skip-gate")


def _skip_reason(*, longrepr: Any) -> str:
    reason = longrepr[2] if isinstance(longrepr, tuple) else str(longrepr)
    return reason.removeprefix("Skipped: ")


def _skip_mismatch(
    *,
    expected: dict[str, str],
    actual: dict[str, str],
    duplicates: list[str],
) -> str | None:
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    wrong_reasons = {
        nodeid: {"expected": expected[nodeid], "actual": actual[nodeid]}
        for nodeid in sorted(set(expected) & set(actual))
        if expected[nodeid] != actual[nodeid]
    }
    if not missing and not unexpected and not wrong_reasons and not duplicates:
        return None
    return json.dumps(
        {
            "missing": missing,
            "unexpected": unexpected,
            "wrong_reasons": wrong_reasons,
            "duplicates": sorted(duplicates),
        },
        sort_keys=True,
    )
