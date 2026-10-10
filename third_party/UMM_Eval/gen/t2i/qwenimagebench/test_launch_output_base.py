"""Contract tests for QIB launch output-directory precedence."""

import os
import subprocess
from pathlib import Path

import pytest


LAUNCH_PATH = Path(__file__).with_name("launch.sh")


def _output_base_assignment() -> str:
    return next(
        line for line in LAUNCH_PATH.read_text(encoding="utf-8").splitlines()
        if line.startswith('OUTPUT_BASE="')
    )


def test_launch_honors_output_base_after_legacy_overrides():
    assert (
        _output_base_assignment()
        == 'OUTPUT_BASE="${OUTPUT_FOLDER:-${OUTPUT_PATH:-${OUTPUT_BASE:-${REPO_ROOT}/outputs}}}"'
    )


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, "/repo/outputs"),
        ({"OUTPUT_BASE": "/base"}, "/base"),
        ({"OUTPUT_BASE": "/base", "OUTPUT_PATH": "/path"}, "/path"),
        (
            {"OUTPUT_BASE": "/base", "OUTPUT_PATH": "/path", "OUTPUT_FOLDER": "/folder"},
            "/folder",
        ),
    ],
)
def test_launch_output_directory_precedence(values, expected):
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"OUTPUT_BASE", "OUTPUT_PATH", "OUTPUT_FOLDER"}
    }
    env.update(values)
    result = subprocess.run(
        [
            "bash",
            "-c",
            f'REPO_ROOT=/repo\n{_output_base_assignment()}\nprintf "%s" "$OUTPUT_BASE"',
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout == expected
