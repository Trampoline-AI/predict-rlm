from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_SCRIPT = REPO_ROOT / "src/predict_rlm/remote/bootstrap_controller.sh"
FIXTURE_ROOT = REPO_ROOT / "tests/fixtures/bootstrap_controller"
DOCKER_SCENARIOS = (
    "ubuntu24-python-pip-no-venv",
    "ubuntu24-no-python",
    "python313-slim-bookworm",
    "alpine",
)


def test_bootstrap_controller_script_is_packaged_asset() -> None:
    assert BOOTSTRAP_SCRIPT.is_file()
    assert os.access(BOOTSTRAP_SCRIPT, os.X_OK)
    assert BOOTSTRAP_SCRIPT.read_text(encoding="utf-8").startswith("#!/bin/sh\n")


def test_bootstrap_controller_script_has_expected_bootstrap_semantics() -> None:
    text = BOOTSTRAP_SCRIPT.read_text(encoding="utf-8")

    assert 'python3 -m venv "$PROBE_DIR"' in text
    assert "python3 -m pip --version" in text
    assert "python3 -m venv --help" not in text
    assert "python3 python3-pip python3-venv" in text
    assert 'python$version-venv' in text
    assert "apk add --no-cache python3 py3-pip py3-virtualenv" in text
    assert 'python3 -m venv "$UV_BOOTSTRAP"' in text
    assert '$UV_COMMAND venv --seed --python "$REQUESTED_PYTHON" "$CONTROLLER_VENV"' in text
    assert '-e "$REPO$EXTRA"' in text


def test_bootstrap_controller_script_is_valid_sh() -> None:
    subprocess.run(["sh", "-n", str(BOOTSTRAP_SCRIPT)], check=True)


@pytest.mark.parametrize("scenario", DOCKER_SCENARIOS)
def test_bootstrap_controller_docker_scenario(scenario: str) -> None:
    if os.environ.get("PREDICT_RLM_RUN_BOOTSTRAP_DOCKER_TESTS") != "1":
        pytest.skip(
            "set PREDICT_RLM_RUN_BOOTSTRAP_DOCKER_TESTS=1 to run bootstrap Docker scenarios"
        )
    if shutil.which("docker") is None:
        pytest.skip("docker executable is not available")
    docker_info = subprocess.run(
        ["docker", "info"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if docker_info.returncode != 0:
        pytest.skip(f"docker daemon is not available: {docker_info.stderr.strip()}")

    dockerfile = FIXTURE_ROOT / scenario / "Dockerfile"
    assert dockerfile.is_file()

    subprocess.run(
        [
            "docker",
            "build",
            "--pull=false",
            "--progress=plain",
            "-f",
            str(dockerfile),
            "-t",
            f"predict-rlm-bootstrap-controller:{scenario}",
            str(REPO_ROOT),
        ],
        check=True,
    )
