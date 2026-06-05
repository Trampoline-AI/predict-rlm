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
    "alpine",
    "busybox-unsupported-package-manager",
    "debian13-no-python",
    "python311-slim",
    "python313-slim-bookworm",
    "tbench-python-313",
    "ubuntu24-no-python",
    "ubuntu24-nonroot-python-pip-no-venv",
    "ubuntu24-python-pip-no-venv",
)


def test_bootstrap_controller_script_is_packaged_asset() -> None:
    assert BOOTSTRAP_SCRIPT.is_file()
    assert os.access(BOOTSTRAP_SCRIPT, os.X_OK)
    assert BOOTSTRAP_SCRIPT.read_text(encoding="utf-8").startswith("#!/bin/sh\n")


def test_bootstrap_controller_script_is_valid_sh() -> None:
    subprocess.run(["sh", "-n", str(BOOTSTRAP_SCRIPT)], check=True)


def test_bootstrap_controller_docker_matrix_covers_daytona_cases() -> None:
    missing = [
        scenario
        for scenario in DOCKER_SCENARIOS
        if not (FIXTURE_ROOT / scenario / "Dockerfile").is_file()
    ]
    assert missing == []


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
