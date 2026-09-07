from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests/fixtures/bootstrap_controller"
DOCKER_SCENARIOS = (
    "alpine",
    "busybox-unsupported-package-manager",
    "python313-slim-bookworm",
    "ubuntu24-no-python",
    "ubuntu24-nonroot-python-pip-no-venv",
)


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
