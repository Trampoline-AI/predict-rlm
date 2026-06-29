from __future__ import annotations

import os
import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from predict_rlm import PredictRLM
from predict_rlm.rlm_skills import Skill

SLUGIFY_SKILL = Skill(
    name="slugify",
    instructions="Use python-slugify when converting text into URL slugs.",
    packages=["python-slugify"],
)

SLUGIFY_CODE = (
    "from slugify import slugify\n"
    "answer = slugify('Skill Packages Work')\n"
    "SUBMIT(answer=answer)"
)


class SequentialActions:
    def __init__(self, *actions: SimpleNamespace) -> None:
        self.actions = list(actions)
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        assert self.actions, "PredictRLM requested more actions than the test provided"
        return self.actions.pop(0)


def _real_sbx_available() -> bool:
    if os.environ.get("PREDICT_RLM_RUN_SBX_TESTS") != "1":
        return False
    if shutil.which("sbx") is None:
        return False
    return (
        subprocess.run(
            ["sbx", "ls"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        ).returncode
        == 0
    )


def _run_skill_package_rlm(**kwargs) -> None:
    actions = SequentialActions(
        SimpleNamespace(
            reasoning="import and use the package declared by the skill",
            code=SLUGIFY_CODE,
        )
    )
    rlm = PredictRLM(
        "prompt -> answer",
        sub_lm=MagicMock(),
        max_iterations=1,
        skills=[SLUGIFY_SKILL],
        **kwargs,
    )
    rlm.generate_action = actions

    prediction = rlm(prompt="slugify this phrase")

    assert prediction.answer == "skill-packages-work"
    assert len(actions.calls) == 1


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("deno") is None, reason="JSPI skill-package test requires Deno")
def test_predict_rlm_installs_skill_packages_in_jspi_sandbox() -> None:
    _run_skill_package_rlm()


@pytest.mark.sbx
@pytest.mark.integration
@pytest.mark.skipif(
    not _real_sbx_available(),
    reason="real SBX tests require PREDICT_RLM_RUN_SBX_TESTS=1, sbx CLI, and sbx login",
)
def test_predict_rlm_installs_skill_packages_in_sbx_sandbox() -> None:
    from predict_rlm import SbxConfig

    _run_skill_package_rlm(
        sandbox_backend="sbx",
        sbx_config=SbxConfig(name=f"predict-rlm-skill-package-{os.getpid()}"),
    )


@pytest.mark.sbx
@pytest.mark.integration
@pytest.mark.skipif(
    not _real_sbx_available(),
    reason="real SBX tests require PREDICT_RLM_RUN_SBX_TESTS=1, sbx CLI, and sbx login",
)
def test_predict_rlm_installs_skill_packages_in_injected_sbx_sandbox(tmp_path) -> None:
    from predict_rlm import SbxConfig
    from predict_rlm.backends import SbxBackend
    from predict_rlm.workspace import DirectWorkspaceMount

    interpreter = SbxBackend(
        config=SbxConfig(name=f"predict-rlm-skill-package-injected-{os.getpid()}"),
        direct_workspace_mounts=[
            DirectWorkspaceMount(
                host_path=str(tmp_path.resolve()),
                sandbox_path=str(tmp_path.resolve()),
            )
        ],
    )
    try:
        _run_skill_package_rlm(interpreter=interpreter)
    finally:
        interpreter.shutdown()


@pytest.mark.sbx
@pytest.mark.integration
@pytest.mark.skipif(
    not _real_sbx_available(),
    reason="real SBX tests require PREDICT_RLM_RUN_SBX_TESTS=1, sbx CLI, and sbx login",
)
def test_predict_rlm_installs_skill_packages_in_sbx_pool_sandbox() -> None:
    from predict_rlm import SbxConfig, SbxPool

    pool = SbxPool(
        size=1,
        config=SbxConfig(name=f"predict-rlm-skill-package-pool-{os.getpid()}"),
        preinstall_packages=False,
    )
    try:
        _run_skill_package_rlm(sandbox_backend="sbx", sbx_pool=pool)
    finally:
        pool.shutdown()
