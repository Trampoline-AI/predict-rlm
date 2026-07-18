"""Regression checks for packaged RLM skill docs."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _installable_skill_docs_text() -> str:
    skill_docs = sorted((ROOT / ".agents" / "skills").glob("**/*.md"))
    return "\n".join(path.read_text() for path in skill_docs)


def test_public_rlm_skill_version_snippets_match_package_version():
    package_version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "version"
    ]
    skill_text = _installable_skill_docs_text()

    generated_versions = re.findall(r'predict_rlm_version = "([^"]+)"', skill_text)
    dependency_versions = re.findall(
        r'predict-rlm(?:\[[^\]]+\])?>=([^,\s"]+)', skill_text
    )

    assert generated_versions
    assert dependency_versions
    assert set(generated_versions) == {package_version}
    assert set(dependency_versions) == {package_version}


def test_public_rlm_skill_requires_shared_eval_adapter_semantics():
    skill_text = _installable_skill_docs_text()

    assert "rlm_gepa.runtime.adapter.RLMGepaAdapter" in skill_text
    assert "eval.json" in skill_text
    assert "rlm-gepa stats <run_dir>" in skill_text


def test_public_rlm_skill_matches_current_runtime_api():
    skill_text = _installable_skill_docs_text()

    assert "max_output_chars: int = 50_000" in skill_text
    assert "CtxStr" in skill_text
    assert "adapters: Sequence[InputAdapter" in skill_text
    assert "execution: ExecutionBackend | None = None" in skill_text
    assert "modules: Sequence[RuntimeModule | RuntimeContribution] = ()" in skill_text
    assert "events: Sequence[EventSink] = ()" in skill_text
