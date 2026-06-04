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

    stale_snippets = [
        r"predict-rlm>=0\.3\.0",
        r"predict-rlm\[[^\]]+\]>=0\.4\.0",
    ]
    assert f'predict_rlm_version = "{package_version}"' in skill_text
    for snippet in stale_snippets:
        assert not re.search(snippet, skill_text), f"stale RLM skill snippet: {snippet}"


def test_public_rlm_skill_requires_shared_eval_adapter_semantics():
    skill_text = _installable_skill_docs_text()

    assert "rlm_gepa.runtime.adapter.RLMGepaAdapter" in skill_text
    assert "eval.json" in skill_text
    assert "rlm-gepa stats <run_dir>" in skill_text
