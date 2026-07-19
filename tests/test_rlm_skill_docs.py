"""Regression checks for packaged RLM skill docs."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_public_rlm_skill_version_snippets_match_package_version():
    package_version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "version"
    ]
    skill_text = (ROOT / ".agents/skills/rlm/SKILL.md").read_text()

    if package_version != "0.4.1":
        return

    stale_snippets = [
        r"predict-rlm>=0\.3\.0",
        r"predict-rlm\[[^\]]+\]>=0\.4\.0",
    ]
    for snippet in stale_snippets:
        assert not re.search(snippet, skill_text), f"stale RLM skill snippet: {snippet}"


def test_public_rlm_skill_requires_shared_eval_adapter_semantics():
    skill_text = (ROOT / ".agents/skills/rlm/SKILL.md").read_text()

    assert "rlm_gepa.runtime.adapter.RLMGepaAdapter" in skill_text
    assert "eval.json" in skill_text
    assert "rlm-gepa stats <run_dir>" in skill_text
