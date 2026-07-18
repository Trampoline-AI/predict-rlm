"""Regression checks for packaged RLM skill docs."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_RLM_SKILLS = ("rlm", "rlm-gepa", "predict-rlm-contributor")


def _installable_skill_docs_text() -> str:
    skill_docs = sorted((ROOT / ".agents" / "skills").glob("**/*.md"))
    return "\n".join(path.read_text() for path in skill_docs)


def test_public_rlm_skill_entrypoints_and_references_are_valid():
    for skill_name in PUBLIC_RLM_SKILLS:
        skill_dir = ROOT / ".agents" / "skills" / skill_name
        skill_file = skill_dir / "SKILL.md"
        skill_text = skill_file.read_text()

        assert skill_text.startswith("---\n")
        frontmatter, body = skill_text.removeprefix("---\n").split("\n---\n", 1)
        assert re.search(rf"(?m)^name: {re.escape(skill_name)}$", frontmatter)
        assert re.search(r"(?m)^description: [>|]?$", frontmatter)
        assert "## Skill Freshness" in body
        assert f".agents/skills/{skill_name}/SKILL.md" in body, (
            f"wrong update URL for {skill_name}"
        )

        references = re.findall(r"`(references/[^`]+\.md)`", body)
        assert references, f"{skill_name} does not link any references"
        for reference in references:
            assert (skill_dir / reference).is_file(), f"missing {skill_name}/{reference}"

        for markdown_file in skill_dir.glob("**/*.md"):
            assert markdown_file.read_text().count("```") % 2 == 0, (
                f"unbalanced code fences in {markdown_file.relative_to(ROOT)}"
            )


def test_public_rlm_skill_version_snippets_match_package_version():
    package_version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    skill_text = _installable_skill_docs_text()

    generated_versions = re.findall(r'predict_rlm_version = "([^"]+)"', skill_text)
    dependency_versions = re.findall(r'predict-rlm(?:\[[^\]]+\])?>=([^,\s"]+)', skill_text)

    assert generated_versions
    assert dependency_versions
    assert set(generated_versions) == {package_version}
    assert set(dependency_versions) == {package_version}


def test_public_rlm_skill_requires_shared_eval_adapter_semantics():
    skill_text = _installable_skill_docs_text()

    assert "rlm_gepa.runtime.adapter.RLMGepaAdapter" in skill_text
    assert "eval.json" in skill_text
    assert "rlm-gepa stats <run_dir>" in skill_text
    assert re.search(r"reserve\s+official dev/test/challenge splits for reporting", skill_text)
    assert "evaluator feedback or hidden scoring APIs" in skill_text
    assert re.search(r"typed\s+JSON boundary", skill_text)


def test_public_rlm_skill_matches_current_runtime_api():
    skill_text = _installable_skill_docs_text()

    assert "max_output_chars: int = 50_000" in skill_text
    assert "CtxStr" in skill_text
    assert "adapters: Sequence[InputAdapter" in skill_text
    assert "execution: ExecutionBackend | None = None" in skill_text
    assert "modules: Sequence[RuntimeModule | RuntimeContribution] = ()" in skill_text
    assert "events: Sequence[EventSink] = ()" in skill_text


def test_public_rlm_project_layout_is_packaged_and_importable():
    project_layout = (
        ROOT / ".agents" / "skills" / "rlm" / "references" / "project-layout.md"
    ).read_text()
    gepa_layout = (
        ROOT / ".agents" / "skills" / "rlm-gepa" / "references" / "project-layout.md"
    ).read_text()
    gepa_readme = (ROOT / "src" / "rlm_gepa" / "README.md").read_text()

    assert "├── my_rlm/" in project_layout
    assert "[build-system]" in project_layout
    assert 'packages = ["my_rlm"]' in project_layout
    assert "from .agent import AnalyzeDocuments, DocumentAnalyzer" in project_layout
    assert "│   └── gepa/" in gepa_layout
    assert "├── my_rlm/\n│   ├── agent/" in gepa_readme


def test_public_rlm_gepa_agent_spec_examples_satisfy_schema():
    agent_spec = (
        ROOT / ".agents" / "skills" / "rlm-gepa" / "references" / "agent-spec.md"
    ).read_text()
    project_layout = (
        ROOT / ".agents" / "skills" / "rlm-gepa" / "references" / "project-layout.md"
    ).read_text()

    assert '"document behaviors"' in agent_spec
    assert '"document behaviors"' in project_layout
    assert "agent_spec_from_rlm(build_rlm(SEED_SKILL_INSTRUCTIONS), ...)" not in (
        project_layout
    )
    assert "scoring_description=" in project_layout


def test_public_rlm_gepa_project_forwards_runtime_logging_flags():
    project_layout = (
        ROOT / ".agents" / "skills" / "rlm-gepa" / "references" / "project-layout.md"
    ).read_text()

    assert "verbose=context.verbose_rlm" in project_layout
    assert "debug=context.debug_rlm" in project_layout


def test_rlm_gepa_readme_matches_runtime_logging_wiring():
    gepa_readme = (ROOT / "src" / "rlm_gepa" / "README.md").read_text()

    assert "`--verbose-rlm` controls\nexecutor rollout logs" in gepa_readme
    assert "proposer RLMs emit their own\nprogress traces" in gepa_readme
    assert "`--debug-rlm` enables lifecycle diagnostics" in gepa_readme
    assert "The flags are forwarded into executor rollouts" not in gepa_readme


def test_public_rlm_gepa_cli_targets_are_importable():
    project_layout = (
        ROOT / ".agents" / "skills" / "rlm-gepa" / "references" / "project-layout.md"
    ).read_text()

    assert 'rlm-gepa = "my_rlm.gepa:main"' in project_layout
    assert "# my_rlm/gepa/__init__.py\nfrom .cli import main" in project_layout
    assert "# my_rlm/gepa/__main__.py\nfrom .cli import main" in project_layout
    assert "raise SystemExit(main())" in project_layout


def test_public_rlm_skill_routing_keeps_eval_only_work_with_rlm():
    gepa_readme = (ROOT / "src" / "rlm_gepa" / "README.md").read_text()

    assert "for eval-only projects" in gepa_readme
    assert "when evals or\nRLM-GEPA optimization" not in gepa_readme


def test_contributor_skill_uses_existing_targeted_test_node():
    repo_map = (
        ROOT / ".agents" / "skills" / "predict-rlm-contributor" / "references" / "repo-map.md"
    ).read_text()
    contributor_guidance = (repo_map, (ROOT / "AGENTS.md").read_text())

    for guidance in contributor_guidance:
        assert "TestPredictTool::test_predict_returns_dict_response" in guidance
        assert "::test_name" not in guidance
