"""Agent Skills (SKILL.md) load from folders and are disclosed progressively."""

import asyncio
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from predict_rlm import Capability, PredictRLM, Skill


def _write_skill(root: Path, name: str, frontmatter: str, body: str = "Body.") -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(f"---\n{frontmatter}\n---\n{body}\n")
    return folder


@pytest.fixture
def invoice_skill_dir(tmp_path: Path) -> Path:
    folder = _write_skill(
        tmp_path,
        "invoice-totals",
        "name: invoice-totals\ndescription: Reconcile invoice line items with totals.\n"
        "license: MIT",
        body="# Invoice totals\n\nSum the line items, then compare with the stated total.",
    )
    (folder / "reference").mkdir()
    (folder / "reference" / "currencies.json").write_text('{"CAD": 2}')
    (folder / "helper.py").write_text("def total(items):\n    return sum(items)\n")
    return folder


def _tool(rlm: PredictRLM, name: str):
    return {tool.name: tool for tool in rlm.runtime_spec.tools}.get(name)


class TestSkillFromDir:
    def test_parses_frontmatter_body_and_resources(self, invoice_skill_dir: Path):
        skill = Skill.from_dir(invoice_skill_dir)

        assert skill.name == "invoice-totals"
        assert skill.description == "Reconcile invoice line items with totals."
        assert skill.instructions.startswith("# Invoice totals")
        assert skill.path == invoice_skill_dir.resolve()
        assert skill.resource_files == ["helper.py", "reference/currencies.json"]

    def test_utf8_bom_before_frontmatter_is_accepted(self, tmp_path: Path):
        folder = tmp_path / "bom-skill"
        folder.mkdir()
        (folder / "SKILL.md").write_bytes(
            "﻿---\nname: bom-skill\ndescription: d\n---\nBody.\n".encode()
        )

        assert Skill.from_dir(folder).name == "bom-skill"

    def test_missing_skill_md_is_rejected(self, tmp_path: Path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match="SKILL.md"):
            Skill.from_dir(tmp_path / "empty")

    def test_missing_frontmatter_is_rejected(self, tmp_path: Path):
        folder = tmp_path / "plain"
        folder.mkdir()
        (folder / "SKILL.md").write_text("# No frontmatter\n")
        with pytest.raises(ValueError, match="frontmatter"):
            Skill.from_dir(folder)

    def test_name_must_match_folder(self, tmp_path: Path):
        folder = _write_skill(tmp_path, "folder-name", "name: other-name\ndescription: d")
        with pytest.raises(ValueError, match="must match its folder name"):
            Skill.from_dir(folder)

    def test_missing_description_is_rejected(self, tmp_path: Path):
        folder = _write_skill(tmp_path, "no-desc", "name: no-desc")
        with pytest.raises(ValidationError, match="description"):
            Skill.from_dir(folder)


class TestSkillValidation:
    @pytest.mark.parametrize("name", ["Upper", "has_underscore", "-leading", "a--b", "x" * 65])
    def test_invalid_names_are_rejected(self, name: str):
        with pytest.raises(ValidationError):
            Skill(name=name, description="d", instructions="i")

    @pytest.mark.parametrize("description", ["", "x" * 1025])
    def test_description_length_is_enforced(self, description: str):
        with pytest.raises(ValidationError):
            Skill(name="ok", description=description, instructions="i")

    @pytest.mark.parametrize("field", ["packages", "modules", "tools"])
    def test_capability_fields_point_to_capability(self, field: str):
        with pytest.raises(TypeError, match="Capability"):
            Skill(name="ok", description="d", instructions="i", **{field: {}})


class TestProgressiveDisclosure:
    def test_prompt_lists_name_and_description_but_not_body(self, invoice_skill_dir: Path):
        rlm = PredictRLM("query -> answer", skills=[Skill.from_dir(invoice_skill_dir)])

        for instructions in (
            rlm.generate_action.signature.instructions,
            rlm.extract.signature.instructions,
        ):
            assert "`invoice-totals`: Reconcile invoice line items with totals." in instructions
            assert "/sandbox/skills/" in instructions
            assert "Sum the line items" not in instructions

    def test_load_skill_returns_instructions_and_files(self, invoice_skill_dir: Path):
        rlm = PredictRLM("query -> answer", skills=[Skill.from_dir(invoice_skill_dir)])

        loaded = asyncio.run(_tool(rlm, "load_skill")(name="invoice-totals"))

        assert "Sum the line items" in loaded
        assert "/sandbox/skills/invoice-totals/" in loaded
        assert "reference/currencies.json" in loaded
        assert "helper.py" in loaded

    def test_load_skill_rejects_unknown_names(self):
        rlm = PredictRLM(
            "query -> answer",
            skills=[Skill(name="inline", description="d", instructions="i")],
        )
        with pytest.raises(ValueError, match="Unknown skill 'missing'.*inline"):
            asyncio.run(_tool(rlm, "load_skill")(name="missing"))

    def test_no_load_skill_tool_without_skills(self):
        rlm = PredictRLM("query -> answer")
        assert _tool(rlm, "load_skill") is None
        assert "load_skill" not in rlm.generate_action.signature.instructions


class TestSkillWiring:
    def test_duplicate_skill_names_are_rejected(self):
        skill = Skill(name="same", description="d", instructions="i")
        with pytest.raises(ValueError, match="Duplicate skill name.*same"):
            PredictRLM("query -> answer", skills=[skill, skill])

    def test_load_skill_cannot_replace_user_tool(self):
        def load_skill(name: str) -> str:
            return name

        with pytest.raises(ValueError, match="Tool name conflict.*load_skill"):
            PredictRLM(
                "query -> answer",
                skills=[Skill(name="inline", description="d", instructions="i")],
                tools=[load_skill],
            )

    def test_capability_in_skills_points_to_capabilities(self):
        with pytest.raises(TypeError, match="capabilities="):
            PredictRLM("query -> answer", skills=[Capability(name="pdf")])


SKILL_DATA_CODE = (
    "import json\n"
    "loaded = await load_skill('invoice-totals')\n"
    "rates = json.load(open('/sandbox/skills/invoice-totals/reference/currencies.json'))\n"
    "SUBMIT(answer=f\"{'Sum the line items' in loaded}:{rates['CAD']}\")"
)
SKILL_IMPORT_CODE = (
    "import sys\n"
    "sys.path.insert(0, '/sandbox/skills/invoice-totals')\n"
    "from helper import total\n"
    "SUBMIT(answer=str(total([1, 2, 3])))"
)


def _skill_files_rlm(skill_dir: Path, code: str, **kwargs):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    rlm = PredictRLM(
        "query -> answer",
        sub_lm=MagicMock(),
        max_iterations=1,
        skills=[Skill.from_dir(skill_dir)],
        **kwargs,
    )
    rlm.generate_action = lambda **_: SimpleNamespace(plan="use the skill", code=code)
    return rlm


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("deno") is None, reason="JSPI sandbox requires Deno")
@pytest.mark.parametrize(
    ("code", "expected"), [(SKILL_DATA_CODE, "True:2"), (SKILL_IMPORT_CODE, "6")]
)
def test_skill_files_are_mounted_in_sync_runs(invoice_skill_dir: Path, code: str, expected: str):
    prediction = _skill_files_rlm(invoice_skill_dir, code)(query="total")
    assert prediction.answer == expected


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("deno") is None, reason="JSPI sandbox requires Deno")
def test_skill_files_are_mounted_in_async_runs(invoice_skill_dir: Path):
    prediction = asyncio.run(
        _skill_files_rlm(invoice_skill_dir, SKILL_DATA_CODE).acall(query="total")
    )
    assert prediction.answer == "True:2"


def test_skill_files_are_mounted_on_direct_python_backend(invoice_skill_dir: Path, tmp_path: Path):
    from predict_rlm.backends.supervisor.runner import DirectPythonBackend

    interpreter = DirectPythonBackend(
        runner_path=str(tmp_path / "runner.py"), workdir=str(tmp_path), exec_timeout=30
    )
    try:
        rlm = _skill_files_rlm(invoice_skill_dir, SKILL_DATA_CODE, interpreter=interpreter)
        prediction = rlm(query="total")
    finally:
        interpreter.shutdown()
    assert prediction.answer == "True:2"
