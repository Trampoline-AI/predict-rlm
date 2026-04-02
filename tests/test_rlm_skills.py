"""Tests for the Skill model and merge_skills."""

import pytest

from predict_rlm.rlm_skills import Skill, merge_skills


class TestSkill:
    def test_minimal_skill(self):
        skill = Skill(name="test")
        assert skill.name == "test"
        assert skill.instructions == ""
        assert skill.packages == []
        assert skill.tools == {}

    def test_full_skill(self):
        def my_tool(x: str) -> str:
            return x

        skill = Skill(
            name="full",
            instructions="Do the thing.",
            packages=["pandas", "numpy"],
            tools={"my_tool": my_tool},
        )
        assert skill.name == "full"
        assert skill.instructions == "Do the thing."
        assert skill.packages == ["pandas", "numpy"]
        assert "my_tool" in skill.tools


class TestMergeSkills:
    def test_empty_list(self):
        instructions, packages, modules, tools = merge_skills([])
        assert instructions == ""
        assert packages == []
        assert tools == {}

    def test_single_skill(self):
        skill = Skill(
            name="pdf",
            instructions="Use pdfplumber.",
            packages=["pdfplumber"],
        )
        instructions, packages, modules, tools = merge_skills([skill])
        assert "Skill: pdf" in instructions
        assert "Use pdfplumber." in instructions
        assert packages == ["pdfplumber"]
        assert tools == {}

    def test_multiple_skills_merge_instructions(self):
        s1 = Skill(name="a", instructions="Do A.")
        s2 = Skill(name="b", instructions="Do B.")
        instructions, _, _, _ = merge_skills([s1, s2])
        assert "Skill: a" in instructions
        assert "Skill: b" in instructions
        assert "Do A." in instructions
        assert "Do B." in instructions

    def test_package_dedup(self):
        s1 = Skill(name="a", packages=["pandas", "numpy"])
        s2 = Skill(name="b", packages=["numpy", "scipy"])
        _, packages, _, _ = merge_skills([s1, s2])
        assert packages == ["pandas", "numpy", "scipy"]

    def test_tool_merge(self):
        def tool_a():
            pass

        def tool_b():
            pass

        s1 = Skill(name="a", tools={"tool_a": tool_a})
        s2 = Skill(name="b", tools={"tool_b": tool_b})
        _, _, _, tools = merge_skills([s1, s2])
        assert set(tools.keys()) == {"tool_a", "tool_b"}

    def test_tool_name_conflict_raises(self):
        def tool_x():
            pass

        s1 = Skill(name="a", tools={"shared": tool_x})
        s2 = Skill(name="b", tools={"shared": tool_x})
        with pytest.raises(ValueError, match="Tool name conflict.*shared"):
            merge_skills([s1, s2])

    def test_empty_instructions_skipped(self):
        s1 = Skill(name="a", instructions="")
        s2 = Skill(name="b", instructions="  ")
        s3 = Skill(name="c", instructions="Real instructions.")
        instructions, _, _, _ = merge_skills([s1, s2, s3])
        assert "Skill: a" not in instructions
        assert "Skill: b" not in instructions
        assert "Skill: c" in instructions

    def test_module_merge(self):
        s1 = Skill(name="a", modules={"formula_eval": "/path/to/formula_eval.py"})
        s2 = Skill(name="b", modules={"chart_helper": "/path/to/chart_helper.py"})
        _, _, modules, _ = merge_skills([s1, s2])
        assert modules == {
            "formula_eval": "/path/to/formula_eval.py",
            "chart_helper": "/path/to/chart_helper.py",
        }

    def test_module_name_conflict_raises(self):
        s1 = Skill(name="a", modules={"shared_mod": "/path/a.py"})
        s2 = Skill(name="b", modules={"shared_mod": "/path/b.py"})
        with pytest.raises(ValueError, match="Module name conflict.*shared_mod"):
            merge_skills([s1, s2])

    def test_single_skill_with_modules(self):
        s = Skill(name="spreadsheet", modules={"formula_eval": "/path/formula_eval.py"})
        _, _, modules, _ = merge_skills([s])
        assert modules == {"formula_eval": "/path/formula_eval.py"}

    def test_empty_modules_skipped(self):
        s1 = Skill(name="a", modules={})
        s2 = Skill(name="b", modules={"mod": "/path/mod.py"})
        _, _, modules, _ = merge_skills([s1, s2])
        assert modules == {"mod": "/path/mod.py"}
