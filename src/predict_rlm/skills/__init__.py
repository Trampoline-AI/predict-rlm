"""Agent Skills — ``SKILL.md`` folders loaded on demand by the RLM.

A skill follows the `Agent Skills <https://agentskills.io>`_ format: a folder
holding a ``SKILL.md`` whose YAML frontmatter declares a ``name`` and a
``description``, whose Markdown body holds the instructions, and any number of
resource files next to it (reference docs, Python helpers, data)::

    invoice-totals/
    ├── SKILL.md
    ├── helper.py
    └── reference/currencies.json

Skills use progressive disclosure. The prompt only lists each skill's name
and description. When the RLM decides a skill is relevant, it calls the
built-in ``load_skill(name)`` tool to read the instructions and the list of
resource files, which are mounted in the sandbox at
``/sandbox/skills/<name>/``. The sandbox runs Pyodide, so Python and data files
in a skill are usable but shell scripts cannot run.

Skills differ from :class:`predict_rlm.Capability`, whose instructions are
always in the prompt and which also installs packages and exposes tools.

Example::

    from predict_rlm import PredictRLM, Skill

    rlm = PredictRLM(
        "invoice -> total: float",
        skills=[Skill.from_dir("skills/invoice-totals")],
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

SKILL_FILE = "SKILL.md"
SANDBOX_SKILLS_ROOT = "/sandbox/skills"

_CAPABILITY_FIELDS = ("packages", "modules", "tools")


class Skill(BaseModel):
    """An Agent Skill: a name and description up front, instructions and files on demand."""

    name: str = Field(
        max_length=64,
        pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$",
        description="Lowercase hyphenated identifier; matches the skill folder name",
    )
    description: str = Field(
        min_length=1,
        max_length=1024,
        description="What the skill does and when to use it; always shown in the prompt",
    )
    instructions: str = Field(description="SKILL.md body, returned by load_skill()")
    path: Path | None = Field(
        default=None,
        description="Skill folder whose files are mounted in the sandbox; None for inline skills",
    )

    model_config = {"frozen": True, "extra": "forbid"}

    @model_validator(mode="before")
    @classmethod
    def _reject_capability_fields(cls, data: Any) -> Any:
        if isinstance(data, dict) and any(field in data for field in _CAPABILITY_FIELDS):
            raise TypeError(
                "Skill now means an Agent Skill (SKILL.md). Use predict_rlm.Capability "
                "for bundles of instructions, packages, modules, and tools."
            )
        return data

    @classmethod
    def from_dir(cls, path: str | Path) -> Skill:
        """Load a skill from a folder containing a ``SKILL.md``."""
        folder = Path(path).resolve()
        skill_file = folder / SKILL_FILE
        if not skill_file.is_file():
            raise FileNotFoundError(f"No {SKILL_FILE} in skill folder: {folder}")

        frontmatter, body = _split_frontmatter(skill_file.read_text(encoding="utf-8"))
        if frontmatter.get("name") != folder.name:
            raise ValueError(
                f"Skill name {frontmatter.get('name')!r} in {skill_file} must match its "
                f"folder name {folder.name!r}"
            )
        return cls(
            name=frontmatter["name"],
            description=frontmatter.get("description", ""),
            instructions=body,
            path=folder,
        )

    @property
    def sandbox_path(self) -> str:
        return f"{SANDBOX_SKILLS_ROOT}/{self.name}"

    @property
    def resource_files(self) -> list[str]:
        """Skill files other than SKILL.md, as sorted paths relative to the skill folder."""
        if self.path is None:
            return []
        return sorted(
            file.relative_to(self.path).as_posix()
            for file in self.path.rglob("*")
            if file.is_file() and file != self.path / SKILL_FILE
        )


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{SKILL_FILE} must start with YAML frontmatter delimited by '---'")
    try:
        end = next(i for i, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    except StopIteration:
        raise ValueError(f"{SKILL_FILE} frontmatter is not closed by '---'") from None
    frontmatter = yaml.safe_load("\n".join(lines[1:end])) or {}
    if not isinstance(frontmatter, dict):
        raise ValueError(f"{SKILL_FILE} frontmatter must be a YAML mapping")
    return frontmatter, "\n".join(lines[end + 1 :]).strip()


def format_skill_catalog(skills: list[Skill]) -> str:
    """Render the always-visible part of each skill: its name and description."""
    if not skills:
        return ""
    entries = "\n".join(f"- `{skill.name}`: {skill.description}" for skill in skills)
    return (
        "Skills are loaded on demand. When the task matches a skill's description, call "
        "`await load_skill(name)` to read its instructions and list its files. Skill files "
        f"are mounted at {SANDBOX_SKILLS_ROOT}/<name>/. Python and data files there "
        "can be used; shell scripts cannot run in this sandbox.\n\n"
        f"{entries}"
    )


def create_load_skill_tool(skills: list[Skill]):
    """Build the ``load_skill`` tool that discloses a skill's instructions and files."""
    by_name = {skill.name: skill for skill in skills}

    async def load_skill(name: str) -> str:
        """Load a skill's full instructions and the list of its files.

        Args:
            name: Skill name, as listed in the Skills section.

        Returns:
            The skill's instructions followed by its file paths in the sandbox.
        """
        skill = by_name.get(name)
        if skill is None:
            raise ValueError(f"Unknown skill {name!r}. Available skills: {', '.join(by_name)}")
        files = skill.resource_files
        if not files:
            return skill.instructions
        listing = "\n".join(f"- {file}" for file in files)
        return f"{skill.instructions}\n\nFiles in {skill.sandbox_path}/:\n{listing}"

    return load_skill
