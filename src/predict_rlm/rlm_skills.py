"""Skills — reusable instruction + package bundles for RLMs.

A Skill packages domain-specific knowledge that an RLM needs to solve a
particular class of problems. It combines:

1. **Instructions** — prose guidance injected into the RLM's system prompt,
   teaching it *how* to approach the task and what patterns to follow.

2. **Packages** — PyPI packages installed into the REPL sandbox so the RLM
   can ``import`` and use them in generated code.

3. **Tools** — optional callable functions exposed to the RLM alongside the
   built-in ``predict()`` / ``sub_lm_query()`` tools.

Skills are composable: pass one or many to a ``PredictRLM`` or ``VisionRLM``
and the framework merges their instructions, packages, and tools
automatically.

Example — defining a skill::

    from predict_rlm import Skill

    pdf_extraction = Skill(
        name="pdf-extraction",
        instructions=\"\"\"
        You are extracting structured data from PDF documents.
        Use `pdfplumber` to open documents and iterate over pages.
        Prefer table extraction (`page.extract_tables()`) over raw text
        when the content is tabular.
        \"\"\",
        packages=["pdfplumber"],
    )

Example — skill with tools::

    async def fetch_document(doc_id: str) -> str:
        \"\"\"Fetch a document by ID and return its URL.\"\"\"
        return await my_storage.get_signed_url(doc_id)

    doc_skill = Skill(
        name="document-access",
        instructions="Use `fetch_document(doc_id)` to get document URLs.",
        packages=[],
        tools={"fetch_document": fetch_document},
    )

Example — using skills with an RLM::

    from predict_rlm import PredictRLM

    rlm = PredictRLM(
        "documents -> extracted_data: list[dict]",
        skills=[pdf_extraction, doc_skill],
        sub_lm="anthropic/claude-sonnet-4-5-20250929",
    )
    result = rlm(documents=doc_list)
"""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, Field


class Skill(BaseModel):
    """A reusable bundle of instructions, packages, modules, and tools for an RLM.

    Attributes:
        name: Short identifier for the skill (e.g. "pdf-extraction").
        instructions: Prose instructions injected into the RLM prompt.
            Describes patterns, best practices, and domain knowledge
            the RLM should apply.
        packages: PyPI package names to install in the REPL sandbox.
            These are installed via ``micropip`` in Pyodide before
            the RLM's first code execution.
        modules: Python module files to mount into the sandbox. Maps
            import name to the host filesystem path of the .py file.
            The module becomes importable by the RLM code in the sandbox.
        tools: Optional tool functions exposed to the RLM. Keys are
            tool names visible in the REPL, values are callables.
            Can be sync or async — async is preferred for I/O.
    """

    name: str = Field(description="Short identifier for the skill")
    instructions: str = Field(
        default="",
        description="Prose instructions injected into the RLM prompt",
    )
    packages: list[str] = Field(
        default_factory=list,
        description="PyPI packages to install in the REPL sandbox",
    )
    modules: dict[str, str] = Field(
        default_factory=dict,
        description="Python modules to mount in sandbox: {import_name: host_path}",
    )
    tools: dict[str, Callable[..., Any]] = Field(
        default_factory=dict,
        description="Tool functions exposed to the RLM",
    )

    model_config = {"arbitrary_types_allowed": True}


def merge_skills(
    skills: list[Skill],
) -> tuple[str, list[str], dict[str, str], dict[str, Callable]]:
    """Merge multiple skills into combined instructions, packages, modules, and tools.

    Args:
        skills: List of Skill instances to merge.

    Returns:
        Tuple of (merged_instructions, merged_packages, merged_modules, merged_tools).
        Instructions are joined with section headers per skill.
        Packages are deduplicated preserving order.
        Modules are merged; duplicate import names raise ValueError.
        Tools are merged; duplicate names raise ValueError.
    """
    instructions_parts: list[str] = []
    seen_packages: dict[str, None] = {}
    merged_modules: dict[str, str] = {}
    merged_tools: dict[str, Callable] = {}

    for skill in skills:
        if skill.instructions.strip():
            instructions_parts.append(f"## Skill: {skill.name}\n\n{skill.instructions.strip()}")

        for pkg in skill.packages:
            seen_packages.setdefault(pkg, None)

        for mod_name, mod_path in skill.modules.items():
            if mod_name in merged_modules:
                raise ValueError(
                    f"Module name conflict: '{mod_name}' is provided by multiple skills"
                )
            merged_modules[mod_name] = mod_path

        for tool_name, tool_fn in skill.tools.items():
            if tool_name in merged_tools:
                raise ValueError(
                    f"Tool name conflict: '{tool_name}' is provided by multiple skills"
                )
            merged_tools[tool_name] = tool_fn

    merged_instructions = "\n\n".join(instructions_parts)
    merged_packages = list(seen_packages.keys())

    return merged_instructions, merged_packages, merged_modules, merged_tools
