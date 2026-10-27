"""Capabilities — reusable instruction + package + module + tool bundles for RLMs.

A Capability extends what the RLM's sandbox can do. It combines:

1. **Instructions** — prose guidance injected in full into the RLM's system
   prompt, teaching it *how* to use the capability.

2. **Packages** — PyPI packages installed into the REPL sandbox so the RLM
   can ``import`` and use them in generated code.

3. **Modules** — host ``.py`` files mounted into the sandbox and importable
   by the RLM's code.

4. **Tools** — optional host-side callable functions exposed to the RLM
   alongside the built-in ``predict()`` helper.

Capabilities are composable: pass one or many to a ``PredictRLM`` and the
framework merges their instructions, packages, modules, and tools.

Capabilities differ from :class:`predict_rlm.Skill`, which follows the Agent
Skills (``SKILL.md``) format: a skill only advertises its name and description
up front and is loaded on demand.

Example — defining a capability::

    from predict_rlm import Capability

    pdf_extraction = Capability(
        name="pdf-extraction",
        instructions=\"\"\"
        You are extracting structured data from PDF documents.
        Use PyMuPDF (`pymupdf`) to open documents and iterate over pages.
        Render pages when layout matters, and extract text directly when the
        content is simple.
        \"\"\",
        packages=["pymupdf"],
    )

Example — capability with tools::

    async def fetch_document(doc_id: str) -> str:
        \"\"\"Fetch a document by ID and return its URL.\"\"\"
        return await my_storage.get_signed_url(doc_id)

    doc_access = Capability(
        name="document-access",
        instructions="Use `fetch_document(doc_id)` to get document URLs.",
        tools={"fetch_document": fetch_document},
    )

Example — using capabilities with an RLM::

    from predict_rlm import PredictRLM

    rlm = PredictRLM(
        "documents -> extracted_data: list[dict]",
        capabilities=[pdf_extraction, doc_access],
        sub_lm="openai/gpt-5.4-mini",
    )
    result = rlm(documents=doc_list)
"""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, Field


class Capability(BaseModel):
    """A reusable bundle of instructions, packages, modules, and tools for an RLM.

    Attributes:
        name: Short identifier for the capability (e.g. "pdf-extraction").
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

    name: str = Field(description="Short identifier for the capability")
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


def merge_capabilities(
    capabilities: list[Capability],
) -> tuple[str, list[str], dict[str, str], dict[str, Callable]]:
    """Merge capabilities into combined instructions, packages, modules, and tools.

    Args:
        capabilities: List of Capability instances to merge.

    Returns:
        Tuple of (merged_instructions, merged_packages, merged_modules, merged_tools).
        Instructions are joined with section headers per capability.
        Packages are deduplicated preserving order.
        Modules are merged; duplicate import names raise ValueError.
        Tools are merged; duplicate names raise ValueError.
    """
    instructions_parts: list[str] = []
    seen_packages: dict[str, None] = {}
    merged_modules: dict[str, str] = {}
    merged_tools: dict[str, Callable] = {}

    for capability in capabilities:
        if capability.instructions.strip():
            instructions_parts.append(
                f"## Capability: {capability.name}\n\n{capability.instructions.strip()}"
            )

        for pkg in capability.packages:
            seen_packages.setdefault(pkg, None)

        for mod_name, mod_path in capability.modules.items():
            if mod_name in merged_modules:
                raise ValueError(
                    f"Module name conflict: '{mod_name}' is provided by multiple capabilities"
                )
            merged_modules[mod_name] = mod_path

        for tool_name, tool_fn in capability.tools.items():
            if tool_name in merged_tools:
                raise ValueError(
                    f"Tool name conflict: '{tool_name}' is provided by multiple capabilities"
                )
            merged_tools[tool_name] = tool_fn

    merged_instructions = "\n\n".join(instructions_parts)
    merged_packages = list(seen_packages.keys())

    return merged_instructions, merged_packages, merged_modules, merged_tools
