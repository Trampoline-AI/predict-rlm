"""Shared utilities for RLM subclasses."""

from __future__ import annotations

import inspect
import textwrap
import typing
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Callable

import dspy
from dspy.adapters.utils import translate_field_type

if TYPE_CHECKING:
    from dspy.signatures.signature import Signature


def format_tool_docs_full(tools: dict[str, Callable]) -> str:
    """Format tools with full docstrings for inclusion in instructions.

    Unlike DSPy's default _format_tool_docs which only uses the first line of
    the docstring, this includes the full docstring (Args, Returns, etc.).
    """
    if not tools:
        return ""

    lines = [
        "\n## Additional Tools\n\nAll tools are async — use `await` when calling them. Use `asyncio.gather()` to run multiple tool calls in parallel."
    ]
    for name, func in tools.items():
        # Get function signature with types
        try:
            sig = inspect.signature(func)
            # Resolve string annotations (from `from __future__ import annotations`)
            try:
                resolved = typing.get_type_hints(func, include_extras=True)
            except (TypeError, NameError):
                resolved = {}

            params = []
            for p in sig.parameters.values():
                ann = resolved.get(p.name)
                if ann is not None:
                    # Unwrap Annotated[X, ...] → X (e.g. SyncedFile markers)
                    if typing.get_origin(ann) is Annotated:
                        ann = typing.get_args(ann)[0]
                    # Show Path as str — the RLM passes sandbox paths as strings
                    if ann is Path:
                        ann = str
                    type_name = getattr(ann, "__name__", str(ann))
                    params.append(f"{p.name}: {type_name}")
                else:
                    params.append(p.name)
            params_str = ", ".join(params)

            # Get return type
            ret_ann = resolved.get("return")
            if ret_ann is not None:
                ret_type = getattr(ret_ann, "__name__", str(ret_ann))
                sig_str = f"{name}({params_str}) -> {ret_type}"
            else:
                sig_str = f"{name}({params_str})"
        except (ValueError, TypeError):
            sig_str = f"{name}(...)"

        # Get full docstring, cleaned up
        if func.__doc__:
            doc = textwrap.dedent(func.__doc__).strip()
        else:
            doc = "No description"

        lines.append(f"\n### `await {sig_str}`")
        lines.append(doc)

    return "\n".join(lines)


def build_rlm_signatures(
    signature: Signature,
    instructions_template: str,
    user_tools: dict[str, Callable],
    format_tool_docs: Callable[[dict[str, Callable]], str],
    skill_instructions: str = "",
    file_instructions: str = "",
) -> tuple[Signature, Signature]:
    """Build action and extract signatures for RLM subclasses.

    Full override of base RLM because its ACTION_INSTRUCTIONS_TEMPLATE embeds
    llm_query/llm_query_batched docs. Since instructions are baked into Signature
    at creation, we rebuild with custom instructions.
    """
    inputs_str = ", ".join(f"`{n}`" for n in signature.input_fields)
    final_output_names = ", ".join(signature.output_fields.keys())

    output_fields = "\n".join(
        f"- {translate_field_type(n, f)}" for n, f in signature.output_fields.items()
    )

    # Include original signature instructions if present
    task_instructions = f"{signature.instructions}\n\n" if signature.instructions else ""

    # Format tool documentation for user-provided tools
    tool_docs = format_tool_docs(user_tools)

    # Build the full instructions with optional skill instructions
    full_instructions = (
        task_instructions
        + instructions_template.format(
            inputs=inputs_str,
            final_output_names=final_output_names,
            output_fields=output_fields,
        )
        + tool_docs
    )
    if file_instructions:
        full_instructions += f"\n\n{file_instructions}"
    if skill_instructions:
        full_instructions += f"\n\n## Skills\n\n{skill_instructions}"

    action_sig = dspy.Signature({}, full_instructions)
    action_sig = action_sig.append(
        "variables_info",
        dspy.InputField(desc="Metadata about the variables available in the REPL"),
        type_=str,
    )
    action_sig = action_sig.append(
        "repl_history",
        dspy.InputField(desc="Previous REPL code executions and their outputs"),
        type_=dspy.primitives.repl_types.REPLHistory,
    )
    action_sig = action_sig.append(
        "iteration",
        dspy.InputField(desc="Current iteration number (1-indexed) out of max_iterations"),
        type_=str,
    )
    action_sig = action_sig.append(
        "reasoning",
        dspy.OutputField(
            desc="Think step-by-step: what do you know? What remains? Plan your next action."
        ),
        type_=str,
    )
    action_sig = action_sig.append(
        "code",
        dspy.OutputField(desc="Python code wrapped in ```python blocks."),
        type_=str,
    )

    # Extract signature with original task instructions
    extract_instructions = """Based on the REPL trajectory, extract the final outputs now.

        Review your trajectory to see what information you gathered and what values you computed, then provide the final outputs."""

    extended_task_instructions = ""
    if task_instructions:
        extended_task_instructions = (
            "The trajectory was generated with the following objective: \n"
            + task_instructions
            + "\n"
        )
    full_extract_instructions = extended_task_instructions + extract_instructions

    extract_sig = dspy.Signature(
        {**signature.output_fields},
        full_extract_instructions,
    )
    extract_sig = extract_sig.prepend(
        "repl_history",
        dspy.InputField(desc="Your REPL interactions so far"),
        type_=dspy.primitives.repl_types.REPLHistory,
    )
    extract_sig = extract_sig.prepend(
        "variables_info",
        dspy.InputField(desc="Metadata about the variables available in the REPL"),
        type_=str,
    )

    return action_sig, extract_sig
