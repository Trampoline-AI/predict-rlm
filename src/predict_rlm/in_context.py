"""Prompt-injected string input type for PredictRLM signatures."""

from __future__ import annotations

import typing
from typing import Any

from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema


class CtxStr(str):
    """String input that PredictRLM also injects verbatim into the outer prompt."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(cls, core_schema.str_schema())

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        schema: core_schema.CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        return handler(core_schema.str_schema())


def is_in_context_type(annotation: Any) -> bool:
    """Check if a field annotation is the CtxStr marker."""
    return isinstance(annotation, type) and issubclass(annotation, CtxStr)


def _annotation_contains_in_context(annotation: Any) -> bool:
    if is_in_context_type(annotation):
        return True
    return any(
        _annotation_contains_in_context(arg)
        for arg in typing.get_args(annotation)
        if arg is not type(None)
    )


def scan_in_context_fields(signature: Any) -> dict[str, Any]:
    """Scan a DSPy signature for CtxStr input fields."""
    fields: dict[str, Any] = {}

    for name, field in signature.input_fields.items():
        annotation = field.annotation
        if is_in_context_type(annotation):
            fields[name] = field
        elif _annotation_contains_in_context(annotation):
            raise TypeError(
                "CtxStr inputs must be annotated directly as `field: CtxStr`; "
                f"unsupported wrapper on input field {name!r}."
            )

    output_fields = [
        name
        for name, field in signature.output_fields.items()
        if _annotation_contains_in_context(field.annotation)
    ]
    if output_fields:
        names = ", ".join(output_fields)
        raise TypeError(f"CtxStr fields are input-only: {names}.")

    return fields


def build_in_context_instructions(signature: Any, input_args: dict[str, Any]) -> str:
    """Build a prompt appendix for CtxStr inputs."""
    fields = scan_in_context_fields(signature)
    if not fields:
        return ""

    lines = [
        "## In-Context Inputs",
        "",
        "These input fields are provided verbatim as task context. Each value is "
        "also available in the REPL as the same-named Python string variable.",
    ]
    for name in fields:
        if name not in input_args:
            continue
        value = input_args[name]
        if not isinstance(value, str):
            raise TypeError(
                f"CtxStr input field {name!r} expects a string, "
                f"got {type(value).__name__}."
            )
        lines.extend(
            [
                "",
                f"### `{name}`",
                "",
                f"<BEGIN_IN_CONTEXT_INPUT name=\"{name}\">",
                value,
                f"<END_IN_CONTEXT_INPUT name=\"{name}\">",
            ]
        )

    return "\n".join(lines)


__all__ = [
    "CtxStr",
    "build_in_context_instructions",
    "is_in_context_type",
    "scan_in_context_fields",
]
