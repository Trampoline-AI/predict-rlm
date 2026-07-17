"""Prompt-injected string input type for PredictRLM signatures."""

from __future__ import annotations

import typing
from typing import Any

import dspy
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

from .runtime import InputAdapter, PreparedInput


class CtxStr(str):
    """String input whose final bound model value is injected into the outer prompt."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(cls, core_schema.str_schema())

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        schema: core_schema.CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        return handler(core_schema.str_schema())


def _is_in_context_type(annotation: object) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, CtxStr)


def _annotation_contains_in_context(annotation: object) -> bool:
    if _is_in_context_type(annotation):
        return True
    return any(
        _annotation_contains_in_context(arg)
        for arg in typing.get_args(annotation)
        if arg is not type(None)
    )


def _in_context_field_names(signature: type[dspy.Signature]) -> tuple[str, ...]:
    fields: list[str] = []

    for name, field in signature.input_fields.items():
        annotation = field.annotation
        if _is_in_context_type(annotation):
            fields.append(name)
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

    return tuple(fields)


def _in_context_delimiters(name: str, value: str) -> tuple[str, str]:
    index = 0
    while True:
        suffix = "" if index == 0 else f"_{index}"
        opening = f'<BEGIN_IN_CONTEXT_INPUT{suffix} name="{name}">'
        closing = f'<END_IN_CONTEXT_INPUT{suffix} name="{name}">'
        if opening not in value and closing not in value:
            return opening, closing
        index += 1


def _format_in_context_inputs(values: list[tuple[str, str]]) -> str:
    lines = [
        "## In-Context Inputs",
        "",
        "The final bound model strings for these input fields are provided as "
        "task context. Each value is also available in the REPL as the same-named "
        "Python string variable.",
    ]
    for name, value in values:
        opening, closing = _in_context_delimiters(name, value)
        lines.extend(
            [
                "",
                f"### `{name}`",
                "",
                opening,
                value,
                closing,
            ]
        )

    return "\n".join(lines)


class CtxStrInputAdapter(InputAdapter[CtxStr]):
    """Prepare CtxStr values and add their final bindings to the outer prompt."""

    name = "ctx_str"
    value_type = CtxStr

    async def prepare(self, field, value, ctx):
        if not isinstance(value, str):
            raise TypeError(
                f"CtxStr input field {field.name!r} expects a string, "
                f"got {type(value).__name__}."
            )
        return PreparedInput(model_value=CtxStr(value))

    def append_prompt(self, prompt, field, prepared, ctx):
        owned = [
            binding
            for binding in ctx.input_bindings.values()
            if isinstance(binding.adapter, CtxStrInputAdapter)
            if getattr(
                binding.adapter.append_prompt,
                "__func__",
                binding.adapter.append_prompt,
            )
            is CtxStrInputAdapter.append_prompt
        ]
        if not owned or owned[0].field.name != field.name:
            return prompt
        values = []
        for binding in owned:
            value = binding.prepared.model_value
            if not isinstance(value, str):
                raise TypeError(
                    f"CtxStr input field {binding.field.name!r} expects a string, "
                    f"got {type(value).__name__}."
                )
            values.append((binding.field.name, value))
        appendix = _format_in_context_inputs(values)
        return "\n\n".join(part for part in (prompt, appendix) if part)


__all__ = ["CtxStr", "CtxStrInputAdapter"]
