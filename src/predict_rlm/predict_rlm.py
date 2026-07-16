"""PredictRLM - RLM subclass with predict tool for DSPy signatures."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import inspect
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from copy import copy, deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

import dspy
import json_repair
import regex
from dspy.adapters.base import Adapter
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.primitives.code_interpreter import CodeInterpreter, CodeInterpreterError, FinalOutput
from dspy.utils.callback import ACTIVE_CALL_ID
from dspy.utils.exceptions import AdapterParseError
from litellm import ContextWindowExceededError
from pydantic import ConfigDict, Field, ValidationError, create_model

from ._logging import (
    configure_predict_rlm_logging,
    emit_trace_iteration_end,
    emit_trace_iteration_output,
    emit_trace_iteration_start,
    emit_trace_iteration_submit,
    format_log_fields,
    live_tool_call_logging,
    suppress_interpreter_result_logging,
)
from ._shared import build_rlm_signatures, format_tool_docs_full, strip_code_fences
from .backends import (
    BackendName,
)
from .backends.adapters import (
    ExistingExecutionBackendAdapter,
)
from .backends.base import LegacyExecutionBackend, SandboxFatalError
from .backends.jspi import JspiBackend
from .compatibility import (
    SyncedFileToolOperation,
    ValueInputAdapter,
    execution_from_options,
)
from .compatibility import (
    files as file_compatibility,
)
from .evidence import EvidenceRecorder, RunEventKind
from .execution_timeout import validate_execution_timeout
from .files import File, build_file_plan, scan_file_fields
from .rlm_skills import Skill, merge_skills
from .runtime import (
    Artifact,
    CallableTool,
    EventSink,
    ExecutionBackend,
    ExecutionFatalError,
    ExecutionResult,
    ExecutionSpec,
    FieldDescriptor,
    InputAdapter,
    OutputAdapter,
    PreparedInputBinding,
    RunContext,
    RuntimeContribution,
    RuntimeModule,
    RuntimeSpec,
    SessionRequirements,
    compile_prepared_input,
    current_run_context,
    invoke_host_callable,
    merge_session_requirements,
    preserve_sync_leaf,
    resolve_input_adapter,
    resolve_output_adapter,
    resolve_runtime_spec,
    use_run_context,
    validate_output_sandbox_root_reservation,
    validate_sandbox_root_reservations,
)
from .runtime_hooks import RuntimeHook, RuntimeHookEvent
from .telemetry import TelemetryContext, make_span_id
from .trace import (
    IterationStep,
    LMUsage,
    RunEvidence,
    RunEvidenceEvent,
    RunTrace,
    TokenUsage,
    _RawPredictCall,
    drain_predict_calls,
    drain_tool_calls,
    init_predict_call_collector,
    init_tool_call_collector,
    lm_completion_metadata_since,
    lm_finish_since,
    ms_since,
    record_predict_call,
    reset_predict_call_collector,
    reset_tool_call_collector,
    snapshot_lm_history_len,
    usage_since,
)

_CB_LOGGER = logging.getLogger("predict_rlm.callbacks")


@dataclass
class _IterationCallbackState:
    step: IterationStep | None = None
    is_final: bool = False
    exception: Exception | None = None


@dataclass
class _TraceExportContext:
    steps: list[IterationStep]
    lm: Any
    sub_lm: Any
    lm_hist_start: int
    sub_hist_start: int
    run_start: float


class _RunStateField:
    """Store mutable invocation state on the active RunContext."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        if instance is None:
            return self
        ctx = current_run_context()
        if ctx is not None and self.name in ctx.state:
            return ctx.state[self.name]
        return instance.__dict__.get(self.name)

    def __set__(self, instance: Any, value: Any) -> None:
        ctx = current_run_context()
        if ctx is not None:
            ctx.state[self.name] = value
        else:
            instance.__dict__[self.name] = value


class _ExecutionSessionRepl:
    """Compatibility view used while the iteration loop migrates to sessions."""

    def __init__(
        self,
        session: Any,
        run_code: Callable[..., Any] | None = None,
    ) -> None:
        self._session = session
        self._run_code = run_code or session.run_code

    async def aexecute(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        result = await self._run_code(code, variables, timeout=timeout)
        return result.value if isinstance(result, ExecutionResult) else result

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(
            f"Execution session {type(self._session).__name__} does not expose legacy "
            f"interpreter operation {name!r}"
        )


# Capture the real dspy.Image class at import time so type comparisons
# work even when tests patch predict_rlm.dspy.Image to a mock.
_ImageType = dspy.Image

_PARENT_TAKES_CODE = "code" in inspect.signature(dspy.RLM._process_execution_result).parameters
_DSPY_EXTRACT_FALLBACK = dspy.RLM._extract_fallback
logger = logging.getLogger(__name__)

_NO_FIELD_DEFAULT = object()
_RUN_PROMPT_SIGNATURE: ContextVar[tuple[object, Any] | None] = ContextVar(
    "predict_rlm_run_prompt_signature", default=None
)


def _get_field_default(field: Any) -> Any:
    if hasattr(field, "is_required") and field.is_required():
        return _NO_FIELD_DEFAULT
    if hasattr(field, "get_default"):
        return field.get_default(call_default_factory=True)
    if hasattr(field, "default_factory") and field.default_factory is not None:
        return field.default_factory()
    default = getattr(field, "default", _NO_FIELD_DEFAULT)
    if type(default).__name__ == "PydanticUndefinedType":
        return _NO_FIELD_DEFAULT
    return default


def _to_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return _to_json_safe(value.model_dump(mode="python"))
    if isinstance(value, dict):
        return {k: _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(v) for v in value]
    return value


@runtime_checkable
class RuntimeLoggingConfigurable(Protocol):
    def configure_runtime(
        self,
        *,
        debug: bool | None = None,
        verbose: bool | None = None,
    ) -> Any: ...


@runtime_checkable
class DebugLoggingConfigurable(Protocol):
    def configure_debug(self, enabled: bool) -> Any: ...


@runtime_checkable
class VerboseLoggingConfigurable(Protocol):
    def configure_verbose(self, enabled: bool) -> Any: ...


def _format_execution_error(code: str, exc: Exception) -> str:
    err = getattr(exc, "error_message", str(exc))
    partial_output = getattr(exc, "partial_output", "")
    if code.count('"""') >= 3 and (
        "unterminated" in err.lower() or "invalid syntax" in err.lower()
    ):
        error = (
            f"[Error] {err}\n\n"
            'Hint: your code contains nested `"""` inside a '
            "triple-quoted string literal. Python cannot distinguish "
            "the inner delimiter from the outer one. Fix: build the "
            "string with '\\n'.join([line1, line2, ...]) using "
            "single-quoted strings per line, or switch the outer "
            "delimiter to '''...'''."
        )
    else:
        error = f"[Error] {err}"
    return _prepend_partial_output(partial_output, error)


def _prepend_partial_output(partial_output: str, error: str) -> str:
    partial_output = partial_output.rstrip()
    if not partial_output:
        return error
    return f"{partial_output}\n{error}"


def _shorten_repl_output(output: str, max_output_chars: int) -> str:
    if len(output) <= max_output_chars:
        return output
    return (
        output[:max_output_chars]
        + f"\n... (truncated to {max_output_chars}/{len(output):,} chars)"
    )


def _output_indicates_error(output: str) -> bool:
    return any(
        line.startswith(("[Error]", "[Type Error]")) for line in output.splitlines()
    )


def _sha256_text(value: str) -> str:
    return "sha256_" + hashlib.sha256(value.encode("utf-8")).hexdigest()


if TYPE_CHECKING:
    from dspy.signatures.signature import Signature

    from .backends.sbx import SbxConfig, SbxPool


@dataclass(frozen=True)
class SubmitConfirmationContext:
    """Context passed to PredictRLM submit-confirmation callbacks."""

    inputs: dict[str, Any]
    signature: type[Any]
    output_field_names: tuple[str, ...]
    prediction: dspy.Prediction
    submitted_payload: dict[str, Any]
    history: Any
    latest_observation: str
    reasoning: str
    code: str
    iteration: int


_OUTPUT_VALIDATION_MODELS: dict[type[Any], type[Any]] = {}


def _annotation_allows_none(annotation: Any) -> bool:
    """Return True when a type annotation explicitly accepts None."""
    return annotation is Any or FieldDescriptor("", annotation).allows_none


def _output_validation_model(signature: type[Any]) -> type[Any]:
    """Build a Pydantic model containing only a signature's output fields."""
    model = _OUTPUT_VALIDATION_MODELS.get(signature)
    if model is not None:
        return model

    fields = {
        name: (field.annotation, deepcopy(field))
        for name, field in signature.output_fields.items()
    }
    model = create_model(
        f"{signature.__name__}Outputs",
        __config__=ConfigDict(arbitrary_types_allowed=True, extra="ignore"),
        **fields,
    )
    _OUTPUT_VALIDATION_MODELS[signature] = model
    return model


def _raise_output_validation_error(
    *,
    adapter_name: str,
    signature: type[Any],
    lm_response: str,
    parsed_result: dict[str, Any],
    error: Exception,
) -> None:
    raise AdapterParseError(
        adapter_name=adapter_name,
        signature=signature,
        lm_response=lm_response,
        parsed_result=parsed_result,
        message=f"Parsed output failed client-side signature validation: {error}",
    )


def _validate_signature_outputs(
    *,
    adapter_name: str,
    signature: type[Any],
    parsed_result: dict[str, Any],
    lm_response: str,
) -> None:
    try:
        _output_validation_model(signature).model_validate(parsed_result)
    except ValidationError as e:
        _raise_output_validation_error(
            adapter_name=adapter_name,
            signature=signature,
            lm_response=lm_response,
            parsed_result=parsed_result,
            error=e,
        )


def _with_optional_output_defaults(
    signature: type[Any],
    parsed_result: Any,
) -> dict[str, Any] | None:
    if not isinstance(parsed_result, dict):
        return None
    missing = [
        name for name in signature.output_fields
        if name not in parsed_result
    ]
    if not missing:
        return parsed_result
    if any(signature.output_fields[name].is_required() for name in missing):
        return None
    filled = dict(parsed_result)
    for name in missing:
        filled[name] = signature.output_fields[name].default
    return filled


class _ValidatingOutputAdapterMixin:
    """Validate adapter outputs before they become DSPy Predictions."""

    def _call_postprocess(
        self,
        processed_signature: type[Any],
        original_signature: type[Any],
        outputs: list[dict[str, Any] | str],
        lm: Any,
        lm_kwargs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        values = super()._call_postprocess(  # type: ignore[misc]
            processed_signature,
            original_signature,
            outputs,
            lm,
            lm_kwargs,
        )
        for output, value in zip(outputs, values, strict=False):
            lm_response = output.get("text", "") if isinstance(output, dict) else output
            _validate_signature_outputs(
                adapter_name=type(self).__name__,
                signature=original_signature,
                parsed_result=value,
                lm_response=str(lm_response or ""),
            )
        return values


class _ValidatingJSONAdapter(_ValidatingOutputAdapterMixin, JSONAdapter):
    """JSONAdapter that rejects raw nulls for required output fields."""

    def parse(self, signature: type[Any], completion: str) -> dict[str, Any]:
        self._reject_required_json_nulls(signature, completion)
        try:
            fields = super().parse(signature, completion)
        except AdapterParseError as exc:
            fields = _with_optional_output_defaults(
                signature,
                getattr(exc, "parsed_result", None),
            )
            if fields is None:
                raise
        _validate_signature_outputs(
            adapter_name=type(self).__name__,
            signature=signature,
            parsed_result=fields,
            lm_response=completion,
        )
        return fields

    def _reject_required_json_nulls(
        self,
        signature: type[Any],
        completion: str,
    ) -> None:
        try:
            fields = json_repair.loads(completion)
            if not isinstance(fields, dict):
                match = regex.search(
                    r"\{(?:[^{}]|(?R))*\}",
                    completion,
                    regex.DOTALL,
                )
                fields = json_repair.loads(match.group(0)) if match else None
        except Exception:
            return

        if not isinstance(fields, dict):
            return

        parsed_result = {k: v for k, v in fields.items() if k in signature.output_fields}
        for name, value in parsed_result.items():
            field = signature.output_fields[name]
            if value is None and not _annotation_allows_none(field.annotation):
                _raise_output_validation_error(
                    adapter_name=type(self).__name__,
                    signature=signature,
                    lm_response=completion,
                    parsed_result=parsed_result,
                    error=ValueError(f"Field {name!r} is required and cannot be null."),
                )


class _ValidatingChatAdapter(_ValidatingOutputAdapterMixin, ChatAdapter):
    """ChatAdapter whose fallback preserves client-side output validation."""

    def parse(self, signature: type[Any], completion: str) -> dict[str, Any]:
        try:
            fields = super().parse(signature, completion)
        except AdapterParseError as exc:
            fields = _with_optional_output_defaults(
                signature,
                getattr(exc, "parsed_result", None),
            )
            if fields is None:
                raise
        _validate_signature_outputs(
            adapter_name=type(self).__name__,
            signature=signature,
            parsed_result=fields,
            lm_response=completion,
        )
        return fields

    def __call__(
        self,
        lm: Any,
        lm_kwargs: dict[str, Any],
        signature: type[Any],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        try:
            return Adapter.__call__(self, lm, lm_kwargs, signature, demos, inputs)
        except Exception as e:
            if isinstance(e, ContextWindowExceededError) or not self.use_json_adapter_fallback:
                raise e
            return _ValidatingJSONAdapter()(
                lm,
                lm_kwargs,
                signature,
                demos,
                inputs,
            )

    async def acall(
        self,
        lm: Any,
        lm_kwargs: dict[str, Any],
        signature: type[Any],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        try:
            return await Adapter.acall(self, lm, lm_kwargs, signature, demos, inputs)
        except Exception as e:
            if isinstance(e, ContextWindowExceededError) or not self.use_json_adapter_fallback:
                raise e
            return await _ValidatingJSONAdapter().acall(
                lm,
                lm_kwargs,
                signature,
                demos,
                inputs,
            )


_NO_RECOVERABLE_DEFAULT = object()


def _schema_field_default(field_info: dict) -> Any:
    """Default for a not-required field whose schema carries no explicit ``default``.

    These are almost always ``default_factory`` collection fields (pydantic does not
    serialize a factory's value into the JSON schema). Return a type-appropriate empty
    default so the field stays OMITTABLE without becoming NULLABLE. Returns the
    ``_NO_RECOVERABLE_DEFAULT`` sentinel when no sensible default can be inferred
    (e.g. a scalar ``default_factory``), so the caller can fall back to a lenient type.
    """
    schema_type = field_info.get("type")
    if schema_type == "array":
        return Field(default_factory=list)
    if schema_type == "object":
        return Field(default_factory=dict)
    return _NO_RECOVERABLE_DEFAULT


def _models_from_schema(schema: dict) -> dict[str, type]:
    """Build Pydantic models from a JSON schema including $defs for nested types.

    Given a JSON schema from model_json_schema(), reconstructs an equivalent
    Pydantic model. Handles nested models via $ref references in the schema.

    Args:
        schema: JSON schema dict from Pydantic's model_json_schema()

    Returns:
        Dict mapping model names to dynamically created Pydantic model classes.
        Includes both the root model and any nested models from $defs.

    Example:
        >>> class Address(BaseModel):
        ...     street: str
        ...     city: str
        >>> class Person(BaseModel):
        ...     name: str
        ...     address: Address
        >>> schema = Person.model_json_schema()
        >>> models = _models_from_schema(schema)
        >>> models.keys()
        dict_keys(['Address', 'Person'])
    """
    defs = schema.get("$defs", {})
    built_models: dict[str, type] = {}

    def get_python_type(info: dict) -> type:
        """Convert JSON schema type info to Python type, handling $ref."""
        # Handle $ref to nested model
        if "$ref" in info:
            ref_name = info["$ref"].split("/")[-1]  # "#/$defs/Address" -> "Address"
            if ref_name not in built_models:
                # Build referenced model first (recursion)
                if ref_name in defs:
                    _build_model(ref_name, defs[ref_name])
                else:
                    # Reference to unknown model, fall back to dict
                    return dict
            return built_models[ref_name]

        # Handle array types
        if info.get("type") == "array":
            item_info = info.get("items", {})
            item_type = get_python_type(item_info)
            return List[item_type]  # type: ignore[valid-type]

        # Handle anyOf (Optional or Union types)
        if "anyOf" in info:
            non_null = [t for t in info["anyOf"] if t.get("type") != "null"]
            if non_null:
                inner = get_python_type(non_null[0])
                # Check if null is one of the options (Optional)
                has_null = any(t.get("type") == "null" for t in info["anyOf"])
                if has_null:
                    return Optional[inner]  # type: ignore[valid-type]
                return inner
            return Optional[str]  # type: ignore[return-value]

        # Handle enum (Literal) types — e.g. {"enum": ["p1", "p2"], "type": "string"}
        if "enum" in info:
            return Literal[tuple(info["enum"])]  # type: ignore[valid-type]

        # Handle type-array shorthand: {"type": ["string", "null"]} → Optional[str]
        # This is valid JSON Schema (draft-07+) but not produced by Pydantic;
        # LMs write it by hand instead of using anyOf.
        raw_type = info.get("type", "string")
        if isinstance(raw_type, list):
            non_null = [t for t in raw_type if t != "null"]
            has_null = "null" in raw_type
            inner = get_python_type({"type": non_null[0]}) if non_null else str
            return Optional[inner] if has_null else inner  # type: ignore[valid-type]

        # Handle primitive types
        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "object": dict,
            "array": list,
        }
        return type_map.get(raw_type, str)

    def _build_model(name: str, model_schema: dict) -> None:
        """Build a single model, recursively building dependencies first."""
        if name in built_models:
            return

        fields: dict[str, tuple] = {}
        props = model_schema.get("properties", {})
        required = set(model_schema.get("required", []))

        for field_name, field_info in props.items():
            py_type = get_python_type(field_info)
            # required-ness controls the DEFAULT, not nullability. A field's
            # nullability is already encoded in its type (get_python_type maps
            # anyOf-with-null to Optional); "not required" only means "may be omitted
            # because it has a default", NOT "may be null". Reconstructing every
            # not-required field as Optional[T]=None (the old behavior) handed the LM
            # a weaker schema than the user's model -- so the LM emitted null for
            # defaulted non-Optional fields, and the original model then rejected it.
            if field_name in required:
                fields[field_name] = (py_type, ...)
            elif "default" in field_info:
                # Preserve the real default (e.g. mandatory: bool = True); keep the
                # declared, possibly non-nullable type.
                fields[field_name] = (py_type, field_info["default"])
            else:
                # Not required, no explicit default in the schema (default_factory).
                # Provide a type-appropriate empty default so the field stays
                # omittable without advertising null to the LM.
                default = _schema_field_default(field_info)
                if default is _NO_RECOVERABLE_DEFAULT:
                    # Can't recover the default (e.g. a scalar default_factory); fall
                    # back to the lenient Optional[T]=None so we don't over-constrain.
                    fields[field_name] = (Optional[py_type], None)  # type: ignore[valid-type]
                else:
                    fields[field_name] = (py_type, default)

        built_models[name] = create_model(name, **fields)  # type: ignore[call-overload]

    # Build all models from $defs first (nested models)
    for def_name, def_schema in defs.items():
        _build_model(def_name, def_schema)

    # Build the root model (uses "title" as name)
    root_name = schema.get("title", "RootModel")
    _build_model(root_name, schema)

    return built_models


PREDICT_RLM_INSTRUCTIONS = """You are tasked with producing the following outputs given the inputs {inputs}:
{output_fields}

You work inside a Python REPL environment. Write Python code and it will be executed. You will see the output, then write more code based on what you learned. This is an iterative, interactive process — explore your data, plan your approach, and build up your answer step by step across multiple iterations.

## Workflow Overview

### Iterative loop
- Explore first: inspect sample inputs before extracting anything complicated.
- Plan each block: treat every iteration as a chance to learn, react to results, and refine.
- Persist intermediate work: save state to variables (`all_items`, `records`, `chunks`, etc.) between iterations.

### Think before you build
On your first iteration, **explore before you extract**. Print samples of your input data — check types, lengths, what the data looks like. Understand what you're working with before writing any extraction logic. Even a quick `print(type(records), len(records))` or examining the first item/chunk can save you from going down the wrong path.

After exploring, plan your approach, then execute it in focused steps. Save intermediate results to variables so you can build on them across iterations.

Use `predict()` whenever you need the model to read content and produce structured output — classification, extraction, summarization, clustering, comparison, or perception over text, code, JSON, tables, documents, or images. Use Python for deterministic computation, I/O, formatting, and aggregation.

## Environment & Tools

### Available interfaces

- Variables: {inputs} (your input data).
- `await predict(signature, *, instructions=None, **kwargs)` — your primary analysis tool (async, must await).
  - signature: str with type hints, e.g. `"text: str, question: str -> answer: str"` or `"image: dspy.Image, question: str -> answer: str"`.
  - Use ordinary Python types for text/data inputs (`str`, `list[str]`, `dict`, custom Pydantic models, etc.).
  - For image inputs, use the `dspy.Image` type hint and pass URL or base64 string directly.
  - For fields that may be absent (the caller sometimes has no value to pass), use `Optional[T]` in the signature and pass `None` when missing. Example: `"context: Optional[str], question: str -> answer: str"` lets you call `predict(..., context=None, question=q)` when no context is available. Works for any type: `Optional[int]`, `Optional[dspy.Image]`, `Optional[list[str]]`.
  - instructions: optional str describing the task.
  - Returns a result that supports BOTH `result.answer` (attribute) and `result["answer"]` (subscript). Attribute access works for ALL field names including `items`, `keys`, `values` — no collision issues.
  - **Type contract is enforced on outputs**: if you declare `field: list[X]` (non-Optional) and the LM fails to produce a valid list, `predict()` raises `RuntimeError` with a clear message telling you to simplify the schema or mark the field `Optional`. An empty list `[]` is ALWAYS valid — it means "the LM extracted nothing matching the criteria", which is distinct from "the LM failed to produce a list". Don't speculate "predict returned None" — check what you actually got back: `[]` is valid empty output, `None` is only possible for Optional-declared fields.
  - Capacity: ~400K tokens per call — you can pass substantial data.
- `print()` — ALWAYS print to see results (output is truncated per iteration — see "Managing state & output" below).
- `SUBMIT({final_output_names})` — submit final output when done. The argument names are EXACTLY the output field names from your task's signature (shown above) — use them verbatim. If the task outputs `-> result: SomeType`, call `SUBMIT(result=...)`, NOT `SUBMIT(items=...)` or `SUBMIT(output=...)`.
- Standard libraries: re, json, collections, math, asyncio, etc.

### Execution model
The REPL runs inside an async event loop — use `await` directly, not `asyncio.run()`.

{execution_timeout_instructions}## Managing state & output

Each iteration's printed output is captured and shown to you in subsequent iterations, but **truncated to {max_output_chars:,} characters**. Output beyond that limit is cut off.

**What persists fully:** Python variables. Everything you store in a variable (`all_items`, `records`, `chunks`, etc.) survives intact across successful iterations and cooperative timeouts. If a hard-kill timeout is required, the runtime restores only the pre-timeout pickleable globals snapshot and reports any unpickleable names that were lost.

**What gets truncated:** printed output. If you `print(big_list)` and it's longer than {max_output_chars:,} characters, you'll only see a shortened view in the next iteration's context.

**How to work effectively across iterations:**
- **Store in variables, print summaries:** `items.extend(new_items); print(f"{{len(items)}} items so far")` — the data is in `items` (full), you see the count (tiny).
- **Inspect with lightweight checks:** `print(type(x), len(x), list(x.keys())[:5])` instead of `print(x)`.
- **Slice large output:** `print(str(data)[:2000])` now, `print(str(data)[2000:4000])` next iteration.
- **Never rely on seeing full print output** — if you need the data, it should be in a variable.

## Using `predict()` effectively

### Core usage pattern
`predict()` is async and much faster when run concurrently. For independent calls, always use `asyncio.gather()`:

```python
import asyncio
tasks = [
    predict(
        "text: str -> category: str, confidence: float",
        instructions="Classify this item into the most relevant category.",
        text=item,
    )
    for item in items
]
results = await asyncio.gather(*tasks)
```

Use sequential iteration only when each step depends on previous results (e.g. accumulating context across pages).

### Typed outputs and schemas
Prefer typed outputs over JSON strings:

```python
# Typed fields — attribute access works for ALL field names
result = await predict("text: str -> summary: str, topic: str, sentiment: str", text=excerpt)
print(result.summary, result.topic, result.sentiment)

# Lists
result = await predict("text: str -> keywords: list[str]", text=doc)
print(result.keywords)

# Images: use dspy.Image only when the input is visual
result = await predict("page: dspy.Image -> visible_text: str", page=page_image_url)
print(result.visible_text)
```

For complex/nested structures, define a Pydantic `BaseModel` and reference it by name in the signature string. The type name IS resolved — `predict()` automatically finds your class definition, sends its schema to the LLM for structured output, and returns real model instances inside the result. Use `list[YourModel]`, NOT `list[dict]`:

```python
from pydantic import BaseModel
from typing import Optional

class Person(BaseModel):
    name: str
    role: str
    email: Optional[str] = None

# The class name "Person" in the signature string resolves to the class above.
result = await predict(
    "text: str -> people: list[Person]",
    text=doc_text,
)
for person in result.people:
    print(person.name, person.role, person.email)
```

Important: the model MUST extend `BaseModel` (not `dict`). Only include fields the model can actually produce from the input. Any fields you populate yourself after prediction must have defaults.

### Working with Pydantic values returned by `predict`

When your signature uses a custom type like `list[Person]` or `list[LineItem]`, the values inside the result are **real Pydantic instances** (not dicts):

- **Read fields**: `item.description` (attribute access — natural Pydantic). Do NOT use `item["description"]` — Pydantic instances don't support subscript.
- **Convert to dict** (e.g. for JSON-serializing or spreading): `item.model_dump()`. Do NOT use `**item`, `dict(item)`, or `.dict()` — `**item` fails because a Pydantic instance is not a mapping, and `.dict()` is Pydantic v1 (deprecated).
- **Build a new instance from a dict**: `LineItem(**some_dict)`. Pydantic constructors take keyword arguments, so you must spread a dict, not pass a Pydantic instance.
- **Check type**: `isinstance(item, LineItem)`.
- **Copy with changes**: `item.model_copy(update={{"field": new_value}})`.

Don't add defensive handling like `if isinstance(x, dict): ...else: ...` for fields your signature declared as typed — if the signature says `items: list[LineItem]`, predict will give you LineItem instances (or `None`/`[]`, already coerced).

## Work patterns & examples

### Explore inputs before writing logic
Whatever the task, start by understanding what the runtime handed you. Quick probes prevent wasted iterations:

```repl
print(f"Total inputs: {{len(records)}}")
print(type(records[0]))

orientation = await predict(
    "record: str -> overview: str, notable_elements: list[str]",
    instructions="Give a short description so I know what this input is about and what signals it contains.",
    record=str(records[0]),
)
print(orientation.overview)
print(orientation.notable_elements[:2])
```

Swap `records` for the variables your task provides — the idea is to look before you build.

### Sequential context-building
When each step depends on the last (requirements evolving across drafts, calculations that feed future pages, etc.), surface what you already know and pass it forward:

```python
insights = []
for i, chunk in enumerate(chunks):
    context = "; ".join(insights[-2:]) if insights else "None yet"
    result = await predict(
        "chunk: str, prior: str -> new_insights: list[str]",
        instructions=f"Section {{i+1}} of {{len(chunks)}}. Capture new takeaways that aren't already in prior.",
        chunk=chunk,
        prior=context,
    )
    insights.extend(result.new_insights)
    print(f"Section {{i+1}}: +{{len(result.new_insights)}} (total {{len(insights)}})")
```

This approach works for reasoning chains, progressive audits, or simulations where later iterations depend on earlier conclusions.

### Parallel mapping then synthesis
If inputs are independent, fan out `predict()` calls concurrently, then use Python (or another `predict`) to combine the partial work:

```python
import asyncio
tasks = [
    predict(
        "chunk: str -> signals: list[str], follow_up: Optional[str]",
        instructions="List the most important insights in this chunk and anything that needs deeper review.",
        chunk=chunk,
    )
    for chunk in chunks
]
chunk_results = await asyncio.gather(*tasks)

signals = []
for i, r in enumerate(chunk_results):
    for signal in r.signals:
        signals.append(f"Chunk {{i+1}}: {{signal}}")

summary = await predict(
    "signals: list[str] -> clusters: list[str], action_items: list[str]",
    instructions="Group related signals and surface concrete next steps.",
    signals=signals,
)
print(summary)
```

Adjust the signatures to match your task (classification, QA, calculations, simulations, planning, etc.); the structure—map, collect, synthesize—stays the same.


## Submitting results

**SUBMIT ends execution immediately.** When you call `SUBMIT(...)`, the current turn stops — any `print()` calls or code in the same block will not produce output you can see. This means:
- **Never mix verification and SUBMIT in the same turn.** If you want to inspect or verify results, do that in one turn (print, check values), then call `SUBMIT` in the next turn with nothing else.
- **A SUBMIT turn should contain only the SUBMIT call** (and minimal variable references). No prints, no logic you need to observe.

Once you've verified the results, submit them. `SUBMIT` takes **one argument per output field**, named after each field. Use keyword arguments for clarity, especially when there are multiple outputs:

```python
# For a signature like: ... -> items: list[str], total_count: int, sources: list[int]
SUBMIT(
    items=all_items,
    total_count=len(all_items),
    sources=sorted(set(all_sources)),
)
```

Positional also works, in the order of the output fields: `SUBMIT(all_items, len(all_items), sorted(set(all_sources)))`. Prefer keyword form when there are more than two outputs — it's easier to spot a missing or swapped argument.

**SUBMIT accepts Pydantic instances OR plain dicts OR mixed.** Pass Pydantic objects directly, or nest them inside dicts/lists, whatever's natural:
```python
# All of these work:
SUBMIT(result={{"items": [TaskItem(title="a"), TaskItem(title="b")]}})
SUBMIT(result=ExtractionResult(items=[TaskItem(title="a")]))
SUBMIT(result=[task.model_dump() for task in tasks])
```

Your work is only captured when you call `SUBMIT({final_output_names})`. The REPL loop keeps running until SUBMIT is called — it will NOT stop on its own. If the session ends without a SUBMIT call, nothing is returned and your work is lost. So always end with SUBMIT. Remember: SUBMIT terminates the turn instantly, so verify first in a separate turn, then SUBMIT alone.

**This is NOT a Jupyter notebook.** Writing a variable name alone as the last expression (e.g. just `result` at the end of a block) does NOT submit it — bare expressions evaluate and get discarded. You MUST call `SUBMIT(...)` explicitly. If you've done the work and put it in a variable, the final step is always `SUBMIT(field_name=your_variable)`.

Before submitting, verify your results — if something looks wrong (empty lists, zeros, unexpected values), reconsider your approach. But don't stall: once you're confident, submit.

### If SUBMIT fails validation

SUBMIT's arguments are validated against the output field types. If validation fails, you will see a `[Error]` or `[Type Error]` message in the next iteration's output, e.g.:
- `[Error] Missing output fields: ['sources']. Use SUBMIT(items, total_count, sources)`
- `[Type Error] items: expected list, got str: ...`

When you see these, fix the call and try SUBMIT again — the loop continues until a valid SUBMIT succeeds.
"""


class PredictRLM(dspy.RLM):
    """RLM with DSPy predict tool for structured extraction and reasoning.

    Provides:
    - Built-in predict tool for running DSPy signatures (supports dspy.Image)
    - JSPI-enabled interpreter for proper tool calling in Pyodide/WASM
    - Customized instructions optimized for structured extraction
    - Skills support for domain-specific instructions, packages, and tools
    - Configurable network access for the sandbox

    Use this when you need structured outputs with DSPy signatures, type hints,
    and instructions.

    Example::

        rlm = PredictRLM(
            "images, query -> answer",
            lm="openai/gpt-5.4",
            sub_lm="openai/gpt-5.1",
            max_iterations=10,
        )
        result = rlm(images=["base64_img1", "base64_img2"], query="Compare totals")
        print(result.answer)

    Example with skills::

        from predict_rlm import Skill

        pdf_skill = Skill(
            name="pdf-extraction",
            instructions="Use pdfplumber for table extraction.",
            packages=["pdfplumber"],
        )
        rlm = PredictRLM(
            "documents -> tables: list[dict]",
            lm="openai/gpt-5.4",
            skills=[pdf_skill],
            sub_lm="openai/gpt-5.1",
        )
    """

    # Override reserved tool names - we allow predict but not llm_query
    _RESERVED_TOOL_NAMES = frozenset({"SUBMIT", "print"})

    _context_lm = _RunStateField("_context_lm")
    _partial_history = _RunStateField("_partial_history")
    _partial_pending_entry = _RunStateField("_partial_pending_entry")
    _partial_pending_start = _RunStateField("_partial_pending_start")
    _current_telemetry_context = _RunStateField("_current_telemetry_context")
    _current_predictor_id = _RunStateField("_current_predictor_id")
    _trace_export_context = _RunStateField("_trace_export_context")
    _pending_submit_confirmation = _RunStateField("_pending_submit_confirmation")
    _last_action_lm_usage = _RunStateField("_last_action_lm_usage")
    _last_action_lm_metadata = _RunStateField("_last_action_lm_metadata")

    def __init__(
        self,
        signature: type[Signature] | str,
        lm: dspy.LM | str | None = None,
        sub_lm: dspy.LM | str | None = None,
        max_iterations: int = 30,
        max_llm_calls: int = 50,
        max_output_chars: int = 50_000,
        verbose: bool = True,
        tools: dict[str, Callable[..., str]] | list[Callable] | None = None,
        interpreter: CodeInterpreter | None = None,
        sandbox_backend: BackendName | str | None = None,
        sbx_config: SbxConfig | None = None,
        sbx_pool: SbxPool | None = None,
        allowed_domains: list[str] | None = None,
        skills: list[Skill] | None = None,
        debug: bool = False,
        output_dir: str | Path | None = None,
        telemetry_context: TelemetryContext | None = None,
        submit_confirmation: Callable[[SubmitConfirmationContext], str | None] | None = None,
        trace_export_path: str | Path | None = None,
        runtime_hooks: list[RuntimeHook] | None = None,
        on_runtime_hook_event: Callable[[RuntimeHookEvent], Any] | None = None,
        model_execution_timeout: bool = False,
        *,
        adapters: Sequence[InputAdapter[Any] | OutputAdapter[Any]] = (),
        execution: ExecutionBackend | None = None,
        modules: tuple[RuntimeModule | RuntimeContribution, ...]
        | list[RuntimeModule | RuntimeContribution] = (),
        events: Sequence[EventSink] = (),
    ):
        """
        Args:
            signature: Defines inputs and outputs. String like "images, query -> answer"
                      or a Signature class.
            lm: Main LM that drives the RLM (writes and executes code). Can be a
               dspy.LM instance or a model string like "openai/gpt-4o". If None,
               uses the current context LM (from dspy.settings.lm or dspy.context).
            sub_lm: LM for the predict tool. Can be a dspy.LM instance or a model
                   string like "anthropic/claude-haiku-4-5". If None, uses the
                   current context LM (from dspy.settings.lm or dspy.context).
            max_iterations: Maximum REPL interaction iterations.
            max_llm_calls: Maximum LM calls per execution.
            max_output_chars: Maximum characters to include from REPL output.
            verbose: Whether to log compact per-iteration progress.
                  Defaults to True; pass False for quiet execution.
            tools: Additional tool functions callable from interpreter code.
                  Accepts a dict mapping names to callables, or a list of
                  callables (names inferred from __name__).
                  predict is added automatically if not already provided.
            interpreter: CodeInterpreter-compatible execution backend to use.
                        Defaults to JSPI-enabled JspiBackend.
            sandbox_backend: Named sandbox backend to create when interpreter
                            is not provided. Defaults to "jspi"; pass "sbx"
                            to opt into Docker Sandboxes.
            sbx_config: Docker Sandboxes backend configuration.
            sbx_pool: Prewarmed Docker Sandboxes backend pool. Requires
                      ``sandbox_backend="sbx"`` and owns its backend config.
            allowed_domains: Domains/IPs the sandbox can access via network.
                            By default, no network access is allowed.
                            Example: ["api.example.com", "192.168.1.100:8080"]
            skills: List of Skill instances providing domain-specific instructions,
                   PyPI packages, and tools. Skills are merged automatically —
                   instructions are appended to the prompt, packages are installed
                   in the sandbox, and tools are exposed alongside predict().
            debug: If True, enable interpreter debug output and lifecycle
                  diagnostics. Debug does not imply verbose RLM trace output.
            output_dir: Constructor convenience that collects generated File and
                       list[File] outputs under ``<output_dir>/<field>/``. It does
                       not redirect scalar or JSON outputs, logs, or traces. If
                       None, generated files are collected in a temp directory.
            telemetry_context: Optional run/case telemetry context for
                       OTel-shaped local JSONL events. Disabled/failed
                       telemetry writes are always ignored.
            submit_confirmation: Optional callback invoked after a valid
                       SUBMIT. It receives SubmitConfirmationContext and may
                       return a message to feed back as the next observation.
                       Returning None or "" preserves normal immediate submit.
            trace_export_path: Optional path for best-effort atomic RunTrace
                       snapshots while the run is in progress. The file is
                       replaced in place and trace export failures are ignored.
            model_execution_timeout: Whether the action LM may set a per-iteration
                       ``execution_timeout_seconds`` to soft-cap (recoverably
                       interrupt) its own code blocks. Off by default — models
                       tend to under-set it and kill legitimately slow work (e.g.
                       30s-2min predict() calls). When off, the host watchdog
                       (``SbxConfig.exec_timeout`` / JSPI ``exec_timeout``, default
                       300s) remains as the hang guard. Set True to restore the
                       model-chosen timeout knob.
            adapters: Additional input and output adapters contributed to the runtime.
            execution: Small-kernel execution backend. Mutually exclusive with
                       interpreter and sandbox backend options.
            modules: Runtime contribution factories or resolved contributions.
            events: Runtime event sinks.
        """
        if execution is not None and any(
            value is not None
            for value in (interpreter, sandbox_backend, sbx_config, sbx_pool)
        ):
            raise ValueError(
                "Pass execution or interpreter/sandbox_backend/sbx_config/sbx_pool "
                "options, not both."
            )
        if interpreter is not None and sbx_pool is not None:
            raise ValueError(
                "Pass either interpreter or sbx_pool, not both. "
                "A custom interpreter is already a complete sandbox backend."
            )
        if interpreter is not None and sandbox_backend is not None:
            raise ValueError(
                "Pass either interpreter or sandbox_backend, not both. "
                "A custom interpreter is already a complete sandbox backend."
            )
        self._sandbox_backend = BackendName(sandbox_backend or BackendName.JSPI)
        if sbx_pool is not None and self._sandbox_backend is not BackendName.SBX:
            raise ValueError("sbx_pool requires sandbox_backend='sbx'.")
        if sbx_pool is not None and sbx_config is not None:
            raise ValueError("Pass sbx_config to SbxPool when using sbx_pool.")
        runtime_hooks = list(runtime_hooks or [])
        if (
            runtime_hooks
            and interpreter is None
            and sbx_pool is None
            and self._sandbox_backend is not BackendName.SBX
        ):
            raise ValueError("runtime_hooks require the SBX backend in PredictRLM v1")
        self._runtime_hooks = runtime_hooks
        self._on_runtime_hook_event = on_runtime_hook_event
        self._model_execution_timeout = model_execution_timeout
        self._sbx_pool = sbx_pool
        self._sbx_config = sbx_config

        # Store main LM. ``dspy.LM.copy()`` gives a fresh history so
        # concurrent PredictRLM instances don't share mutable state —
        # without this, running N RLMs in parallel (asyncio.gather) would
        # make each call's usage_since delta include every other
        # concurrent call's history entries. Cache / callbacks / config
        # stay shared by reference; only the ``history`` list is reset.
        if lm is None:
            self._lm = None
        elif isinstance(lm, str):
            self._lm = dspy.LM(lm, cache=False)
        else:
            self._lm = lm.copy() if hasattr(lm, "copy") else lm

        if sub_lm is None:
            self._sub_lm = None
        elif isinstance(sub_lm, str):
            self._sub_lm = dspy.LM(sub_lm, cache=False)
        else:
            self._sub_lm = sub_lm.copy() if hasattr(sub_lm, "copy") else sub_lm

        # Will be set during forward() to capture context LM for thread-safe predict calls
        self._context_lm = None

        # Partial trajectory capture. Updated per-iteration during execution
        # so a caller that catches a timeout or exception can still recover
        # whatever REPL steps completed before the failure. Two slots:
        #   _partial_history: REPLHistory of fully-executed steps
        #   _partial_pending_entry: reasoning+code from the CURRENT iteration
        #       captured before execution, so a crash mid-execution still
        #       surfaces what the agent was trying to do.
        self._partial_history = None
        self._partial_pending_entry = None
        self._partial_pending_start = None

        # Store allowed_domains, debug, and output_dir for interpreter creation
        self._allowed_domains = allowed_domains
        self._debug = debug
        configure_predict_rlm_logging(
            debug=True if debug else None,
            verbose=True if verbose else None,
        )
        self._output_dir = str(output_dir) if output_dir else None
        self._telemetry_context = telemetry_context
        self._current_telemetry_context: TelemetryContext | None = None
        self._current_predictor_id: str | None = None
        self._submit_confirmation = submit_confirmation
        self._trace_export_path = Path(trace_export_path) if trace_export_path else None
        self._trace_export_context: _TraceExportContext | None = None

        # Merge skills into instructions, packages, modules, and tools
        self._skill_instructions = ""
        self._skill_packages: list[str] = []
        self._skill_modules: dict[str, str] = {}
        if skills:
            skill_instructions, skill_packages, skill_modules, skill_tools = merge_skills(
                skills
            )
            self._skill_instructions = skill_instructions
            self._skill_packages = skill_packages
            self._skill_modules = skill_modules
        else:
            skill_tools = {}

        # Normalize tools to dict[str, Callable] — accept both dict and list forms
        if isinstance(tools, list):
            merged_tools = {fn.__name__: fn for fn in tools}
        else:
            merged_tools = dict(tools or {})

        # Merge skill tools (skill_tools conflicts already checked by merge_skills)
        for name, fn in skill_tools.items():
            if name in merged_tools:
                raise ValueError(
                    f"Tool name conflict: '{name}' is provided by both a skill and "
                    f"the tools parameter"
                )
            merged_tools[name] = fn

        if "predict" not in merged_tools:
            merged_tools["predict"] = self._create_predict_tool()

        # Convert dict → list of dspy.Tool for dspy 3.1.3+ which expects list[Callable | Tool]
        tools_list = [dspy.Tool(func, name=name) for name, func in merged_tools.items()]

        # Call parent __init__ with modified tools
        # Note: we pass sub_lm=None since we don't want the default llm_query tools
        super().__init__(
            signature=signature,
            max_iterations=max_iterations,
            max_llm_calls=max_llm_calls,
            max_output_chars=max_output_chars,
            verbose=verbose,
            tools=tools_list,
            sub_lm=None,  # Disable default llm_query tools
            interpreter=interpreter,
        )

        from .in_context import _in_context_field_names

        _in_context_field_names(self.signature)
        expanded_modules = tuple(
            module() if callable(module) else module for module in modules
        )
        if not all(isinstance(module, RuntimeContribution) for module in expanded_modules):
            raise TypeError("Runtime modules must return RuntimeContribution")
        module_selects_execution = any(
            module.execution is not None for module in expanded_modules
        )
        execution_options_selected = any(
            value is not None
            for value in (interpreter, sandbox_backend, sbx_config, sbx_pool)
        )

        if execution is None and not (
            module_selects_execution and not execution_options_selected
        ):
            resolved_execution = execution_from_options(
                owner=self,
                interpreter=interpreter,
                sandbox_backend=self._sandbox_backend,
                sbx_config=self._sbx_config,
                sbx_pool=self._sbx_pool,
                allowed_domains=self._allowed_domains,
                runtime_hooks=self._runtime_hooks,
                on_runtime_hook_event=self._on_runtime_hook_event,
            )
        elif execution is not None:
            resolved_execution = ExistingExecutionBackendAdapter(execution)
        else:
            resolved_execution = None
        runtime_tools = tuple(
            CallableTool(
                name=name,
                function=tool.func,
                description=str(getattr(tool, "desc", "") or ""),
                schema=getattr(tool, "args", {}) or {},
            )
            for name, tool in self._user_tools.items()
        )
        configured_adapters = (
            *tuple(adapters),
            *(adapter for module in expanded_modules for adapter in module.adapters),
        )
        configured_input_names = {
            adapter.name
            for adapter in configured_adapters
            if isinstance(adapter, InputAdapter)
        }
        configured_output_names = {
            adapter.name
            for adapter in configured_adapters
            if isinstance(adapter, OutputAdapter)
        }
        configured_tool_operation_names = {
            operation.name
            for module in expanded_modules
            for operation in module.tool_operations
        }
        file_compatibility_contribution = file_compatibility(
            output_dir=self._output_dir
        )
        compatibility_inputs = tuple(
            adapter
            for adapter in (ValueInputAdapter(), *file_compatibility_contribution.adapters)
            if isinstance(adapter, InputAdapter)
            if adapter.name not in configured_input_names
        )
        compatibility_outputs = tuple(
            adapter
            for adapter in file_compatibility_contribution.adapters
            if isinstance(adapter, OutputAdapter)
            if adapter.name not in configured_output_names
        )
        compatibility_tool_operations = (
            ()
            if SyncedFileToolOperation.name in configured_tool_operation_names
            else (SyncedFileToolOperation(),)
        )
        runtime_spec = resolve_runtime_spec(
            direct=RuntimeContribution(
                instructions=(self._skill_instructions,) if self._skill_instructions else (),
                adapters=(
                    *compatibility_inputs,
                    *compatibility_outputs,
                    *tuple(adapters),
                ),
                tools=runtime_tools,
                packages=tuple(self._skill_packages),
                execution=resolved_execution,
                events=tuple(events),
                validators=file_compatibility_contribution.validators,
                tool_operations=compatibility_tool_operations,
            ),
            modules=expanded_modules,
        )
        if not callable(
            getattr(runtime_spec.execution, "validate_host_directory_mounts", None)
        ):
            runtime_spec = replace(
                runtime_spec,
                execution=ExistingExecutionBackendAdapter(runtime_spec.execution),
            )
        self._runtime_spec = runtime_spec
        for validator in self._runtime_spec.validators:
            validator(self.signature)
        for field_name, field_info in self.signature.output_fields.items():
            field = FieldDescriptor(field_name, field_info.annotation)
            matches = [
                adapter
                for adapter in self._runtime_spec.output_adapters
                if adapter.supports(field)
            ]
            if len(matches) > 1:
                resolve_output_adapter(matches, field)
        action_signature, extract_signature = self._build_signatures()
        self.generate_action = dspy.Predict(action_signature)
        self.extract = dspy.Predict(extract_signature)

    @property
    def runtime_spec(self) -> RuntimeSpec:
        return self._runtime_spec

    def _acquire_runtime_interpreter(
        self,
        spec: ExecutionSpec,
        ctx: RunContext,
    ):
        return self._interpreter_context(execution_tools=dict(spec.tools))

    def _new_run_context(self, input_values: dict[str, Any]) -> RunContext:
        ctx = RunContext(self.runtime_spec, input_values)
        ctx.state["lm"] = (
            self._lm.copy() if isinstance(self._lm, dspy.LM) else self._lm
        )
        ctx.state["sub_lm"] = (
            self._sub_lm.copy() if isinstance(self._sub_lm, dspy.LM) else self._sub_lm
        )
        return ctx

    def _active_sub_lm(self) -> Any:
        ctx = current_run_context()
        return ctx.state.get("sub_lm") if ctx is not None else self._sub_lm

    def _evidence(self) -> EvidenceRecorder | None:
        ctx = current_run_context()
        if ctx is None:
            return None
        recorder = ctx.state.get("evidence")
        return recorder if isinstance(recorder, EvidenceRecorder) else None

    def _runtime_packages(self) -> tuple[str, ...]:
        spec = getattr(self, "_runtime_spec", None)
        return spec.packages if spec is not None else tuple(self._skill_packages)

    def _runtime_tool_functions(self) -> dict[str, Callable[..., Any]]:
        spec = getattr(self, "_runtime_spec", None)
        if spec is None:
            return {name: tool.func for name, tool in self._user_tools.items()}
        return {
            tool.name: tool.function if isinstance(tool, CallableTool) else tool
            for tool in spec.tools
        }

    def _runtime_instructions(self) -> str:
        spec = getattr(self, "_runtime_spec", None)
        if spec is None:
            return self._skill_instructions
        return "\n\n".join(spec.instructions)

    def _create_predict_tool(self) -> Callable[..., dict[str, Any]]:
        """Create the predict tool for running DSPy signatures."""
        # Capture self to access _sub_lm at call time
        rlm_instance = self

        def _unwrap_optional(anno: Any) -> Any:
            """Strip Optional[X] / Union[X, None] / X | None → X."""
            import typing as _typing

            if _typing.get_origin(anno) is _typing.Union:
                non_none = [a for a in _typing.get_args(anno) if a is not type(None)]
                if len(non_none) == 1:
                    return non_none[0]
            return anno

        def _image_field_info(anno: Any) -> tuple[bool, bool]:
            """Returns (is_image, is_list) for an annotation.

            Recognizes dspy.Image inside any Optional/Union/list wrapper by
            unwrapping and inspecting origins via typing.get_origin.
            """
            import typing as _typing

            anno = _unwrap_optional(anno)
            if _typing.get_origin(anno) is list:
                args = _typing.get_args(anno)
                if args:
                    inner = _unwrap_optional(args[0])
                    return (inner is _ImageType, True)
                return (False, False)
            return (anno is _ImageType, False)

        def _is_list_output(anno: Any) -> bool:
            """True if anno is list[...] or Optional[list[...]]."""
            import typing as _typing

            return _typing.get_origin(_unwrap_optional(anno)) is list

        def _allows_none(anno: Any) -> bool:
            """True if the annotation allows None (Optional[T], T | None, etc.)."""
            return _unwrap_optional(anno) is not anno

        def _wrap_images_from_sig(sig_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
            """Wrap URL/base64 strings as dspy.Image for Image-typed input fields."""
            wrapped: dict[str, Any] = {}
            for key, value in kwargs.items():
                if key not in sig_obj.input_fields or value is None:
                    wrapped[key] = value
                    continue
                is_image, is_list = _image_field_info(sig_obj.input_fields[key].annotation)
                if is_image and is_list and isinstance(value, list):
                    wrapped[key] = [
                        dspy.Image(url=item) if isinstance(item, str) else item
                        for item in value
                    ]
                elif is_image and not is_list and isinstance(value, str):
                    wrapped[key] = dspy.Image(url=value)
                else:
                    wrapped[key] = value
            return wrapped

        async def predict(
            signature: str,
            *,
            instructions: str | None = None,
            pydantic_schemas: dict[str, dict] | None = None,
            **kwargs: Any,
        ) -> dict[str, Any]:
            """Run a DSPy prediction with the given signature and inputs.

            Args:
                signature: DSPy signature string with optional type hints.
                          Use "field: dspy.Image" to mark image inputs.
                          Example: "image: dspy.Image, question -> answer"
                instructions: Optional instructions for the LM describing the task.
                             Example: "Mark as toxic if the comment includes insults."
                pydantic_schemas: Optional dict mapping type names to JSON schemas.
                                 Used to reconstruct custom Pydantic types for DSPy.
                                 Automatically provided by sandbox when custom types
                                 are used in the signature.
                **kwargs: Input values matching the signature's input fields.
                         For dspy.Image typed fields, pass URL/base64 strings directly.

            Returns:
                Dict with output field names as keys and predicted values.

            Examples:
                # Simple text query
                result = await predict("question -> answer", question="What is 2+2?")
                print(result["answer"])

                # Image analysis (use dspy.Image type hint)
                result = await predict(
                    "image: dspy.Image, question -> answer",
                    image=img_url,  # Auto-wrapped in dspy.Image
                    question="What text is visible?"
                )
                print(result["answer"])

                # Custom Pydantic types (schemas extracted automatically)
                result = await predict(
                    "page: dspy.Image -> tasks: list[TaskItem]",
                    page=page_url,
                )
                print(result["tasks"])  # List of TaskItem dicts
            """
            recorder = rlm_instance._evidence()
            predict_call_id = uuid.uuid4().hex
            # Priority: sub_lm > captured context LM > current settings.lm
            lm = (
                rlm_instance._active_sub_lm()
                or rlm_instance._context_lm
                or dspy.settings.lm
            )
            if lm is None:
                raise RuntimeError(
                    "No LM available for predict. Either pass sub_lm to PredictRLM "
                    "or set a default LM with dspy.configure(lm=...)"
                )

            # Normalize signature for dspy.Signature parser: collapse
            # multi-line signatures to a single line.
            signature = " ".join(signature.split())

            # Reconstruct custom Pydantic types from schemas if provided
            custom_types: dict[str, type] = {}
            if pydantic_schemas:
                for name, schema in pydantic_schemas.items():
                    try:
                        # Ensure schema has a title matching the key so the
                        # reconstructed model gets the right name (hand-crafted
                        # schemas from the LM often omit "title")
                        if "title" not in schema:
                            schema = {**schema, "title": name}
                        built = _models_from_schema(schema)
                        custom_types.update(built)
                    except Exception as e:
                        logger.warning(
                            f"Failed to reconstruct Pydantic model '{name}' from schema: {e}"
                        )
            else:
                # Check if signature has custom types (capitalized identifiers)
                pattern = r"(?<![.\w])([A-Z][A-Za-z0-9_]*)"
                matches = re.findall(pattern, signature)
                builtins = {
                    "Image",
                    "List",
                    "Optional",
                    "Dict",
                    "Any",
                    "Union",
                    "Literal",
                    "Tuple",
                    "Set",
                    "BaseModel",
                }
                custom_in_sig = [m for m in matches if m not in builtins]
                if custom_in_sig:
                    logger.debug(
                        f"predict() called with custom types {custom_in_sig} in signature "
                        f"but no pydantic_schemas provided. This may cause parsing issues. "
                        f"If using asyncio.gather(), ensure Pydantic models are defined in "
                        f"the REPL's global scope (not inside a function)."
                    )

            # Create signature with optional instructions and custom types
            try:
                if custom_types:
                    sig = dspy.Signature(
                        signature, instructions or "", custom_types=custom_types
                    )
                elif instructions:
                    sig = dspy.Signature(signature, instructions)
                else:
                    sig = dspy.Signature(signature, "")
            except Exception as e:
                if "Unknown name" in str(e):
                    pattern = r"(?<![.\w])([A-Z][A-Za-z0-9_]*)"
                    custom_types_in_sig = re.findall(pattern, signature)
                    builtins = {
                        "Image",
                        "List",
                        "Optional",
                        "Dict",
                        "Any",
                        "Union",
                        "Literal",
                        "Tuple",
                        "Set",
                        "BaseModel",
                    }
                    custom_names = [m for m in custom_types_in_sig if m not in builtins]

                    if custom_names:
                        # Unresolved types — fall back to string signature instead of crashing.
                        # This commonly happens when models are defined inside functions or
                        # the schema extractor couldn't find them via frame introspection.
                        logger.warning(
                            f"Failed to resolve custom types {custom_names} in signature. "
                            f"Falling back to string signature. "
                            f"Error: {e}"
                        )

                logger.warning(
                    f"Failed to create Signature with parsed types: {e}. Falling back to string signature."
                )
                sig = signature  # Use string signature directly

            # Derive image-wrapping and list-output metadata from the parsed
            # signature's annotations (DSPy's own parser is authoritative).
            # If sig is a string (fallback path), skip these features.
            if isinstance(sig, str):
                wrapped_kwargs = kwargs
                output_field_annotations: dict[str, Any] = {}
            else:
                wrapped_kwargs = _wrap_images_from_sig(sig, kwargs)
                output_field_annotations = {
                    name: field.annotation for name, field in sig.output_fields.items()
                }

            # Create predictor and run with the specified LM (async, no threads)
            predictor = dspy.Predict(sig)
            _call_start = time.perf_counter()
            _hist_len_before = snapshot_lm_history_len(lm)
            result: dict[str, Any] = {}
            if recorder is not None:
                await recorder.emit(
                    RunEventKind.PREDICT_STARTED,
                    call_id=predict_call_id,
                    signature=signature,
                    instructions=instructions,
                    model=str(getattr(lm, "model", lm)),
                    input=kwargs,
                )
            try:
                with dspy.context(lm=lm):
                    prediction = await predictor.acall(**wrapped_kwargs)

                # Convert Prediction to dict, serializing any Pydantic models to dicts
                def _to_serializable(value: Any) -> Any:
                    """Convert Pydantic models and other objects to JSON-serializable dicts."""
                    if value is None or isinstance(value, (str, int, float, bool)):
                        return value

                    from pydantic import BaseModel as PydanticBaseModel

                    if isinstance(value, PydanticBaseModel):
                        return value.model_dump(mode="python")

                    if hasattr(value, "dict") and hasattr(value, "__fields__"):
                        return value.dict()

                    if hasattr(value, "__dataclass_fields__"):
                        import dataclasses

                        return {
                            k: _to_serializable(v) for k, v in dataclasses.asdict(value).items()
                        }

                    if isinstance(value, list):
                        return [_to_serializable(item) for item in value]

                    if isinstance(value, tuple):
                        return [_to_serializable(item) for item in value]

                    if isinstance(value, dict):
                        return {k: _to_serializable(v) for k, v in value.items()}

                    if isinstance(value, set):
                        return list(value)

                    if hasattr(value, "__dict__") and type(value).__name__ == "Prediction":
                        return {
                            k: _to_serializable(v)
                            for k, v in value.__dict__.items()
                            if not k.startswith("_")
                        }

                    if hasattr(value, "__dict__"):
                        try:
                            return {
                                k: _to_serializable(v)
                                for k, v in value.__dict__.items()
                                if not k.startswith("_") and not callable(v)
                            }
                        except Exception:
                            pass

                    return str(value)

                # Use prediction[field] for extracting values from DSPy Prediction.
                # Enforce type contract: if the LM returned None for a field
                # whose declared type does NOT allow None (e.g. list[X] not
                # Optional[list[X]]), raise loudly. Silently coercing None→[] or
                # passing None through hides LM failures behind ambiguous empty
                # values and causes models to speculate "predict returned None".
                for field in prediction.keys():
                    if field.startswith("_"):
                        continue
                    value = prediction[field]
                    if value is None and field in output_field_annotations:
                        anno = output_field_annotations[field]
                        if not _allows_none(anno):
                            raise RuntimeError(
                                f"predict: LM returned None for non-Optional output "
                                f"field {field!r} (declared type: {anno}). The LM "
                                f"couldn't produce a valid value. Schema may be too "
                                f"complex — try simplifying the signature, or mark "
                                f"the field Optional (e.g. Optional[list[X]]) if None "
                                f"is acceptable."
                            )
                    result[field] = _to_serializable(value)
            except BaseException as exc:
                record_predict_call(
                    _RawPredictCall(
                        signature=signature,
                        instructions=instructions,
                        model=str(getattr(lm, "model", lm)),
                        duration_ms=ms_since(_call_start),
                        usage=usage_since(lm, _hist_len_before),
                        input=kwargs,
                        output=result,
                        error=str(exc),
                        lm=lm_finish_since(lm, _hist_len_before),
                    )
                )
                if recorder is not None:
                    try:
                        await recorder.emit(
                            RunEventKind.PREDICT_FINISHED,
                            call_id=predict_call_id,
                            error_type=type(exc).__name__,
                            error=str(exc),
                            cancelled=isinstance(exc, asyncio.CancelledError),
                            output=result,
                        )
                    except BaseException as evidence_error:
                        setattr(exc, "evidence_error", evidence_error)
                raise
            else:
                record_predict_call(
                    _RawPredictCall(
                        signature=signature,
                        instructions=instructions,
                        model=str(getattr(lm, "model", lm)),
                        duration_ms=ms_since(_call_start),
                        usage=usage_since(lm, _hist_len_before),
                        input=kwargs,
                        output=result,
                        lm=lm_finish_since(lm, _hist_len_before),
                    )
                )
                if recorder is not None:
                    await recorder.emit(
                        RunEventKind.PREDICT_FINISHED,
                        call_id=predict_call_id,
                        output=result,
                    )
                return result

        return predict

    def _format_tool_docs(self, tools: dict[str, Callable]) -> str:
        """Format tools with full docstrings for inclusion in instructions."""
        return format_tool_docs_full(tools)

    def _get_output_fields_info(self) -> list[dict]:
        """Get output field info for sandbox SUBMIT registration.

        Overrides the base RLM method to replace File-typed output fields
        with ``str`` so the SUBMIT wrapper gets a type hint like
        ``SUBMIT(updated_workbook: str, result: dict)`` instead of omitting
        the type (which confuses the RLM into passing dicts instead of paths).
        """
        from dspy.primitives.code_interpreter import SIMPLE_TYPES

        fields = []
        for name, field in self.signature.output_fields.items():
            annotation = getattr(field, "annotation", str)
            descriptor = FieldDescriptor(name, annotation)
            field_info = {"name": name}
            if descriptor.matches(File):
                field_info["type"] = "list" if descriptor.is_list else "str"
            elif annotation in SIMPLE_TYPES:
                field_info["type"] = annotation.__name__
            default = _get_field_default(field)
            if default is not _NO_FIELD_DEFAULT:
                field_info["has_default"] = True
                field_info["default"] = _to_json_safe(default)
            fields.append(field_info)
        return fields

    def _process_final_output(
        self,
        result: Any,
        output_field_names: list[str],
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Validate FinalOutput, coercing string paths to File for file fields.

        The RLM submits plain path strings for File-typed output fields (as
        instructed). Pydantic's ``File`` model requires a dict ``{"path": ...}``
        or a ``File`` instance, so we wrap bare strings before the parent's
        ``parse_value`` runs.
        """
        raw = result.output
        if isinstance(raw, dict):
            for name in output_field_names:
                if name not in raw:
                    continue
                field = self.signature.output_fields.get(name)
                if field is None:
                    continue
                descriptor = FieldDescriptor(name, field.annotation)
                if not descriptor.matches(File):
                    continue
                val = raw[name]
                if descriptor.is_list:
                    if isinstance(val, list):
                        raw[name] = [{"path": v} if isinstance(v, str) else v for v in val]
                elif isinstance(val, str):
                    raw[name] = {"path": val}

        return super()._process_final_output(result, output_field_names)

    @contextmanager
    def _interpreter_context(
        self,
        execution_tools: dict[str, Callable],
        file_plan: dict[str, Any] | None = None,
    ) -> Iterator[LegacyExecutionBackend]:
        """Yield interpreter, creating the configured backend if none provided."""
        if self._interpreter is not None:
            if self._runtime_hooks:
                configure_runtime = self._declared_callable(
                    self._interpreter, "configure_runtime"
                )
                runtime_kwargs: dict[str, Any] = {}
                if configure_runtime is not None:
                    runtime_kwargs = self._accepted_kwargs(
                        configure_runtime,
                        tools=execution_tools,
                        output_fields=self._get_output_fields_info(),
                        runtime_hooks=self._runtime_hooks,
                        on_runtime_hook_event=self._on_runtime_hook_event,
                    )
                if configure_runtime is None or "runtime_hooks" not in runtime_kwargs:
                    raise ValueError("runtime_hooks require an SBX-compatible interpreter")
                configure_runtime(**runtime_kwargs)
            else:
                self._inject_execution_context(self._interpreter, execution_tools)
            skill_packages_configured = self._ensure_interpreter_skill_packages(
                self._interpreter
            )
            backend = self._interpreter_backend_label(self._interpreter)
            configured = self._configure_interpreter_debug(self._interpreter)
            self._log_lifecycle(
                "interpreter.injected",
                backend=backend,
                interpreter=type(self._interpreter).__name__,
                debug_configured=configured["debug"],
                verbose_configured=configured["verbose"],
                skill_packages_configured=skill_packages_configured,
                owner="caller",
            )
            try:
                yield self._interpreter
            finally:
                self._log_lifecycle(
                    "interpreter.released",
                    backend=backend,
                    interpreter=type(self._interpreter).__name__,
                    owner="caller",
                )
        else:
            extra_read = list(file_plan["read_paths"]) if file_plan else []
            extra_write = (
                [file_plan["write_dir"]] if file_plan and file_plan["write_dir"] else None
            )

            # Add module host paths to read permissions
            for mod_path in self._skill_modules.values():
                extra_read.append(mod_path)

            interpreter_kwargs = {
                "tools": execution_tools,
                "output_fields": self._get_output_fields_info(),
                "allowed_domains": self._allowed_domains,
                "skill_packages": list(self._runtime_packages()) or None,
                "debug": self._debug,
                "verbose": self._iteration_logging_enabled(),
                "extra_read_paths": extra_read or None,
                "extra_write_paths": extra_write,
            }
            if self._sbx_pool is not None:
                self._log_lifecycle(
                    "sbx.pool.lease.request",
                    backend="sbx",
                    tools=len(execution_tools),
                    output_fields=len(self._get_output_fields_info()),
                )
                with self._sbx_pool.lease(
                    tools=execution_tools,
                    output_fields=self._get_output_fields_info(),
                    skill_packages=list(self._runtime_packages()) or None,
                    debug=self._debug,
                    verbose=self._iteration_logging_enabled(),
                    runtime_hooks=self._runtime_hooks,
                    on_runtime_hook_event=self._on_runtime_hook_event,
                ) as repl:
                    self._log_lifecycle(
                        "sbx.pool.lease.acquired",
                        backend="sbx",
                        interpreter=type(repl).__name__,
                    )
                    try:
                        yield repl
                    finally:
                        self._log_lifecycle(
                            "sbx.pool.lease.released",
                            backend="sbx",
                            interpreter=type(repl).__name__,
                        )
                return
            if self._sandbox_backend is BackendName.SBX:
                from .backends.sbx import SbxBackend, SbxConfig

                self._log_lifecycle(
                    "interpreter.create",
                    backend="sbx",
                    tools=len(execution_tools),
                    output_fields=len(self._get_output_fields_info()),
                    allowed_domains=len(self._allowed_domains or []),
                    skill_packages=len(self._runtime_packages()),
                )
                repl = SbxBackend(
                    config=self._sbx_config or SbxConfig(),
                    runtime_hooks=self._runtime_hooks,
                    on_runtime_hook_event=self._on_runtime_hook_event,
                    **interpreter_kwargs,
                )
            else:
                self._log_lifecycle(
                    "interpreter.create",
                    backend="jspi",
                    tools=len(execution_tools),
                    output_fields=len(self._get_output_fields_info()),
                    allowed_domains=len(self._allowed_domains or []),
                    skill_packages=len(self._runtime_packages()),
                )
                repl = JspiBackend(
                    **interpreter_kwargs,
                    telemetry_context=self._current_telemetry_context,
                )
            try:
                yield repl
            finally:
                self._log_lifecycle(
                    "interpreter.shutdown",
                    backend=self._interpreter_backend_label(repl),
                    interpreter=type(repl).__name__,
                    owner="predict_rlm",
                )
                repl.shutdown()

    @asynccontextmanager
    async def _execution_session(
        self,
        execution_tools: dict[str, Callable[..., Any]],
        file_plan: dict[str, Any] | None = None,
    ) -> AsyncIterator[Any]:
        ctx = current_run_context()
        if ctx is None:
            kwargs = {"execution_tools": execution_tools}
            if file_plan is not None:
                kwargs["file_plan"] = file_plan
            with self._interpreter_context(**kwargs) as repl:
                yield repl
            return
        await self._open_runtime_inputs(ctx)
        try:
            validate_sandbox_root_reservations(ctx.input_bindings)
            requirements = merge_session_requirements(
                (
                    SessionRequirements(
                        allowed_domains=tuple(self._allowed_domains or ()),
                        extra_read_paths=tuple(self._skill_modules.values()),
                    ),
                    *(
                        binding.prepared.requirements
                        for binding in ctx.input_bindings.values()
                    ),
                    *ctx.output_requirements.values(),
                    *(
                        SessionRequirements(
                            extra_read_paths=tuple(
                                getattr(tool, "extra_read_paths", ())
                            ),
                            extra_write_paths=tuple(
                                getattr(tool, "extra_write_paths", ())
                            ),
                        )
                        for tool in self.runtime_spec.tools
                    ),
                )
            )
            execution_spec = ExecutionSpec(
                tools=execution_tools,
                output_fields=tuple(self._get_output_fields_info()),
                packages=tuple(self.runtime_spec.packages),
                allowed_domains=requirements.allowed_domains,
                extra_read_paths=requirements.extra_read_paths,
                extra_write_paths=requirements.extra_write_paths,
                host_directory_mounts=tuple(
                    mount
                    for binding in ctx.input_bindings.values()
                    for mount in binding.prepared.host_directory_mounts
                ),
                sandbox_roots=tuple(
                    root
                    for binding in ctx.input_bindings.values()
                    for root in binding.prepared.sandbox_roots
                ),
                debug=self._debug,
                verbose=self._iteration_logging_enabled(),
                metadata={"has_file_plan": file_plan is not None},
            )
        except BaseException as exc:
            try:
                await self._finalize_runtime_inputs(ctx, None, exc)
            except BaseException as finalize_error:
                setattr(exc, "input_adapter_finalize_error", finalize_error)
                await self._record_pre_session_finalize_failure(finalize_error)
            raise
        recorder = self._evidence()
        session_name: str | None = None
        primary: BaseException | None = None
        release_succeeded = False
        try:
            await self.runtime_spec.execution.validate_host_directory_mounts(
                execution_spec.host_directory_mounts,
                ctx,
            )
            async with self.runtime_spec.execution.start(execution_spec, ctx) as session:
                session_name = session.name
                ctx.session = session
                ctx.ownership = session.ownership
                try:
                    if recorder is not None:
                        await recorder.emit(
                            RunEventKind.SESSION_STARTED,
                            backend=session.name,
                            ownership=getattr(session.ownership, "value", session.ownership),
                        )
                    await session.install_packages(self.runtime_spec.packages)
                    if recorder is not None:
                        await recorder.emit(
                            RunEventKind.PACKAGES_INSTALLED,
                            packages=list(self.runtime_spec.packages),
                        )
                    repl = _ExecutionSessionRepl(
                        session,
                        lambda code, variables=None, timeout=None: self._run_session_code(
                            ctx,
                            code,
                            variables,
                            timeout=timeout,
                        ),
                    )
                    yield repl
                except BaseException as exc:
                    primary = exc
                    try:
                        await asyncio.shield(session.cancel())
                    except BaseException as cancel_error:
                        setattr(exc, "session_cancel_error", cancel_error)
                    raise
                finally:
                    finalize_task = asyncio.create_task(
                        self._finalize_runtime_inputs(ctx, session, primary)
                    )
                    finalize_cancellation: asyncio.CancelledError | None = None
                    try:
                        try:
                            await asyncio.shield(finalize_task)
                        except asyncio.CancelledError as exc:
                            finalize_cancellation = exc
                            if primary is None:
                                primary = exc
                            else:
                                setattr(primary, "session_finalize_cancellation", exc)
                            await finalize_task
                        if recorder is not None:
                            await recorder.emit(
                                RunEventKind.SESSION_FINALIZED,
                                backend=session.name,
                            )
                    except BaseException as finalize_error:
                        if recorder is not None:
                            recorder.mark_incomplete(finalize_error)
                            try:
                                await recorder.emit(
                                    RunEventKind.SESSION_FINALIZE_FAILED,
                                    backend=session.name,
                                    error_type=type(finalize_error).__name__,
                                    error=str(finalize_error),
                                )
                            except BaseException as evidence_error:
                                setattr(finalize_error, "evidence_error", evidence_error)
                        if primary is not None:
                            attribute = (
                                "input_adapter_finalize_error"
                                if getattr(
                                    finalize_error,
                                    "_predict_rlm_input_adapter_finalize",
                                    False,
                                )
                                else "session_finalize_error"
                            )
                            setattr(primary, attribute, finalize_error)
                        else:
                            primary = finalize_error
                            raise
                    if finalize_cancellation is not None:
                        if primary is not finalize_cancellation:
                            setattr(primary, "session_finalize_cancellation", finalize_cancellation)
            release_succeeded = True
        except BaseException as release_or_primary:
            if primary is release_or_primary:
                release_succeeded = True
            elif primary is not None:
                setattr(primary, "session_release_error", release_or_primary)
            else:
                primary = release_or_primary
            if session_name is None:
                try:
                    await self._finalize_runtime_inputs(ctx, None, primary)
                except BaseException as finalize_error:
                    setattr(primary, "input_adapter_finalize_error", finalize_error)
                    await self._record_pre_session_finalize_failure(finalize_error)
            if not release_succeeded and recorder is not None and session_name is not None:
                try:
                    await recorder.emit(
                        RunEventKind.SESSION_RELEASE_FAILED,
                        backend=session_name,
                        error_type=type(release_or_primary).__name__,
                        error=str(release_or_primary),
                    )
                except BaseException as evidence_error:
                    setattr(primary, "release_evidence_error", evidence_error)
            raise primary
        else:
            if primary is not None:
                raise primary
        finally:
            ctx.session = None
            if release_succeeded and recorder is not None and session_name is not None:
                try:
                    await recorder.emit(
                        RunEventKind.SESSION_RELEASED,
                        backend=session_name,
                    )
                except BaseException as release_evidence_error:
                    if primary is not None:
                        setattr(primary, "release_evidence_error", release_evidence_error)
                    else:
                        raise

    def _prepare_execution_tools(self) -> dict[str, Callable]:
        """Return only user-provided tools (including predict).

        Override parent to skip the default llm_query and llm_query_batched tools.
        """
        tools = self._runtime_tool_functions()
        return {
            name: function
            if name == "predict"
            else self._wrap_evidence_tool(name, function)
            for name, function in tools.items()
        }

    def _wrap_evidence_tool(
        self,
        name: str,
        function: Callable[..., Any],
    ) -> Callable[..., Any]:
        @functools.wraps(function)
        async def dispatch(*args: Any, **kwargs: Any) -> Any:
            recorder = self._evidence()
            call_id = uuid.uuid4().hex
            if recorder is not None:
                await recorder.emit(
                    RunEventKind.TOOL_STARTED,
                    call_id=call_id,
                    name=name,
                    args=list(args),
                    kwargs=kwargs,
                )
            try:
                result = await invoke_host_callable(function, *args, **kwargs)
            except BaseException as exc:
                if recorder is not None:
                    try:
                        await recorder.emit(
                            RunEventKind.TOOL_FINISHED,
                            call_id=call_id,
                            name=name,
                            error_type=type(exc).__name__,
                            error=str(exc),
                            cancelled=isinstance(exc, asyncio.CancelledError),
                        )
                    except BaseException as evidence_error:
                        setattr(exc, "evidence_error", evidence_error)
                raise
            if recorder is not None:
                await recorder.emit(
                    RunEventKind.TOOL_FINISHED,
                    call_id=call_id,
                    name=name,
                    result=result,
                )
            return result

        dispatch.__name__ = name
        preserve_sync_leaf(dispatch, function)
        return dispatch

    def _debug_lm_metadata(self, metadata: Any | None) -> dict[str, Any]:
        if metadata is None:
            return {}
        attrs: dict[str, Any] = {}
        for source_name, target_name in (
            ("finish_reason", "lm_finish_reason"),
            ("truncation_reason", "lm_truncation_reason"),
            ("max_tokens", "lm_max_tokens"),
            ("input_tokens", "lm_prompt_tokens"),
            ("cached_input_tokens", "lm_cached_prompt_tokens"),
            ("cache_read_ratio", "lm_prompt_cache_read_ratio"),
            ("output_tokens", "lm_output_tokens"),
        ):
            value = getattr(metadata, source_name, None)
            if value is not None:
                attrs[target_name] = value
        truncated = getattr(metadata, "truncated", None)
        if truncated is not None:
            attrs["lm_truncated"] = truncated
        return attrs

    @contextmanager
    def _lm_context(self):
        """Set the configured LM as active context and capture it for predict() calls."""
        context_kwargs: dict[str, Any] = {}
        run_ctx = current_run_context()
        active_lm = run_ctx.state.get("lm") if run_ctx is not None else self._lm
        if active_lm is not None:
            context_kwargs["lm"] = active_lm
        if getattr(dspy.settings, "adapter", None) is None:
            context_kwargs["adapter"] = _ValidatingChatAdapter()

        with dspy.context(**context_kwargs):
            self._context_lm = dspy.settings.lm
            try:
                yield
            finally:
                self._context_lm = None

    def _begin_telemetry_execution(self) -> TelemetryContext | None:
        """Prepare per-execution telemetry identity."""
        telemetry_context = self._telemetry_context
        predictor_id = make_span_id("prlm")
        self._current_predictor_id = predictor_id
        if telemetry_context is None:
            self._current_telemetry_context = None
            return None
        self._current_telemetry_context = telemetry_context
        return self._current_telemetry_context

    def _clear_telemetry_execution(self) -> None:
        self._current_telemetry_context = None
        self._current_predictor_id = None

    def _telemetry_attrs(
        self,
        *,
        iteration: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        attrs: dict[str, Any] = {}
        if iteration is not None:
            attrs["iteration"] = iteration
        current_predictor_id = getattr(self, "_current_predictor_id", None)
        if current_predictor_id is not None:
            attrs["predict_rlm.predictor_id"] = current_predictor_id
        if extra:
            attrs.update(extra)
        return attrs

    def _write_telemetry_span(
        self,
        name: str,
        *,
        iteration: int | None = None,
        status: str | dict[str, Any] = "OK",
        attributes: dict[str, Any] | None = None,
        start_time_unix_nano: int | None = None,
        end_time_unix_nano: int | None = None,
    ) -> None:
        telemetry_context = getattr(self, "_current_telemetry_context", None)
        if telemetry_context is None:
            return
        try:
            telemetry_context.write_span(
                name,
                event_domain="rlm",
                status=status,
                start_time_unix_nano=start_time_unix_nano,
                end_time_unix_nano=end_time_unix_nano,
                attributes=self._telemetry_attrs(
                    iteration=iteration,
                    extra=attributes,
                ),
            )
        except Exception:
            return

    def _generated_code_attributes(
        self, *, iteration: int, pred: Any, code: str
    ) -> dict[str, Any]:
        reasoning = getattr(pred, "reasoning", "") or ""
        return self._telemetry_attrs(
            iteration=iteration + 1,
            extra={
                "has_code": bool(code),
                "code_chars": len(code),
                "code_sha256": _sha256_text(code),
                "reasoning_chars": len(reasoning),
            },
        )

    def _write_generated_code_event(self, *, iteration: int, pred: Any, code: str) -> None:
        telemetry_context = getattr(self, "_current_telemetry_context", None)
        if telemetry_context is None:
            return
        try:
            telemetry_context.write_span(
                "rlm.iteration.generated_code",
                event_domain="rlm",
                attributes=self._generated_code_attributes(
                    iteration=iteration,
                    pred=pred,
                    code=code,
                ),
            )
        except Exception:
            return

    def _process_execution_result(self, pred: Any, *args: Any) -> dspy.Prediction | Any:
        original_verbose = self.verbose
        self.verbose = False
        try:
            return super()._process_execution_result(pred, *args)
        finally:
            self.verbose = original_verbose

    def _submit_payload_from_prediction(
        self,
        prediction: dspy.Prediction,
        output_field_names: list[str],
    ) -> dict[str, Any]:
        return {
            name: prediction.get(name)
            for name in output_field_names
            if name in prediction.keys()
        }

    def _iteration_logging_enabled(self) -> bool:
        return bool(getattr(self, "verbose", False))

    def _debug_logging_enabled(self) -> bool:
        return bool(getattr(self, "_debug", False))

    def _log_lifecycle(self, event: str, **fields: Any) -> None:
        if self._debug_logging_enabled():
            logger.debug("%s%s", event, format_log_fields(fields))

    def _interpreter_backend_label(self, repl: Any) -> str:
        if type(repl).__name__ == "SbxBackend":
            return "sbx"
        if isinstance(JspiBackend, type) and isinstance(repl, JspiBackend):
            return "jspi"
        return type(repl).__name__

    def _configure_interpreter_debug(self, repl: Any) -> dict[str, bool]:
        return self._configure_interpreter_logging(repl)

    def _ensure_interpreter_skill_packages(self, repl: Any) -> bool:
        packages = self._runtime_packages()
        if not packages:
            return False
        ensure_skill_packages = self._declared_callable(repl, "ensure_skill_packages")
        if ensure_skill_packages is None:
            return False
        ensure_skill_packages(list(packages))
        return True

    def _configure_interpreter_logging(self, repl: Any) -> dict[str, bool]:
        configured = {"debug": False, "verbose": False}
        debug_enabled = self._debug_logging_enabled()
        verbose_enabled = self._iteration_logging_enabled()

        configure_runtime = self._declared_callable(repl, "configure_runtime")
        if configure_runtime is not None:
            kwargs = self._accepted_kwargs(
                configure_runtime,
                debug=debug_enabled,
                verbose=verbose_enabled,
            )
            if kwargs:
                configure_runtime(**kwargs)
                configured["debug"] = "debug" in kwargs
                configured["verbose"] = "verbose" in kwargs

        configure_debug = self._declared_callable(repl, "configure_debug")
        if not configured["debug"] and configure_debug is not None:
            configure_debug(debug_enabled)
            configured["debug"] = True

        configure_verbose = self._declared_callable(repl, "configure_verbose")
        if not configured["verbose"] and configure_verbose is not None:
            configure_verbose(verbose_enabled)
            configured["verbose"] = True

        return configured

    def _declared_callable(self, obj: Any, name: str) -> Callable[..., Any] | None:
        try:
            inspect.getattr_static(obj, name)
        except AttributeError:
            return None
        value = getattr(obj, name, None)
        return value if callable(value) else None

    def _accepted_kwargs(self, func: Callable[..., Any], **kwargs: Any) -> dict[str, Any]:
        try:
            parameters = inspect.signature(func).parameters
        except (TypeError, ValueError):
            return kwargs
        if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
            return kwargs
        return {name: value for name, value in kwargs.items() if name in parameters}

    def _begin_iteration_log(self, *, iteration: int, pred: Any, code: str) -> bool:
        if not self._iteration_logging_enabled():
            return False
        configure_predict_rlm_logging(verbose=True)
        execution_timeout = getattr(pred, "execution_timeout_seconds", None)
        if isinstance(execution_timeout, bool) or not isinstance(
            execution_timeout, (int, float)
        ):
            execution_timeout = None
        emit_trace_iteration_start(
            iteration=iteration + 1,
            max_iterations=self.max_iterations,
            reasoning=getattr(pred, "reasoning", "") or "",
            code=code,
            execution_timeout_seconds=execution_timeout,
        )
        return True

    def _emit_iteration_output_log(self, result: Any) -> None:
        if self._iteration_logging_enabled():
            emit_trace_iteration_output("" if result is None else result)

    def _emit_iteration_submit_log(
        self,
        prediction: dspy.Prediction,
        output_field_names: list[str],
    ) -> None:
        if self._iteration_logging_enabled():
            emit_trace_iteration_submit(
                self._submit_payload_from_prediction(prediction, output_field_names)
            )

    def _finish_iteration_log(self, opened: bool) -> None:
        if opened:
            emit_trace_iteration_end()

    def _action_generation_error_attrs(
        self,
        exc: BaseException,
        *,
        lm_metadata: Any | None = None,
    ) -> dict[str, Any]:
        exception_message = str(exc)
        invalid_action_output = isinstance(exc, RuntimeError) and exception_message.startswith(
            "PredictRLM action adapter returned invalid"
        )
        normalized_exception_message = exception_message.lower()
        truncated = bool(lm_metadata and lm_metadata.truncated) or (
            "lm response was truncated" in normalized_exception_message
            or (
                "truncated" in normalized_exception_message
                and any(
                    marker in normalized_exception_message
                    for marker in ("max_tokens", "max output")
                )
            )
        )
        failure_class = "unknown"
        if truncated:
            failure_class = "model_output_truncated"
        elif isinstance(exc, (AdapterParseError, ValidationError)) or invalid_action_output:
            failure_class = "model_no_code_generated"
        attrs: dict[str, Any] = {
            "has_code": False,
            "code_chars": 0,
            "code_sha256": _sha256_text(""),
            "reasoning_chars": 0,
            "error.type": type(exc).__name__,
            "error.message_chars": len(exception_message),
            "error.message_sha256": _sha256_text(exception_message),
            "failure.class": failure_class,
        }
        if lm_metadata is not None:
            attrs["lm.truncated"] = lm_metadata.truncated
            if lm_metadata.truncation_reason is not None:
                attrs["lm.truncation_reason"] = lm_metadata.truncation_reason
            if lm_metadata.finish_reason is not None:
                attrs["lm.finish_reason"] = lm_metadata.finish_reason
            if lm_metadata.max_tokens is not None:
                attrs["lm.max_tokens"] = lm_metadata.max_tokens
            if lm_metadata.output_tokens is not None:
                attrs["lm.output_tokens"] = lm_metadata.output_tokens
        elif truncated:
            attrs["lm.truncated"] = True
            attrs["lm.truncation_reason"] = "max_tokens"
        return attrs

    def _action_generation_error_status(self, exc: BaseException) -> dict[str, Any]:
        return {
            "code": "ERROR",
            "message": f"{type(exc).__name__}: action generation did not produce parsed code",
        }

    def _action_execution_timeout(self, pred: Any) -> float | None:
        value = getattr(pred, "execution_timeout_seconds", None)
        if type(value).__module__ == "unittest.mock":
            return None
        try:
            return validate_execution_timeout(value)
        except ValueError as exc:
            raise RuntimeError(
                "PredictRLM action adapter returned invalid "
                "execution_timeout_seconds; expected a positive number or null."
            ) from exc

    def _telemetry_ref(self) -> dict[str, Any] | None:
        telemetry_context = getattr(self, "_current_telemetry_context", None)
        if telemetry_context is None:
            return None
        ref: dict[str, Any] = {"trace_id": telemetry_context.trace_id}
        current_predictor_id = getattr(self, "_current_predictor_id", None)
        if current_predictor_id is not None:
            ref["predictor_id"] = current_predictor_id
        sink_path = getattr(telemetry_context.sink, "path", None)
        if sink_path is not None:
            parts = Path(sink_path).parts
            if "telemetry" in parts:
                idx = parts.index("telemetry")
                ref["events_path"] = str(Path(*parts[idx:]))
            else:
                ref["events_path"] = str(sink_path)
        return ref

    def _get_active_dspy_callbacks(self) -> list:
        """Return active callbacks (global + instance-level), in dispatch order.

        Mirrors DSPy's callback collection semantics so that any
        ``BaseCallback`` registered via ``dspy.configure(callbacks=[...])``
        or ``rlm.callbacks = [...]`` receives RLM iteration events alongside
        the standard module/lm/tool events.
        """
        global_cbs = dspy.settings.get("callbacks", []) or []
        instance_cbs = getattr(self, "callbacks", []) or []
        return list(global_cbs) + list(instance_cbs)

    def _dispatch_sync(self, method: str, /, **kwargs: Any) -> None:
        """Invoke ``method`` on each active callback. Sync-only path.

        Handler exceptions are logged and swallowed so a misbehaving
        callback can never break the run (matches DSPy's own behavior in
        ``_execute_end_callbacks``). If a handler returns a coroutine the
        coroutine is closed and a warning is logged — sync ``forward()``
        cannot await it. Use ``acall()`` for async handlers.
        """
        callbacks = self._get_active_dspy_callbacks()
        if not callbacks:
            return
        for cb in callbacks:
            handler = getattr(cb, method, None)
            if handler is None:
                continue
            try:
                result = handler(**kwargs)
            except Exception as e:
                _CB_LOGGER.warning(
                    "Error in RLM callback %s.%s: %s", type(cb).__name__, method, e
                )
                continue
            if inspect.iscoroutine(result):
                _CB_LOGGER.warning(
                    "Async callback %s.%s invoked from sync forward(); "
                    "coroutine was not awaited. Use acall() for async handlers.",
                    type(cb).__name__,
                    method,
                )
                result.close()

    async def _dispatch_async(self, method: str, /, **kwargs: Any) -> None:
        """Async variant of ``_dispatch_sync`` that awaits coroutine handlers."""
        callbacks = self._get_active_dspy_callbacks()
        if not callbacks:
            return
        for cb in callbacks:
            handler = getattr(cb, method, None)
            if handler is None:
                continue
            try:
                result = handler(**kwargs)
                if inspect.iscoroutine(result):
                    await result
            except Exception as e:
                _CB_LOGGER.warning(
                    "Error in RLM callback %s.%s: %s", type(cb).__name__, method, e
                )

    @contextmanager
    def _iteration_callback_scope(
        self, call_id: str | None, iteration: int
    ) -> Iterator[_IterationCallbackState]:
        state = _IterationCallbackState()
        self._dispatch_sync(
            "on_rlm_iteration_start",
            call_id=call_id,
            instance=self,
            iteration=iteration + 1,
            max_iterations=self.max_iterations,
        )
        try:
            yield state
        except Exception as e:
            state.exception = e
            raise
        finally:
            self._dispatch_sync(
                "on_rlm_iteration_end",
                call_id=call_id,
                instance=self,
                iteration=iteration + 1,
                step=state.step,
                is_final=state.is_final,
                exception=state.exception,
            )

    @asynccontextmanager
    async def _aiteration_callback_scope(
        self, call_id: str | None, iteration: int
    ) -> AsyncIterator[_IterationCallbackState]:
        state = _IterationCallbackState()
        await self._dispatch_async(
            "on_rlm_iteration_start",
            call_id=call_id,
            instance=self,
            iteration=iteration + 1,
            max_iterations=self.max_iterations,
        )
        try:
            yield state
        except Exception as e:
            state.exception = e
            raise
        finally:
            await self._dispatch_async(
                "on_rlm_iteration_end",
                call_id=call_id,
                instance=self,
                iteration=iteration + 1,
                step=state.step,
                is_final=state.is_final,
                exception=state.exception,
            )

    def forward(self, **kwargs: Any) -> dspy.Prediction:
        """Execute the RLM with captured context LM for thread-safe predict calls.

        Handles File inputs by mounting them into the sandbox,
        and File outputs by syncing them back to the host.
        """
        if type(self)._forward_traced is not _KERNEL_FORWARD_TRACED:
            with self._lm_context():
                return self._forward_traced(None, **kwargs)
        bound_forward = getattr(self._aforward_traced, "__func__", None)
        if bound_forward is not _KERNEL_AFORWARD_TRACED:
            with self._lm_context():
                result = self._aforward_traced(None, **kwargs)
            return self._run_async(result) if inspect.isawaitable(result) else result
        return self._run_async(self._aforward_kernel(kwargs, sync_callbacks=True))

    def _run_async(self, awaitable: Any) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(awaitable)

        import nest_asyncio

        nest_asyncio.apply(loop)
        return loop.run_until_complete(awaitable)

    def _build_run_trace(
        self,
        status: Literal["in_progress", "completed", "max_iterations", "error"],
        steps: list[IterationStep],
        lm: Any,
        sub_lm: Any,
        lm_hist_start: int,
        sub_hist_start: int,
        run_start: float,
    ) -> RunTrace:
        trace_steps = steps
        pending_entry = getattr(self, "_partial_pending_entry", None)
        if status in {"in_progress", "error"} and pending_entry is not None:
            trace_steps = list(steps)
            pending_start = getattr(self, "_partial_pending_start", None)
            trace_steps.append(
                IterationStep(
                    iteration=len(steps) + 1,
                    reasoning=pending_entry.reasoning,
                    code=pending_entry.code,
                    output=pending_entry.output,
                    untruncated_output=pending_entry.output,
                    error=True,
                    duration_ms=ms_since(pending_start) if pending_start else 0,
                    tool_calls=drain_tool_calls(),
                    predict_calls=drain_predict_calls(),
                )
            )
        # Per-RLM instance ``lm`` has its own history (isolated via
        # ``lm.copy()`` in ``__init__``) so ``usage_since`` is free of
        # the shared-history concurrency bug that plagued the old
        # delta-read pattern. Cost is populated by DSPy's
        # ``BaseLM._process_lm_response`` from
        # ``_hidden_params["response_cost"]`` — LiteLLM's authoritative
        # number with cache/batch discounts baked in.
        main_usage = usage_since(lm, lm_hist_start)
        sub_usage = (
            usage_since(sub_lm, sub_hist_start)
            if sub_lm is not None and sub_lm is not lm
            else None
        )

        return RunTrace(
            status=status,
            model=str(getattr(lm, "model", lm)),
            sub_model=str(getattr(sub_lm, "model", sub_lm)) if sub_lm is not lm else None,
            iterations=len(trace_steps),
            max_iterations=self.max_iterations,
            duration_ms=ms_since(run_start),
            usage=LMUsage(main=main_usage, **({"sub": sub_usage} if sub_usage else {})),
            telemetry_ref=self._telemetry_ref() if status == "error" else None,
            steps=trace_steps,
        )

    def _export_current_trace(
        self,
        status: Literal["in_progress", "completed", "max_iterations", "error"],
    ) -> None:
        context = getattr(self, "_trace_export_context", None)
        if context is None:
            return
        self._export_run_trace(
            self._build_run_trace(
                status=status,
                steps=context.steps,
                lm=context.lm,
                sub_lm=context.sub_lm,
                lm_hist_start=context.lm_hist_start,
                sub_hist_start=context.sub_hist_start,
                run_start=context.run_start,
            )
        )

    def _export_run_trace(self, trace: RunTrace) -> None:
        path = getattr(self, "_trace_export_path", None)
        if path is None:
            return
        tmp_path: Path | None = None
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
            tmp_path.write_text(trace.to_exportable_json(), encoding="utf-8")
            os.replace(tmp_path, path)
            tmp_path = None
        except Exception as exc:
            try:
                self._log_lifecycle(
                    "rlm.trace_export.error",
                    error_type=type(exc).__name__,
                    path=str(path),
                )
            except Exception:
                pass
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def _history_from_prediction(self, prediction: dspy.Prediction, fallback_history: Any):
        from dspy.primitives.repl_types import REPLEntry, REPLHistory

        entries = []
        for item in getattr(prediction, "trajectory", None) or []:
            if isinstance(item, REPLEntry):
                entries.append(item)
            elif isinstance(item, dict):
                entries.append(
                    REPLEntry(
                        reasoning=item.get("reasoning", "") or "",
                        code=item.get("code", "") or "",
                        output=item.get("output", "") or "",
                    )
                )

        if entries:
            return REPLHistory(
                entries=entries,
                max_output_chars=getattr(fallback_history, "max_output_chars", 10_000),
            )
        return fallback_history

    def _submitted_payload(
        self,
        prediction: dspy.Prediction,
        output_field_names: list[str],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for name in output_field_names:
            if hasattr(prediction, name):
                payload[name] = getattr(prediction, name)
        return payload

    def _submit_confirmation_message(
        self, context: SubmitConfirmationContext
    ) -> str | None:
        callback = getattr(self, "_submit_confirmation", None)
        if callback is None:
            return None
        message = callback(context)
        if message is None or message == "":
            return None
        if not isinstance(message, str):
            raise TypeError(
                "submit_confirmation must return a string, None, or an empty string."
            )
        return message

    def _maybe_request_submit_confirmation(
        self,
        prediction: dspy.Prediction,
        history: Any,
        input_args: dict[str, Any],
        output_field_names: list[str],
        iteration: int,
        pending: bool,
    ):
        if pending or getattr(self, "_submit_confirmation", None) is None:
            return None

        final_history = self._history_from_prediction(prediction, history)
        latest_entry = final_history.entries[-1] if getattr(final_history, "entries", None) else None
        latest_observation = latest_entry.output if latest_entry is not None else ""
        context = SubmitConfirmationContext(
            inputs=dict(input_args),
            signature=self.signature,
            output_field_names=tuple(output_field_names),
            prediction=prediction,
            submitted_payload=self._submitted_payload(prediction, output_field_names),
            history=final_history,
            latest_observation=latest_observation,
            reasoning=(
                latest_entry.reasoning
                if latest_entry is not None
                else getattr(prediction, "final_reasoning", "") or ""
            ),
            code=latest_entry.code if latest_entry is not None else "",
            iteration=iteration + 1,
        )
        message = self._submit_confirmation_message(context)
        if not message:
            return None

        return history.append(
            reasoning=context.reasoning,
            code=context.code,
            output=message,
        )

    async def aforward(self, **kwargs: Any) -> dspy.Prediction:
        """Async version of forward(). Sets the LM context for async execution.

        Uses aexecute() on the interpreter so asyncio.gather can run
        multiple rollouts concurrently.
        """
        return await self._aforward_kernel(kwargs, sync_callbacks=False)

    async def _aforward_kernel(
        self,
        kwargs: dict[str, Any],
        *,
        sync_callbacks: bool,
    ) -> dspy.Prediction:
        ctx = self._new_run_context(kwargs)
        ctx.state["sync_callbacks"] = sync_callbacks
        recorder = EvidenceRecorder(ctx, self.runtime_spec.events)
        ctx.state["evidence"] = recorder
        async with use_run_context(ctx):
            cleanup_complete = False
            try:
                await recorder.emit(
                    RunEventKind.RUN_STARTED,
                    inputs=dict(ctx.input_values),
                )
                await self._prepare_runtime_inputs(ctx, kwargs)
                await self._prepare_runtime_outputs(ctx)
                kwargs = {
                    **kwargs,
                    **{
                        name: binding.prepared.model_value
                        for name, binding in ctx.input_bindings.items()
                    },
                }
                with self._lm_context():
                    result = self._aforward_traced(None, **kwargs)
                    prediction = await result if inspect.isawaitable(result) else result
                await ctx.cleanup()
                cleanup_complete = True
            except BaseException as exc:
                if not cleanup_complete:
                    try:
                        await ctx.cleanup()
                    except BaseException as cleanup_error:
                        setattr(exc, "cleanup_error", cleanup_error)
                await recorder.finish_failure(exc)
                self._attach_runtime_evidence(getattr(exc, "trace", None), recorder)
                raise
            else:
                trace = getattr(prediction, "trace", None)
                outputs = {
                    name: getattr(prediction, name, None)
                    for name in self.signature.output_fields
                }
                try:
                    await recorder.finish_success(
                        status=getattr(trace, "status", "completed"),
                        outputs=outputs,
                    )
                except BaseException as exc:
                    if trace is not None:
                        trace.status = "error"
                        setattr(exc, "trace", trace)
                    self._attach_runtime_evidence(trace, recorder)
                    raise
                self._attach_runtime_evidence(trace, recorder)
                return prediction

    def _attach_runtime_evidence(
        self,
        trace: RunTrace | None,
        recorder: EvidenceRecorder,
    ) -> None:
        if trace is None:
            return
        trace.evidence = RunEvidence(
            run_id=recorder.ctx.run_id,
            complete=recorder.complete,
            terminal_outcome=recorder.ctx.terminal_outcome,
            events=[
                RunEvidenceEvent(
                    sequence=event.sequence,
                    kind=event.kind.value,
                    timestamp_ns=event.timestamp_ns,
                    data=dict(event.data),
                )
                for event in recorder.events
            ],
        )

    @asynccontextmanager
    async def _iteration_callback_scope_for_run(
        self,
        call_id: str | None,
        iteration: int,
    ) -> AsyncIterator[_IterationCallbackState]:
        ctx = current_run_context()
        if ctx is not None and ctx.state.get("sync_callbacks"):
            with self._iteration_callback_scope(call_id, iteration) as state:
                yield state
            return
        async with self._aiteration_callback_scope(call_id, iteration) as state:
            yield state

    async def _prepare_runtime_inputs(
        self,
        ctx: RunContext,
        input_values: dict[str, Any],
    ) -> None:
        recorder = self._evidence()
        for field_name, field_info in self.signature.input_fields.items():
            if field_name not in input_values:
                continue
            value = input_values[field_name]
            field = FieldDescriptor(field_name, field_info.annotation)
            adapter = resolve_input_adapter(
                self.runtime_spec.input_adapters,
                field,
                value,
            )
            prepared = compile_prepared_input(
                field,
                await adapter.prepare(field, value, ctx),
            )
            ctx.input_bindings[field_name] = PreparedInputBinding(
                field=field,
                adapter=adapter,
                prepared=prepared,
            )
            if recorder is not None:
                await recorder.emit(
                    RunEventKind.INPUT_PREPARED,
                    field=field_name,
                    adapter=adapter.name,
                    artifact_ids=[artifact.id for artifact in prepared.artifacts],
                )

    async def _open_runtime_inputs(self, ctx: RunContext) -> None:
        for binding in ctx.input_bindings.values():
            if binding.open_entered:
                continue
            binding.open_entered = True
            try:
                await binding.adapter.open(
                    binding.field,
                    binding.prepared,
                    ctx,
                    self.runtime_spec.execution,
                )
            except BaseException as exc:
                try:
                    await self._finalize_runtime_inputs(ctx, None, exc)
                except BaseException as finalize_error:
                    setattr(exc, "input_adapter_finalize_error", finalize_error)
                    await self._record_pre_session_finalize_failure(finalize_error)
                raise

    async def _record_pre_session_finalize_failure(
        self,
        finalize_error: BaseException,
    ) -> None:
        recorder = self._evidence()
        if recorder is None:
            return
        recorder.mark_incomplete(finalize_error)
        try:
            await recorder.emit(
                RunEventKind.SESSION_FINALIZE_FAILED,
                backend=self.runtime_spec.execution.name,
                error_type=type(finalize_error).__name__,
                error=str(finalize_error),
            )
        except BaseException as evidence_error:
            setattr(finalize_error, "evidence_error", evidence_error)

    async def _prepare_runtime_outputs(self, ctx: RunContext) -> None:
        for field_name, field_info in self.signature.output_fields.items():
            field = FieldDescriptor(field_name, field_info.annotation)
            matches = [
                adapter
                for adapter in self.runtime_spec.output_adapters
                if adapter.supports(field)
            ]
            if not matches:
                continue
            adapter = resolve_output_adapter(matches, field)
            requirements = await adapter.prepare_session(
                field,
                ctx.input_values.get(field_name),
                ctx,
            )
            if requirements is not None:
                if not isinstance(requirements, SessionRequirements):
                    raise TypeError(
                        "OutputAdapter.prepare_session() must return "
                        "SessionRequirements or None"
                    )
                ctx.output_requirements[field_name] = requirements

    async def _bind_runtime_inputs(self, ctx: RunContext) -> None:
        recorder = self._evidence()
        session = ctx.session
        if session is None:
            raise RuntimeError("Artifact binding requires an active execution session")
        for binding in ctx.input_bindings.values():
            if binding.bound:
                continue
            binding.bound = True
            ctx.bound_input_bindings.append(binding)
            bound = await binding.adapter.bind(
                binding.field,
                binding.prepared,
                ctx,
                session,
            )
            for artifact_binding in bound.bindings:
                ctx.bind(artifact_binding)
                if recorder is not None:
                    await recorder.emit(
                        RunEventKind.ARTIFACT_MOUNTED,
                        artifact_id=artifact_binding.artifact_id,
                        path=artifact_binding.path,
                    )
            if bound.model_value != binding.prepared.model_value:
                binding.prepared = replace(
                    binding.prepared,
                    model_value=bound.model_value,
                )

    async def _run_session_code(
        self,
        ctx: RunContext,
        code: str,
        variables: Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> ExecutionResult:
        session = ctx.session
        if session is None:
            raise RuntimeError("Input lifecycle requires an active execution session")
        result: ExecutionResult | None = None
        execution_error: BaseException | None = None
        try:
            result = await session.run_code(code, variables, timeout=timeout)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            execution_error = exc

        await self._after_runtime_input_execution(ctx, result, execution_error)
        if execution_error is not None:
            raise execution_error
        assert result is not None
        return result

    async def _after_runtime_input_execution(
        self,
        ctx: RunContext,
        result: ExecutionResult | None,
        execution_error: BaseException | None,
    ) -> None:
        session = ctx.session
        if session is None:
            raise RuntimeError("Input lifecycle requires an active execution session")
        after_error: BaseException | None = None
        additional_errors = []
        for binding in ctx.bound_input_bindings:
            try:
                await binding.adapter.after_execution(
                    binding.field,
                    binding.prepared,
                    ctx,
                    session,
                    result,
                    execution_error,
                )
            except BaseException as exc:
                if after_error is None:
                    after_error = exc
                else:
                    additional_errors.append(exc)

        if after_error is not None and additional_errors:
            setattr(
                after_error,
                "input_adapter_after_execution_errors",
                tuple(additional_errors),
            )
        if after_error is not None:
            setattr(after_error, "_predict_rlm_input_adapter_after_execution", True)
            if execution_error is not None:
                setattr(
                    execution_error,
                    "input_adapter_after_execution_error",
                    after_error,
                )
                setattr(
                    execution_error,
                    "_predict_rlm_input_adapter_after_execution",
                    True,
                )
                return
            raise after_error

    async def _finalize_runtime_inputs(
        self,
        ctx: RunContext,
        session: Any | None,
        error: BaseException | None,
    ) -> None:
        adapter_error: BaseException | None = None
        additional_errors = []
        for binding in reversed(tuple(ctx.input_bindings.values())):
            if not binding.open_entered or binding.finalized:
                continue
            binding.finalized = True
            try:
                await binding.adapter.finalize(
                    binding.field,
                    binding.prepared,
                    ctx,
                    session,
                    error,
                )
            except BaseException as exc:
                if adapter_error is None:
                    adapter_error = exc
                else:
                    additional_errors.append(exc)
        session_error: BaseException | None = None
        if session is not None and not ctx.session_finalized:
            ctx.session_finalized = True
            try:
                await session.finalize()
            except BaseException as exc:
                session_error = exc
        if adapter_error is not None:
            setattr(adapter_error, "_predict_rlm_input_adapter_finalize", True)
            if additional_errors:
                setattr(adapter_error, "input_finalize_errors", tuple(additional_errors))
            if session_error is not None:
                setattr(adapter_error, "session_finalize_error", session_error)
            raise adapter_error
        if session_error is not None:
            raise session_error

    def _configure_run_predictors(
        self,
        ctx: RunContext,
        file_plan: dict[str, Any] | None,
    ) -> None:
        prepared_instructions = tuple(
            instruction
            for binding in ctx.input_bindings.values()
            for instruction in binding.prepared.instructions
        )
        file_instructions = file_plan["instructions"] if file_plan else ""
        output_instructions = tuple(
            f"Output `{field_name}` is reserved as {reservation.model_value!r}."
            for field_name, reservation in ctx.output_reservations.items()
        )
        invocation_instructions = "\n\n".join(
            instruction
            for instruction in (
                file_instructions,
                *prepared_instructions,
                *output_instructions,
            )
            if instruction
        )
        prompt_signature = dspy.Signature(
            deepcopy(self.signature.fields),
            str(self.signature.instructions),
        )
        has_prompt_signature_transform = False
        for binding in ctx.input_bindings.values():
            transform_prompt_signature = binding.adapter._transform_prompt_signature
            has_prompt_signature_transform |= (
                getattr(transform_prompt_signature, "__func__", transform_prompt_signature)
                is not InputAdapter._transform_prompt_signature
            )
            prompt_signature = transform_prompt_signature(
                prompt_signature, binding.field, binding.prepared, ctx
            )
            if not (
                isinstance(prompt_signature, type)
                and issubclass(prompt_signature, dspy.Signature)
            ):
                raise TypeError(
                    f"Input adapter {binding.adapter.name!r} "
                    "_transform_prompt_signature must return a dspy.Signature class"
                )

        if not has_prompt_signature_transform:
            if invocation_instructions:
                action, extract = self._build_signatures_with_files(invocation_instructions)
            else:
                action, extract = self.generate_action, self.extract
        else:
            token = _RUN_PROMPT_SIGNATURE.set((self, prompt_signature))
            try:
                action, extract = self._build_signatures_with_files(invocation_instructions)
            finally:
                _RUN_PROMPT_SIGNATURE.reset(token)

        for binding in ctx.input_bindings.values():
            action = self._with_adapter_prompt(action, binding, ctx)
            extract = self._with_adapter_prompt(extract, binding, ctx)

        from .in_context import _build_in_context_instructions

        in_context = _build_in_context_instructions(
            self.signature,
            {
                name: binding.prepared.model_value
                for name, binding in ctx.input_bindings.items()
            },
        )
        if in_context:
            action = self._with_prompt_appendix(action, in_context)
            extract = self._with_prompt_appendix(extract, in_context)
        ctx.state["generate_action"] = action
        ctx.state["extract"] = extract

    def _with_adapter_prompt(
        self,
        predictor: dspy.Predict,
        binding: PreparedInputBinding,
        ctx: RunContext,
    ) -> dspy.Predict:
        append_prompt = binding.adapter.append_prompt
        if (
            getattr(append_prompt, "__func__", append_prompt)
            is InputAdapter.append_prompt
        ):
            return predictor
        prompt = str(predictor.signature.instructions)
        transformed = append_prompt(prompt, binding.field, binding.prepared, ctx)
        if not isinstance(transformed, str):
            raise TypeError(
                f"Input adapter {binding.adapter.name!r} append_prompt must return str"
            )
        if transformed == prompt:
            return predictor
        run_predictor = predictor.deepcopy()
        run_predictor.signature = predictor.signature.with_instructions(transformed)
        return run_predictor

    def _with_prompt_appendix(self, predictor: dspy.Predict, appendix: str) -> dspy.Predict:
        run_predictor = predictor.deepcopy()
        instructions = "\n\n".join(
            instruction
            for instruction in (str(predictor.signature.instructions), appendix)
            if instruction
        )
        run_predictor.signature = predictor.signature.with_instructions(instructions)
        return run_predictor

    async def _setup_runtime_modules(self, ctx: RunContext) -> None:
        session = ctx.session
        if session is None:
            return
        if not self._skill_modules:
            return
        for module_name, source_path in self._skill_modules.items():
            artifact = Artifact(
                id=f"skill-module-{module_name}-{ctx.run_id}",
                kind="compat.module",
                metadata={
                    "source_path": source_path,
                    "sandbox_path": f"/sandbox/lib/{module_name}.py",
                },
            )
            binding = await session.mount(artifact)
            ctx.bind(binding)
        await session.run_code(
            "import sys; sys.path.insert(0, '/sandbox/lib')",
        )

    async def _reserve_runtime_outputs(self, ctx: RunContext) -> None:
        session = ctx.session
        if session is None:
            raise RuntimeError("Output reservation requires an active execution session")
        recorder = self._evidence()
        for field_name, field_info in self.signature.output_fields.items():
            field = FieldDescriptor(field_name, field_info.annotation)
            matches = [
                adapter
                for adapter in self.runtime_spec.output_adapters
                if adapter.supports(field)
            ]
            if not matches:
                continue
            adapter = resolve_output_adapter(matches, field)
            reservation = await adapter.reserve(
                field,
                ctx.input_values.get(field_name),
                ctx,
                session,
            )
            if reservation.field != field:
                raise ValueError(
                    f"Output adapter {adapter.name!r} reserved field "
                    f"{reservation.field.name!r} while preparing {field.name!r}"
                )
            validate_output_sandbox_root_reservation(
                ctx.input_bindings,
                ctx.output_reservations,
                reservation,
            )
            ctx.reserve(reservation, adapter)
            if recorder is not None:
                binding = ctx.artifact_bindings.get(reservation.artifact.id)
                if binding is not None:
                    await recorder.emit(
                        RunEventKind.ARTIFACT_MOUNTED,
                        artifact_id=reservation.artifact.id,
                        path=binding.path,
                    )
                await recorder.emit(
                    RunEventKind.OUTPUT_RESERVED,
                    field=field_name,
                    adapter=adapter.name,
                    artifact_id=reservation.artifact.id,
                )

    async def _materialize_runtime_outputs(
        self,
        ctx: RunContext,
        prediction: dspy.Prediction,
    ) -> None:
        session = ctx.session
        if session is None:
            raise RuntimeError("Output materialization requires an active execution session")
        recorder = self._evidence()
        for field_name, reservation in ctx.output_reservations.items():
            adapter = ctx.output_adapters[field_name]
            value = await adapter.materialize(
                reservation,
                getattr(prediction, field_name, None),
                ctx,
                session,
            )
            setattr(prediction, field_name, value)
            if recorder is not None:
                await recorder.emit(
                    RunEventKind.ARTIFACT_COLLECTED,
                    artifact_id=reservation.artifact.id,
                    field=field_name,
                )
                await recorder.emit(
                    RunEventKind.OUTPUT_MATERIALIZED,
                    field=field_name,
                    adapter=adapter.name,
                )

    def _begin_action_generation(self, variables, iteration: int) -> tuple[list[str], int]:
        variables_info = [variable.format() for variable in variables]
        action_start_ns = time.time_ns()
        self._write_telemetry_span(
            "rlm.action_generation.start",
            iteration=iteration + 1,
            start_time_unix_nano=action_start_ns,
            end_time_unix_nano=action_start_ns,
        )
        self._log_lifecycle(
            "rlm.action_generation.start",
            iteration=iteration + 1,
            max_iterations=self.max_iterations,
        )
        return variables_info, action_start_ns

    def _validate_iteration_action(self, pred: Any) -> None:
        if not isinstance(getattr(pred, "reasoning", None), str):
            raise RuntimeError(
                "PredictRLM action adapter returned invalid reasoning; "
                "expected a validated non-null string."
            )
        if not isinstance(getattr(pred, "code", None), str) or len(pred.code) < 1:
            raise RuntimeError(
                "PredictRLM action adapter returned invalid code; "
                "expected a validated non-empty string."
            )

    def _record_action_generation_error(
        self,
        exc: BaseException,
        *,
        iteration: int,
        action_start_ns: int,
        lm_metadata: Any,
    ) -> None:
        self._write_telemetry_span(
            "rlm.action_generation.parse_error",
            iteration=iteration + 1,
            status=self._action_generation_error_status(exc),
            attributes=self._action_generation_error_attrs(
                exc,
                lm_metadata=lm_metadata,
            ),
            start_time_unix_nano=action_start_ns,
        )
        self._log_lifecycle(
            "rlm.action_generation.error",
            iteration=iteration + 1,
            error_type=type(exc).__name__,
        )

    def _record_action_generation_ok(
        self,
        pred: Any,
        *,
        iteration: int,
        action_start_ns: int,
        lm_hist_before_action: int,
        execution_timeout: float | None,
    ) -> Any:
        self._write_telemetry_span(
            "rlm.action_generation.ok",
            iteration=iteration + 1,
            start_time_unix_nano=action_start_ns,
            attributes={
                "has_code": True,
                "code_chars": len(getattr(pred, "code", "") or ""),
                "reasoning_chars": len(getattr(pred, "reasoning", "") or ""),
                **(
                    {"execution_timeout_seconds": execution_timeout}
                    if execution_timeout is not None
                    else {}
                ),
            },
        )
        self._log_lifecycle(
            "rlm.action_generation.ok",
            iteration=iteration + 1,
            code_chars=len(getattr(pred, "code", "") or ""),
            reasoning_chars=len(getattr(pred, "reasoning", "") or ""),
            execution_timeout_seconds=execution_timeout,
        )
        self._last_action_lm_usage = usage_since(dspy.settings.lm, lm_hist_before_action)
        return lm_finish_since(dspy.settings.lm, lm_hist_before_action)

    def _prepare_iteration_execution(self, pred: Any, iteration: int) -> tuple[str, bool]:
        code = pred.code or ""
        code = strip_code_fences(code)
        self._write_generated_code_event(iteration=iteration, pred=pred, code=code)
        iteration_log_open = self._begin_iteration_log(
            iteration=iteration,
            pred=pred,
            code=code,
        )

        from dspy.primitives.repl_types import REPLEntry

        self._partial_pending_entry = REPLEntry(
            reasoning=getattr(pred, "reasoning", "") or "",
            code=code,
            output="",
        )
        self._partial_pending_start = time.perf_counter()
        self._export_current_trace("in_progress")
        return code, iteration_log_open

    def _log_iteration_execute_start(self, repl: Any, *, iteration: int, code: str) -> float:
        execute_start = time.perf_counter()
        self._log_lifecycle(
            "rlm.execute.start",
            backend=self._interpreter_backend_label(repl),
            iteration=iteration + 1,
            code_chars=len(code),
        )
        return execute_start

    def _log_iteration_execute_fatal(
        self,
        repl: Any,
        *,
        iteration: int,
        execute_start: float,
        exc: BaseException,
    ) -> None:
        self._log_lifecycle(
            "rlm.execute.fatal",
            backend=self._interpreter_backend_label(repl),
            iteration=iteration + 1,
            duration_ms=ms_since(execute_start),
            error_type=type(exc).__name__,
        )

    def _format_iteration_execute_error(
        self,
        repl: Any,
        *,
        iteration: int,
        execute_start: float,
        code: str,
        exc: BaseException,
    ) -> str:
        result = _format_execution_error(code, exc)
        self._log_lifecycle(
            "rlm.execute.error",
            backend=self._interpreter_backend_label(repl),
            iteration=iteration + 1,
            duration_ms=ms_since(execute_start),
            error_type=type(exc).__name__,
        )
        return result

    def _log_iteration_execute_ok(
        self,
        repl: Any,
        *,
        iteration: int,
        execute_start: float,
        result: Any,
    ) -> None:
        self._log_lifecycle(
            "rlm.execute.ok",
            backend=self._interpreter_backend_label(repl),
            iteration=iteration + 1,
            duration_ms=ms_since(execute_start),
            final=isinstance(result, FinalOutput),
        )

    def _execute_iteration_code(
        self,
        repl: Any,
        *,
        code: str,
        input_args: dict[str, Any],
        iteration: int,
        iteration_log_open: bool,
        execution_timeout: float | None,
    ) -> Any:
        execute_start = self._log_iteration_execute_start(
            repl,
            iteration=iteration,
            code=code,
        )
        try:
            with (
                live_tool_call_logging(iteration_log_open),
                suppress_interpreter_result_logging(iteration_log_open),
            ):
                if (
                    getattr(self, "_submit_confirmation", None) is not None
                    and not getattr(self, "_pending_submit_confirmation", False)
                    and hasattr(repl, "defer_next_submit_finalization")
                ):
                    repl.defer_next_submit_finalization()
                if execution_timeout is None:
                    result = repl.execute(code, variables=dict(input_args))
                else:
                    result = repl.execute(
                        code,
                        variables=dict(input_args),
                        timeout=execution_timeout,
                    )
        except (SandboxFatalError, ExecutionFatalError) as exc:
            self._log_iteration_execute_fatal(
                repl,
                iteration=iteration,
                execute_start=execute_start,
                exc=exc,
            )
            raise
        except (CodeInterpreterError, SyntaxError) as exc:
            return self._format_iteration_execute_error(
                repl,
                iteration=iteration,
                execute_start=execute_start,
                code=code,
                exc=exc,
            )
        self._log_iteration_execute_ok(
            repl,
            iteration=iteration,
            execute_start=execute_start,
            result=result,
        )
        return result

    async def _aexecute_iteration_code(
        self,
        repl: Any,
        *,
        code: str,
        input_args: dict[str, Any],
        iteration: int,
        iteration_log_open: bool,
        execution_timeout: float | None,
    ) -> Any:
        execute_start = self._log_iteration_execute_start(
            repl,
            iteration=iteration,
            code=code,
        )
        recorder = self._evidence()
        operation_id = uuid.uuid4().hex
        if recorder is not None:
            await recorder.emit(
                RunEventKind.CODE_GENERATED,
                operation_id=operation_id,
                iteration=iteration + 1,
                code=code,
            )
        try:
            with (
                live_tool_call_logging(iteration_log_open),
                suppress_interpreter_result_logging(iteration_log_open),
            ):
                if (
                    getattr(self, "_submit_confirmation", None) is not None
                    and not getattr(self, "_pending_submit_confirmation", False)
                    and hasattr(repl, "defer_next_submit_finalization")
                ):
                    repl.defer_next_submit_finalization()
                if hasattr(repl, "aexecute"):
                    if execution_timeout is None:
                        result = await repl.aexecute(code, variables=dict(input_args))
                    else:
                        result = await repl.aexecute(
                            code,
                            variables=dict(input_args),
                            timeout=execution_timeout,
                        )
                else:
                    if execution_timeout is None:
                        result = repl.execute(code, variables=dict(input_args))
                    else:
                        result = repl.execute(
                            code,
                            variables=dict(input_args),
                            timeout=execution_timeout,
                        )
        except asyncio.CancelledError as exc:
            if recorder is not None:
                try:
                    await recorder.emit(
                        RunEventKind.CODE_EXECUTED,
                        operation_id=operation_id,
                        iteration=iteration + 1,
                        error_type=type(exc).__name__,
                        error=str(exc),
                        cancelled=True,
                        fatal=False,
                    )
                except BaseException as evidence_error:
                    setattr(exc, "evidence_error", evidence_error)
            raise
        except (SandboxFatalError, ExecutionFatalError) as exc:
            self._log_iteration_execute_fatal(
                repl,
                iteration=iteration,
                execute_start=execute_start,
                exc=exc,
            )
            if recorder is not None:
                await recorder.emit(
                    RunEventKind.CODE_EXECUTED,
                    operation_id=operation_id,
                    iteration=iteration + 1,
                    error_type=type(exc).__name__,
                    error=str(exc),
                    fatal=True,
                )
            raise
        except Exception as exc:
            if getattr(exc, "_predict_rlm_input_adapter_after_execution", False):
                self._log_iteration_execute_fatal(
                    repl,
                    iteration=iteration,
                    execute_start=execute_start,
                    exc=exc,
                )
                if recorder is not None:
                    await recorder.emit(
                        RunEventKind.CODE_EXECUTED,
                        operation_id=operation_id,
                        iteration=iteration + 1,
                        error_type=type(exc).__name__,
                        error=str(exc),
                        fatal=True,
                    )
                raise
            output = self._format_iteration_execute_error(
                repl,
                iteration=iteration,
                execute_start=execute_start,
                code=code,
                exc=exc,
            )
            if recorder is not None:
                await recorder.emit(
                    RunEventKind.CODE_EXECUTED,
                    operation_id=operation_id,
                    iteration=iteration + 1,
                    output=output,
                    error_type=type(exc).__name__,
                    error=str(exc),
                    fatal=False,
                )
            return output
        self._log_iteration_execute_ok(
            repl,
            iteration=iteration,
            execute_start=execute_start,
            result=result,
        )
        if recorder is not None:
            await recorder.emit(
                RunEventKind.CODE_EXECUTED,
                operation_id=operation_id,
                iteration=iteration + 1,
                output=str(result),
                final=isinstance(result, FinalOutput),
            )
        return result

    def _complete_iteration_execution(
        self,
        pred: Any,
        *,
        code: str,
        result: Any,
        history: Any,
        output_field_names: list[str],
        lm_metadata: Any,
    ) -> Any:
        if not isinstance(result, FinalOutput):
            self._emit_iteration_output_log(result)

        if _PARENT_TAKES_CODE:
            step_result = self._process_execution_result(
                pred, code, result, history, output_field_names
            )
        else:
            step_result = self._process_execution_result(
                pred, result, history, output_field_names
            )
        if isinstance(step_result, dspy.Prediction):
            self._emit_iteration_submit_log(step_result, output_field_names)
        else:
            self._partial_history = step_result
        self._partial_pending_entry = None
        self._partial_pending_start = None
        self._last_action_lm_metadata = lm_metadata
        return step_result

    def _execute_iteration(
        self,
        repl,
        variables,
        history,
        iteration,
        input_args,
        output_field_names,
    ):
        """Execute one synchronous RLM iteration with PredictRLM validation."""
        variables_info, action_start_ns = self._begin_action_generation(
            variables,
            iteration,
        )
        try:
            lm_hist_before_action = snapshot_lm_history_len(dspy.settings.lm)
            run_ctx = current_run_context()
            generate_action = (
                run_ctx.state.get("generate_action", self.generate_action)
                if run_ctx is not None
                else self.generate_action
            )
            pred = generate_action(
                variables_info=variables_info,
                repl_history=history,
                iteration=f"{iteration + 1}/{self.max_iterations}",
            )
            self._validate_iteration_action(pred)
            execution_timeout = self._action_execution_timeout(pred)
        except BaseException as exc:
            lm_metadata = lm_completion_metadata_since(dspy.settings.lm, lm_hist_before_action)
            self._record_action_generation_error(
                exc,
                iteration=iteration,
                action_start_ns=action_start_ns,
                lm_metadata=lm_metadata,
            )
            raise
        lm_metadata = self._record_action_generation_ok(
            pred,
            iteration=iteration,
            action_start_ns=action_start_ns,
            lm_hist_before_action=lm_hist_before_action,
            execution_timeout=execution_timeout,
        )
        code, iteration_log_open = self._prepare_iteration_execution(pred, iteration)

        try:
            result = self._execute_iteration_code(
                repl,
                code=code,
                input_args=input_args,
                iteration=iteration,
                iteration_log_open=iteration_log_open,
                execution_timeout=execution_timeout,
            )
            return self._complete_iteration_execution(
                pred,
                code=code,
                result=result,
                history=history,
                output_field_names=output_field_names,
                lm_metadata=lm_metadata,
            )
        finally:
            self._finish_iteration_log(iteration_log_open)

    async def _aexecute_iteration(
        self,
        repl,
        variables,
        history,
        iteration,
        input_args,
        output_field_names,
    ):
        """Override parent to use non-blocking aexecute() on the interpreter.

        The parent calls repl.execute() synchronously which blocks the event
        loop, serializing all concurrent rollouts. This override uses
        repl.aexecute() (available on JspiBackend) so the event loop
        stays free for other coroutines.
        """
        bound_iteration = getattr(self._execute_iteration, "__func__", None)
        if bound_iteration is not _KERNEL_EXECUTE_ITERATION:
            return self._execute_iteration(
                repl,
                variables,
                history,
                iteration,
                input_args,
                output_field_names,
            )
        variables_info, action_start_ns = self._begin_action_generation(
            variables,
            iteration,
        )
        try:
            lm_hist_before_action = snapshot_lm_history_len(dspy.settings.lm)
            run_ctx = current_run_context()
            generate_action = (
                run_ctx.state.get("generate_action", self.generate_action)
                if run_ctx is not None
                else self.generate_action
            )
            action_kwargs = {
                "variables_info": variables_info,
                "repl_history": history,
                "iteration": f"{iteration + 1}/{self.max_iterations}",
            }
            if inspect.iscoroutinefunction(getattr(generate_action, "acall", None)):
                pred = await generate_action.acall(**action_kwargs)
            else:
                pred = generate_action(**action_kwargs)
            self._validate_iteration_action(pred)
            execution_timeout = self._action_execution_timeout(pred)
        except BaseException as exc:
            lm_metadata = lm_completion_metadata_since(dspy.settings.lm, lm_hist_before_action)
            self._record_action_generation_error(
                exc,
                iteration=iteration,
                action_start_ns=action_start_ns,
                lm_metadata=lm_metadata,
            )
            raise
        lm_metadata = self._record_action_generation_ok(
            pred,
            iteration=iteration,
            action_start_ns=action_start_ns,
            lm_hist_before_action=lm_hist_before_action,
            execution_timeout=execution_timeout,
        )
        code, iteration_log_open = self._prepare_iteration_execution(pred, iteration)

        try:
            result = await self._aexecute_iteration_code(
                repl,
                code=code,
                input_args=input_args,
                iteration=iteration,
                iteration_log_open=iteration_log_open,
                execution_timeout=execution_timeout,
            )
            return self._complete_iteration_execution(
                pred,
                code=code,
                result=result,
                history=history,
                output_field_names=output_field_names,
                lm_metadata=lm_metadata,
            )
        finally:
            self._finish_iteration_log(iteration_log_open)

    def _copy_with_run_extract(self) -> PredictRLM:
        run_ctx = current_run_context()
        extract = (
            run_ctx.state.get("extract", self.extract)
            if run_ctx is not None
            else self.extract
        )
        run_rlm = copy(self)
        run_rlm.extract = extract
        return run_rlm

    def _extract_fallback_for_run(
        self,
        variables: list[Any],
        history: Any,
        output_field_names: list[str],
    ) -> dspy.Prediction:
        run_rlm = self._copy_with_run_extract()
        return run_rlm._extract_fallback(
            variables,
            history,
            output_field_names,
        )

    async def _aextract_fallback_for_run(
        self,
        variables: list[Any],
        history: Any,
        output_field_names: list[str],
    ) -> dspy.Prediction:
        run_rlm = self._copy_with_run_extract()
        bound_fallback = getattr(run_rlm._extract_fallback, "__func__", None)
        if bound_fallback is not _DSPY_EXTRACT_FALLBACK:
            return run_rlm._extract_fallback(
                variables,
                history,
                output_field_names,
            )
        return await run_rlm._aextract_fallback(
            variables,
            history,
            output_field_names,
        )

    def _prepare_file_io(
        self, input_args: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Scan signature for file fields and build file plan.

        Returns (file_plan, transformed_input_args). file_plan is None if
        no file fields are present.
        """
        input_file_fields, output_file_fields = scan_file_fields(self.signature)
        if not input_file_fields and not output_file_fields:
            return None, input_args

        file_plan = build_file_plan(
            input_args,
            input_file_fields,
            output_file_fields,
            self._output_dir,
        )
        if not file_plan:
            return None, input_args

        # Validate input files exist, then replace File values with sandbox path strings
        transformed = dict(input_args)
        for field_name in input_file_fields:
            value = transformed.get(field_name)
            if value is None:
                continue
            kind = input_file_fields[field_name]
            if kind == "list_file":
                for item in value:
                    if not os.path.isfile(item.path):
                        raise FileNotFoundError(
                            f"Input file for field '{field_name}' not found: {item.path}"
                        )
                transformed[field_name] = [
                    f"/sandbox/input/{field_name}/{os.path.basename(item.path)}"
                    for item in value
                ]
            elif kind == "file":
                if not os.path.isfile(value.path):
                    raise FileNotFoundError(
                        f"Input file for field '{field_name}' not found: {value.path}"
                    )
                basename = os.path.basename(value.path)
                transformed[field_name] = f"/sandbox/input/{field_name}/{basename}"
            elif kind in ("dir", "list_dir"):
                if not os.path.isdir(value.path):
                    raise FileNotFoundError(
                        f"Input directory for field '{field_name}' not found: {value.path}"
                    )
                transformed[field_name] = f"/sandbox/input/{field_name}"

        # Remove output file fields from input_args (they aren't RLM inputs)
        for field_name in output_file_fields:
            transformed.pop(field_name, None)

        return file_plan, transformed

    def _setup_sandbox_files(
        self, repl: LegacyExecutionBackend, file_plan: dict[str, Any]
    ) -> None:
        """Mount input files, skill modules, and create output dirs in the sandbox."""
        self._log_lifecycle(
            "sandbox.files.setup.start",
            backend=self._interpreter_backend_label(repl),
            mounts=len(file_plan["mounts"]),
            output_dirs=len(file_plan["output_dirs"]),
            skill_modules=len(self._skill_modules),
        )
        if hasattr(repl, "_ensure_deno_process"):
            repl._ensure_deno_process()

        for host_path, virtual_path in file_plan["mounts"]:
            repl.mount_file_at(host_path, virtual_path)
            self._log_lifecycle(
                "sandbox.files.mount",
                backend=self._interpreter_backend_label(repl),
                virtual_path=virtual_path,
            )

        for virtual_dir in file_plan["output_dirs"]:
            repl.mkdir_p(virtual_dir)
            self._log_lifecycle(
                "sandbox.files.mkdir",
                backend=self._interpreter_backend_label(repl),
                virtual_path=virtual_dir,
            )

        # Mount skill modules into /sandbox/lib/ so the RLM can import them
        if self._skill_modules:
            repl.mkdir_p("/sandbox/lib")
            for mod_name, host_path in self._skill_modules.items():
                repl.mount_file_at(host_path, f"/sandbox/lib/{mod_name}.py")
                self._log_lifecycle(
                    "sandbox.files.skill_module_mount",
                    backend=self._interpreter_backend_label(repl),
                    module=mod_name,
                    virtual_path=f"/sandbox/lib/{mod_name}.py",
                )
            repl.execute("import sys; sys.path.insert(0, '/sandbox/lib')")
        self._log_lifecycle(
            "sandbox.files.setup.ok",
            backend=self._interpreter_backend_label(repl),
        )

    def _sync_output_files(
        self,
        repl: LegacyExecutionBackend,
        prediction: dspy.Prediction,
        output_file_fields: dict[str, str],
        file_plan: dict[str, Any],
    ) -> None:
        """Sync output files from sandbox to host and populate prediction fields.

        The RLM submits the sandbox path(s) it wrote to. We sync those files
        back to the host and replace the prediction value with File objects.
        """
        self._log_lifecycle(
            "sandbox.files.sync_outputs.start",
            backend=self._interpreter_backend_label(repl),
            fields=len(output_file_fields),
        )
        for field_name, kind in output_file_fields.items():
            info = file_plan["output_field_map"][field_name]
            host_dir = info["host_dir"]
            virtual_dir = info["virtual_dir"]

            # The RLM submitted a path string (or the extract LLM produced one).
            # It may also be a File model — extract .path if so.
            submitted_path = getattr(prediction, field_name, None)
            if isinstance(submitted_path, File):
                submitted_path = submitted_path.path
            if submitted_path and isinstance(submitted_path, str):
                submitted_path = submitted_path.strip()

            if kind == "file":
                # RLM submitted a specific file path like "/sandbox/output/excel/result.xlsx"
                if (
                    submitted_path
                    and isinstance(submitted_path, str)
                    and submitted_path.startswith("/sandbox/")
                ):
                    basename = os.path.basename(submitted_path)
                    host_path = os.path.join(host_dir, basename)
                    os.makedirs(host_dir, exist_ok=True)
                    repl.sync_file_to(submitted_path, host_path)
                    self._log_lifecycle(
                        "sandbox.files.sync_output",
                        backend=self._interpreter_backend_label(repl),
                        field=field_name,
                        virtual_path=submitted_path,
                        host_path=host_path,
                    )
                    setattr(prediction, field_name, File(path=host_path))
                else:
                    # Fallback: list the output dir and sync everything
                    virtual_files = repl.list_dir(virtual_dir)
                    self._log_lifecycle(
                        "sandbox.files.list_output_dir",
                        backend=self._interpreter_backend_label(repl),
                        field=field_name,
                        virtual_dir=virtual_dir,
                        files=len(virtual_files),
                    )
                    if virtual_files:
                        os.makedirs(host_dir, exist_ok=True)
                        for vpath in virtual_files:
                            rel = os.path.relpath(vpath, virtual_dir)
                            hp = os.path.join(host_dir, rel)
                            os.makedirs(os.path.dirname(hp), exist_ok=True)
                            repl.sync_file_to(vpath, hp)
                            self._log_lifecycle(
                                "sandbox.files.sync_output",
                                backend=self._interpreter_backend_label(repl),
                                field=field_name,
                                virtual_path=vpath,
                                host_path=hp,
                            )
                        if len(virtual_files) == 1:
                            rel = os.path.relpath(virtual_files[0], virtual_dir)
                            hp = os.path.join(host_dir, rel)
                        else:
                            hp = host_dir
                        setattr(prediction, field_name, File(path=hp))

            elif kind == "list_file":
                # Sync all files from the output dir, return list[File]
                virtual_files = repl.list_dir(virtual_dir)
                result_files: list[File] = []
                self._log_lifecycle(
                    "sandbox.files.list_output_dir",
                    backend=self._interpreter_backend_label(repl),
                    field=field_name,
                    virtual_dir=virtual_dir,
                    files=len(virtual_files),
                )
                if virtual_files:
                    os.makedirs(host_dir, exist_ok=True)
                    for vpath in virtual_files:
                        rel = os.path.relpath(vpath, virtual_dir)
                        hp = os.path.join(host_dir, rel)
                        os.makedirs(os.path.dirname(hp), exist_ok=True)
                        repl.sync_file_to(vpath, hp)
                        self._log_lifecycle(
                            "sandbox.files.sync_output",
                            backend=self._interpreter_backend_label(repl),
                            field=field_name,
                            virtual_path=vpath,
                            host_path=hp,
                        )
                        result_files.append(File(path=hp))
                setattr(prediction, field_name, result_files)
        self._log_lifecycle(
            "sandbox.files.sync_outputs.ok",
            backend=self._interpreter_backend_label(repl),
        )

    def _build_iteration_outcome(
        self,
        result: Any,
        history: Any,
        iteration: int,
        iter_start: float,
        action_lm_metadata: Any | None,
    ) -> tuple[IterationStep, bool]:
        """Build trace/callback payload for a completed RLM iteration."""
        from dspy.primitives.repl_types import REPLEntry, REPLHistory

        new_history = result if isinstance(result, REPLHistory) else None
        if new_history and len(new_history.entries) > len(history.entries):
            entry = new_history.entries[-1]
        elif isinstance(result, dspy.Prediction) and hasattr(result, "trajectory"):
            traj = result.trajectory
            entry_data = traj[-1] if traj else {}
            entry = REPLEntry(
                reasoning=entry_data.get("reasoning", ""),
                code=entry_data.get("code", ""),
                output=entry_data.get("output", ""),
            )
        else:
            entry = None

        full_output = entry.output if entry else ""
        prompt_output = _shorten_repl_output(full_output, self.max_output_chars)

        predict_calls = drain_predict_calls()
        iteration_usage = self._build_iteration_usage(predict_calls)

        return (
            IterationStep(
                iteration=iteration + 1,
                reasoning=entry.reasoning if entry else "",
                code=entry.code if entry else "",
                output=prompt_output,
                untruncated_output=full_output,
                error=_output_indicates_error(full_output),
                duration_ms=ms_since(iter_start),
                tool_calls=drain_tool_calls(),
                predict_calls=predict_calls,
                lm=action_lm_metadata,
                usage=iteration_usage,
            ),
            isinstance(result, dspy.Prediction),
        )

    def _build_iteration_usage(self, predict_calls: list[Any]) -> LMUsage:
        """Per-iteration usage: main action-LM call plus this turn's sub predict() calls."""
        main = getattr(self, "_last_action_lm_usage", None) or TokenUsage()
        self._last_action_lm_usage = None
        sub = TokenUsage()
        for group in predict_calls:
            sub += group.total_usage
        return LMUsage(main=main, sub=sub)

    def _forward_traced(
        self, file_plan: dict[str, Any] | None, **input_args: Any
    ) -> dspy.Prediction:
        """Execute forward() with tracing and optional file I/O."""
        _, output_file_fields = scan_file_fields(self.signature) if file_plan else (None, {})

        orig_action, orig_extract = self.generate_action, self.extract
        if file_plan:
            self.generate_action, self.extract = self._build_signatures_with_files(
                file_plan["instructions"]
            )

        run_start = time.perf_counter()
        lm = dspy.settings.lm
        sub_lm = self._active_sub_lm()
        # Per-RLM instance ``lm`` has its own history (via ``lm.copy()``
        # in ``__init__``) so these snapshots are isolated from other
        # concurrent PredictRLM runs. No track_usage context needed —
        # DSPy's BaseLM stamps cost and tokens into history on every call.
        lm_hist_start = snapshot_lm_history_len(lm)
        sub_hist_start = snapshot_lm_history_len(sub_lm) if sub_lm and sub_lm is not lm else 0
        steps: list[IterationStep] = []
        self._begin_telemetry_execution()

        try:
            self._validate_inputs(input_args)
            output_field_names = list(self.signature.output_fields.keys())
            execution_tools = self._prepare_execution_tools()
            variables = self._build_variables(**input_args)

            ctx_kwargs = (
                dict(execution_tools=execution_tools, file_plan=file_plan)
                if file_plan
                else dict(execution_tools=execution_tools)
            )
            with self._interpreter_context(**ctx_kwargs) as repl:
                if file_plan:
                    self._setup_sandbox_files(repl, file_plan)

                from dspy.primitives.repl_types import REPLHistory

                predict_token = init_predict_call_collector()
                tool_token = init_tool_call_collector()

                try:
                    status = "max_iterations"
                    history = REPLHistory(max_output_chars=self.max_output_chars)
                    pending_submit_confirmation = False
                    # Reset partial-trajectory capture for this forward() call
                    # so crash-recovery consumers (see e.g. the spreadbench
                    # optimizer's _recover_partial_trace) can read the last
                    # committed REPL state if an iteration hangs or errors.
                    self._partial_history = history
                    self._partial_pending_entry = None
                    self._partial_pending_start = None
                    self._trace_export_context = _TraceExportContext(
                        steps=steps,
                        lm=lm,
                        sub_lm=sub_lm,
                        lm_hist_start=lm_hist_start,
                        sub_hist_start=sub_hist_start,
                        run_start=run_start,
                    )
                    self._export_current_trace("in_progress")
                    call_id = ACTIVE_CALL_ID.get()

                    for iteration in range(self.max_iterations):
                        iter_start = time.perf_counter()
                        with self._iteration_callback_scope(call_id, iteration) as state:
                            self._pending_submit_confirmation = pending_submit_confirmation
                            try:
                                result = self._execute_iteration(
                                    repl,
                                    variables,
                                    history,
                                    iteration,
                                    input_args,
                                    output_field_names,
                                )
                            finally:
                                self._pending_submit_confirmation = False

                            if isinstance(result, dspy.Prediction):
                                confirmation_history = self._maybe_request_submit_confirmation(
                                    prediction=result,
                                    history=history,
                                    input_args=input_args,
                                    output_field_names=output_field_names,
                                    iteration=iteration,
                                    pending=pending_submit_confirmation,
                                )
                                if confirmation_history is not None:
                                    result = confirmation_history
                                    self._partial_history = confirmation_history
                                    pending_submit_confirmation = True
                                else:
                                    pending_submit_confirmation = False
                            elif pending_submit_confirmation:
                                pending_submit_confirmation = False

                            action_lm_metadata = getattr(
                                self, "_last_action_lm_metadata", None
                            )
                            self._last_action_lm_metadata = None
                            state.step, state.is_final = self._build_iteration_outcome(
                                result,
                                history,
                                iteration,
                                iter_start,
                                action_lm_metadata,
                            )
                            steps.append(state.step)
                            self._export_current_trace("in_progress")

                        if state.is_final:
                            status = "completed"
                            prediction = result
                            if output_file_fields:
                                self._sync_output_files(
                                    repl, prediction, output_file_fields, file_plan
                                )
                            break
                        history = result
                    else:
                        prediction = self._extract_fallback_for_run(
                            variables, history, output_field_names
                        )
                        if output_file_fields:
                            self._sync_output_files(
                                repl, prediction, output_file_fields, file_plan
                            )

                    prediction.trace = self._build_run_trace(
                        status=status,
                        steps=steps,
                        lm=lm,
                        sub_lm=sub_lm,
                        lm_hist_start=lm_hist_start,
                        sub_hist_start=sub_hist_start,
                        run_start=run_start,
                    )
                    self._export_run_trace(prediction.trace)
                    return prediction
                finally:
                    reset_tool_call_collector(tool_token)
                    reset_predict_call_collector(predict_token)
        except BaseException as exc:
            try:
                trace = self._build_run_trace(
                    status="error",
                    steps=steps,
                    lm=lm,
                    sub_lm=sub_lm,
                    lm_hist_start=lm_hist_start,
                    sub_hist_start=sub_hist_start,
                    run_start=run_start,
                )
                setattr(exc, "trace", trace)
                self._export_run_trace(trace)
            except Exception:
                pass
            raise
        finally:
            self._clear_telemetry_execution()
            self._trace_export_context = None
            if file_plan:
                self.generate_action, self.extract = orig_action, orig_extract

    async def _aforward_traced(
        self, file_plan: dict[str, Any] | None, **input_args: Any
    ) -> dspy.Prediction:
        """Execute aforward() with tracing and optional file I/O."""
        _, output_file_fields = scan_file_fields(self.signature) if file_plan else (None, {})

        run_ctx = current_run_context()

        run_start = time.perf_counter()
        lm = dspy.settings.lm
        sub_lm = self._active_sub_lm()
        # Snapshot lm.history lengths at run start so ``usage_since`` can
        # sum the delta at end. Per-RLM ``lm.copy()`` in __init__ means
        # each PredictRLM instance has its OWN history — concurrent runs
        # don't cross-contaminate. DSPy's BaseLM populates
        # ``entry["cost"]`` from ``_hidden_params["response_cost"]``, so
        # ``usage_since`` returns LiteLLM-computed authoritative cost.
        lm_hist_start = snapshot_lm_history_len(lm)
        sub_hist_start = snapshot_lm_history_len(sub_lm) if sub_lm and sub_lm is not lm else 0
        steps: list[IterationStep] = []
        self._begin_telemetry_execution()

        try:
            self._validate_inputs(input_args)
            output_field_names = list(self.signature.output_fields.keys())
            execution_tools = self._prepare_execution_tools()

            async with self._execution_session(execution_tools=execution_tools) as repl:
                if run_ctx is not None:
                    await self._bind_runtime_inputs(run_ctx)
                    await self._setup_runtime_modules(run_ctx)
                    await self._reserve_runtime_outputs(run_ctx)
                    input_args.update(
                        {
                            name: binding.prepared.model_value
                            for name, binding in run_ctx.input_bindings.items()
                        }
                    )
                    for field_name in run_ctx.output_reservations:
                        input_args.pop(field_name, None)
                    self._configure_run_predictors(run_ctx, None)
                variables = self._build_variables(**input_args)

                from dspy.primitives.repl_types import REPLHistory

                predict_token = init_predict_call_collector()
                tool_token = init_tool_call_collector()

                try:
                    status = "max_iterations"
                    history = REPLHistory(max_output_chars=self.max_output_chars)
                    pending_submit_confirmation = False
                    # Reset partial-trajectory capture for this aforward() call
                    # so crash-recovery consumers (see e.g. the spreadbench
                    # optimizer's _recover_partial_trace) can read the last
                    # committed REPL state if an iteration hangs or errors.
                    self._partial_history = history
                    self._partial_pending_entry = None
                    self._partial_pending_start = None
                    self._trace_export_context = _TraceExportContext(
                        steps=steps,
                        lm=lm,
                        sub_lm=sub_lm,
                        lm_hist_start=lm_hist_start,
                        sub_hist_start=sub_hist_start,
                        run_start=run_start,
                    )
                    self._export_current_trace("in_progress")
                    call_id = ACTIVE_CALL_ID.get()

                    for iteration in range(self.max_iterations):
                        iter_start = time.perf_counter()
                        async with self._iteration_callback_scope_for_run(
                            call_id, iteration
                        ) as state:
                            self._pending_submit_confirmation = pending_submit_confirmation
                            try:
                                result = await self._aexecute_iteration(
                                    repl,
                                    variables,
                                    history,
                                    iteration,
                                    input_args,
                                    output_field_names,
                                )
                            finally:
                                self._pending_submit_confirmation = False

                            if isinstance(result, dspy.Prediction):
                                confirmation_history = self._maybe_request_submit_confirmation(
                                    prediction=result,
                                    history=history,
                                    input_args=input_args,
                                    output_field_names=output_field_names,
                                    iteration=iteration,
                                    pending=pending_submit_confirmation,
                                )
                                if confirmation_history is not None:
                                    result = confirmation_history
                                    self._partial_history = confirmation_history
                                    pending_submit_confirmation = True
                                else:
                                    pending_submit_confirmation = False
                            elif pending_submit_confirmation:
                                pending_submit_confirmation = False

                            action_lm_metadata = getattr(
                                self, "_last_action_lm_metadata", None
                            )
                            self._last_action_lm_metadata = None
                            state.step, state.is_final = self._build_iteration_outcome(
                                result,
                                history,
                                iteration,
                                iter_start,
                                action_lm_metadata,
                            )
                            steps.append(state.step)
                            recorder = self._evidence()
                            if recorder is not None:
                                await recorder.emit(
                                    RunEventKind.ITERATION_RECORDED,
                                    step=state.step.model_dump(mode="python"),
                                )
                            self._export_current_trace("in_progress")

                        if state.is_final:
                            status = "completed"
                            prediction = result
                            if run_ctx is not None and run_ctx.output_reservations:
                                await self._materialize_runtime_outputs(run_ctx, prediction)
                            break
                        history = result
                    else:
                        prediction = await self._aextract_fallback_for_run(
                            variables, history, output_field_names
                        )
                        if run_ctx is not None and run_ctx.output_reservations:
                            await self._materialize_runtime_outputs(run_ctx, prediction)

                    prediction.trace = self._build_run_trace(
                        status=status,
                        steps=steps,
                        lm=lm,
                        sub_lm=sub_lm,
                        lm_hist_start=lm_hist_start,
                        sub_hist_start=sub_hist_start,
                        run_start=run_start,
                    )
                    self._export_run_trace(prediction.trace)
                    return prediction
                finally:
                    reset_tool_call_collector(tool_token)
                    reset_predict_call_collector(predict_token)
        except BaseException as exc:
            try:
                trace = self._build_run_trace(
                    status="error",
                    steps=steps,
                    lm=lm,
                    sub_lm=sub_lm,
                    lm_hist_start=lm_hist_start,
                    sub_hist_start=sub_hist_start,
                    run_start=run_start,
                )
                setattr(exc, "trace", trace)
                self._export_run_trace(trace)
            except Exception:
                pass
            raise
        finally:
            self._clear_telemetry_execution()
            self._trace_export_context = None

    def _build_signatures_with_files(
        self, file_instructions: str
    ) -> tuple[dspy.Predict, dspy.Predict]:
        """Build signatures with per-call file instructions.

        File output fields are replaced with str in the signature so the RLM
        submits a path string, not a JSON object. The framework wraps the
        path in File after syncing.
        """
        scoped_prompt_signature = _RUN_PROMPT_SIGNATURE.get()
        prompt_signature = (
            scoped_prompt_signature[1]
            if scoped_prompt_signature is not None and scoped_prompt_signature[0] is self
            else self.signature
        )
        return self._build_signatures_for_prompt_signature(prompt_signature, file_instructions)

    def _build_signatures_for_prompt_signature(
        self,
        prompt_signature: type[dspy.Signature],
        file_instructions: str,
    ) -> tuple[dspy.Predict, dspy.Predict]:
        # Replace file-typed fields with str/list[str] for the RLM's view
        modified_sig = prompt_signature
        for name, field in prompt_signature.output_fields.items():
            descriptor = FieldDescriptor(name, field.annotation)
            if descriptor.matches(File):
                replacement = descriptor.replace_type(str)
                modified_sig = modified_sig.with_updated_fields(
                    name,
                    desc=f"{field.json_schema_extra.get('desc', '')} "
                    f"(submit the sandbox path you wrote to)",
                    type_=replacement,
                )
        for name, field in prompt_signature.input_fields.items():
            descriptor = FieldDescriptor(name, field.annotation)
            if descriptor.matches(File):
                replacement = descriptor.replace_type(str)
                modified_sig = modified_sig.with_updated_fields(
                    name,
                    desc=field.json_schema_extra.get("desc", ""),
                    type_=replacement,
                )

        raw_tools = self._runtime_tool_functions()
        action_sig, extract_sig = build_rlm_signatures(
            modified_sig,
            PREDICT_RLM_INSTRUCTIONS,
            raw_tools,
            self._format_tool_docs,
            skill_instructions=self._runtime_instructions(),
            file_instructions=file_instructions,
            model_execution_timeout=self._model_execution_timeout,
            max_output_chars=self.max_output_chars,
        )
        return dspy.Predict(action_sig), dspy.Predict(extract_sig)

    def _build_signatures(self) -> tuple[Signature, Signature]:
        """Build action and extract signatures with predict documentation."""
        raw_tools = self._runtime_tool_functions()
        return build_rlm_signatures(
            self.signature,
            PREDICT_RLM_INSTRUCTIONS,
            raw_tools,
            self._format_tool_docs,
            skill_instructions=self._runtime_instructions(),
            model_execution_timeout=self._model_execution_timeout,
            max_output_chars=self.max_output_chars,
        )


_KERNEL_FORWARD_TRACED = PredictRLM._forward_traced
_KERNEL_AFORWARD_TRACED = PredictRLM._aforward_traced
_KERNEL_EXECUTE_ITERATION = PredictRLM._execute_iteration
