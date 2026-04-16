"""PredictRLM - RLM subclass with predict tool for DSPy signatures."""

from __future__ import annotations

import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, List, Literal, Optional

import dspy
from dspy.primitives.code_interpreter import CodeInterpreter
from pydantic import create_model

from ._shared import build_rlm_signatures, format_tool_docs_full
from .files import File, build_file_plan, scan_file_fields
from .interpreter import JspiInterpreter
from .rlm_skills import Skill, merge_skills
from .trace import (
    IterationStep,
    LMUsage,
    RunTrace,
    _RawPredictCall,
    drain_predict_calls,
    drain_tool_calls,
    init_predict_call_collector,
    init_tool_call_collector,
    ms_since,
    record_predict_call,
    reset_predict_call_collector,
    reset_tool_call_collector,
    snapshot_lm_history_len,
    usage_since,
)

# Capture the real dspy.Image class at import time so type comparisons
# work even when tests patch predict_rlm.dspy.Image to a mock.
_ImageType = dspy.Image

if TYPE_CHECKING:
    from dspy.signatures.signature import Signature


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
            if field_name in required:
                fields[field_name] = (py_type, ...)
            else:
                fields[field_name] = (Optional[py_type], None)  # type: ignore[valid-type]

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

You work inside a Python REPL environment. Write code in ```repl blocks and it will be executed. You will see the output, then write more code based on what you learned. This is an iterative, interactive process — explore your data, plan your approach, and build up your answer step by step across multiple iterations.

## Workflow Overview

### Iterative loop
- Explore first: inspect sample inputs before extracting anything complicated.
- Plan each block: treat every iteration as a chance to learn, react to results, and refine.
- Persist intermediate work: save state to variables (`all_items`, `page_urls`, etc.) between iterations.

### Think before you build
On your first iteration, **explore before you extract**. Print samples of your input data — check types, lengths, what the data looks like. Understand what you're working with before writing any extraction logic. Even a quick `print(type(images), len(images))` or examining the first page can save you from going down the wrong path.

After exploring, plan your approach, then execute it in focused steps. Save intermediate results to variables so you can build on them across iterations.

Use `predict()` for anything requiring understanding of meaning — it's a powerful vision-language model. Use Python for computation, formatting, and aggregation.

## Environment & Tools

### Available interfaces

- Variables: {inputs} (your input data).
- `await predict(signature, *, instructions=None, **kwargs)` — your primary analysis tool (async, must await).
  - signature: str with type hints, e.g. `"page: dspy.Image, question: str -> answer: str"`.
  - For images, use `dspy.Image` type hint and pass URL or base64 string directly.
  - For fields that may be absent (the caller sometimes has no value to pass), use `Optional[T]` in the signature and pass `None` when missing. Example: `"context: Optional[str], question: str -> answer: str"` lets you call `predict(..., context=None, question=q)` when no context is available. Works for any type: `Optional[int]`, `Optional[dspy.Image]`, `Optional[list[str]]`.
  - instructions: optional str describing the task.
  - Returns a result that supports BOTH `result.answer` (attribute) and `result["answer"]` (subscript). Attribute access works for ALL field names including `items`, `keys`, `values` — no collision issues.
  - **Type contract is enforced on outputs**: if you declare `field: list[X]` (non-Optional) and the VLM fails to produce a valid list, `predict()` raises `RuntimeError` with a clear message telling you to simplify the schema or mark the field `Optional`. An empty list `[]` is ALWAYS valid — it means "VLM extracted nothing matching the criteria", which is distinct from "VLM failed to produce a list". Don't speculate "predict returned None" — check what you actually got back: `[]` is valid empty output, `None` is only possible for Optional-declared fields.
  - Capacity: ~400K tokens per call — you can pass substantial data.
- `print()` — ALWAYS print to see results (output is truncated per iteration — see "Managing state & output" below).
- `SUBMIT({final_output_names})` — submit final output when done. The argument names are EXACTLY the output field names from your task's signature (shown above) — use them verbatim. If the task outputs `-> result: SomeType`, call `SUBMIT(result=...)`, NOT `SUBMIT(items=...)` or `SUBMIT(output=...)`.
- Standard libraries: re, json, collections, math, asyncio, etc.

### Execution model
The REPL runs inside an async event loop — use `await` directly, not `asyncio.run()`.

## Managing state & output

Each iteration's printed output is captured and shown to you in subsequent iterations, but **truncated to ~5000 characters**. This is a hard limit — output beyond 5K chars is cut off.

**What persists fully:** Python variables. Everything you store in a variable (`all_items`, `page_urls`, etc.) survives intact across iterations. The runtime state is never lost.

**What gets truncated:** printed output. If you `print(big_list)` and it's 50K chars, you'll only see the first 5K in the next iteration's context.

**How to work effectively across iterations:**
- **Store in variables, print summaries:** `items.extend(new_items); print(f"{{len(items)}} items so far")` — the data is in `items` (full), you see the count (tiny).
- **Inspect with lightweight checks:** `print(type(x), len(x), list(x.keys())[:5])` instead of `print(x)`.
- **Slice large output:** `print(str(data)[:2000])` now, `print(str(data)[2000:4000])` next iteration.
- **Never rely on seeing full print output** — if you need the data, it should be in a variable.

## Using `predict()` effectively

### Core usage pattern
`predict()` is async and much faster when run concurrently. For independent calls, always use `asyncio.gather()`:

```repl
import asyncio
tasks = [predict("img: dspy.Image -> text: str", img=url) for url in page_urls]
results = await asyncio.gather(*tasks)
```

Use sequential iteration only when each step depends on previous results (e.g. accumulating context across pages).

### Typed outputs and schemas
Prefer typed outputs over JSON strings:

```repl
# Typed fields — attribute access works for ALL field names
result = await predict("page: dspy.Image -> title: str, date: str, amount: float", page=url)
print(result.title, result.date, result.amount)

# Lists
result = await predict("text: str -> keywords: list[str]", text=doc)
print(result.keywords)
```

For complex/nested structures, define a Pydantic `BaseModel` and reference it by name in the signature string. The type name IS resolved — `predict()` automatically finds your class definition, sends its schema to the LLM for structured output, and returns real model instances inside the result. Use `list[YourModel]`, NOT `list[dict]`:

```repl
from pydantic import BaseModel
from typing import Optional

class LineItem(BaseModel):
    description: str
    amount: float
    category: Optional[str] = None

# The class name "LineItem" in the signature string resolves to the class above.
result = await predict("page: dspy.Image -> items: list[LineItem]", page=url)
for item in result.items:                # attribute access works even for "items"
    print(item.description, item.amount)
```

Important: the model MUST extend `BaseModel` (not `dict`). Only include fields the model can actually produce from the input. Any fields you populate yourself after prediction must have defaults.

### Working with Pydantic values returned by `predict`

When your signature uses a custom type like `list[LineItem]`, the values inside the result are **real Pydantic instances** (not dicts):

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
print(f"Total inputs: {{len(documents)}}")
print(type(documents[0]))

orientation = await predict(
    "doc: dspy.Image -> overview: str, notable_elements: list[str]",
    instructions="Give a short description so I know what this document is about and what signals it contains.",
    doc=documents[0],
)
print(orientation.overview)
print(orientation.notable_elements[:2])
```

Swap `documents` for the variables your task provides — the idea is to look before you build.

### Sequential context-building
When each step depends on the last (requirements evolving across drafts, calculations that feed future pages, etc.), surface what you already know and pass it forward:

```repl
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

```repl
import asyncio
tasks = [
    predict(
        "page: dspy.Image -> signals: list[str], follow_up: Optional[str]",
        instructions="List the most important insights on this page and anything that needs deeper review.",
        page=img,
    )
    for img in images
]
page_results = await asyncio.gather(*tasks)

signals = []
for i, r in enumerate(page_results):
    for signal in r.signals:
        signals.append(f"Page {{i+1}}: {{signal}}")

summary = await predict(
    "signals: list[str] -> clusters: list[str], action_items: list[str]",
    instructions="Group related signals and surface concrete next steps.",
    signals=signals,
)
print(summary)
```

Adjust the signatures to match your task (classification, QA, calculations, simulations, planning, etc.); the structure—map, collect, synthesize—stays the same.


## Submitting results

Once you've verified the results, submit them. `SUBMIT` takes **one argument per output field**, named after each field. Use keyword arguments for clarity, especially when there are multiple outputs:

```repl
# For a signature like: ... -> items: list[str], total_count: int, sources: list[int]
print(f"Submitting {{len(all_items)}} items")
SUBMIT(
    items=all_items,
    total_count=len(all_items),
    sources=sorted(set(all_sources)),
)
```

Positional also works, in the order of the output fields: `SUBMIT(all_items, len(all_items), sorted(set(all_sources)))`. Prefer keyword form when there are more than two outputs — it's easier to spot a missing or swapped argument.

**SUBMIT accepts Pydantic instances OR plain dicts OR mixed.** Pass Pydantic objects directly, or nest them inside dicts/lists, whatever's natural:
```repl
# All of these work:
SUBMIT(result={{"items": [TaskItem(title="a"), TaskItem(title="b")]}})
SUBMIT(result=ExtractionResult(items=[TaskItem(title="a")]))
SUBMIT(result=[task.model_dump() for task in tasks])
```

Your work is only captured when you call `SUBMIT({final_output_names})`. The REPL loop keeps running until SUBMIT is called — it will NOT stop on its own. If the session ends without a SUBMIT call, nothing is returned and your work is lost. So always end with SUBMIT.

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

    # Captured context LM from forward() - used by predict() in thread pool
    _context_lm: dspy.LM | None = None

    def __init__(
        self,
        signature: type[Signature] | str,
        lm: dspy.LM | str | None = None,
        sub_lm: dspy.LM | str | None = None,
        max_iterations: int = 30,
        max_llm_calls: int = 50,
        max_output_chars: int = 100_000,
        verbose: bool = False,
        tools: dict[str, Callable[..., str]] | list[Callable] | None = None,
        interpreter: CodeInterpreter | None = None,
        allowed_domains: list[str] | None = None,
        skills: list[Skill] | None = None,
        debug: bool = False,
        output_dir: str | Path | None = None,
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
            verbose: Whether to log detailed execution info.
            tools: Additional tool functions callable from interpreter code.
                  Accepts a dict mapping names to callables, or a list of
                  callables (names inferred from __name__).
                  predict is added automatically if not already provided.
            interpreter: CodeInterpreter implementation to use. Defaults to
                        JSPI-enabled JspiInterpreter.
            allowed_domains: Domains/IPs the sandbox can access via network.
                            By default, no network access is allowed.
                            Example: ["api.example.com", "192.168.1.100:8080"]
            skills: List of Skill instances providing domain-specific instructions,
                   PyPI packages, and tools. Skills are merged automatically —
                   instructions are appended to the prompt, packages are installed
                   in the sandbox, and tools are exposed alongside predict().
            debug: If True, print REPL code, output, errors, and tool calls
                  to stderr in real-time. Useful for development.
            output_dir: Default host directory for output files. When set,
                       File output fields without an explicit path
                       are written here. If None, a temp directory is used.
        """
        # Store main LM (None means use context LM at call time)
        self._lm = dspy.LM(lm, cache=False) if isinstance(lm, str) else lm

        # Store sub_lm for tool creation (None means use context LM at call time)
        self._sub_lm = dspy.LM(sub_lm, cache=False) if isinstance(sub_lm, str) else sub_lm

        # Will be set during forward() to capture context LM for thread-safe predict calls
        self._context_lm = None

        # Store allowed_domains, debug, and output_dir for interpreter creation
        self._allowed_domains = allowed_domains
        self._debug = debug
        self._output_dir = str(output_dir) if output_dir else None

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

        def _wrap_images_from_sig(
            sig_obj: Any, kwargs: dict[str, Any]
        ) -> dict[str, Any]:
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
            # Priority: sub_lm > captured context LM > current settings.lm
            lm = rlm_instance._sub_lm or rlm_instance._context_lm or dspy.settings.lm
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
                        import logging

                        logging.getLogger(__name__).warning(
                            f"Failed to reconstruct Pydantic model '{name}' from schema: {e}"
                        )
            else:
                import logging

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
                    logging.getLogger(__name__).debug(
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
                import logging

                logger = logging.getLogger(__name__)

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
                    name: field.annotation
                    for name, field in sig.output_fields.items()
                }

            # Create predictor and run with the specified LM (async, no threads)
            predictor = dspy.Predict(sig)
            _call_start = time.perf_counter()
            _hist_len_before = snapshot_lm_history_len(lm)
            with dspy.context(lm=lm):
                prediction = await predictor.acall(**wrapped_kwargs)
            _call_duration = ms_since(_call_start)
            _call_usage = usage_since(lm, _hist_len_before)

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
            # Enforce type contract: if the VLM returned None for a field
            # whose declared type does NOT allow None (e.g. list[X] not
            # Optional[list[X]]), raise loudly. Silently coercing None→[] or
            # passing None through hides VLM failures behind ambiguous empty
            # values and causes models to speculate "predict returned None".
            result = {}
            for field in prediction.keys():
                if field.startswith("_"):
                    continue
                value = prediction[field]
                if value is None and field in output_field_annotations:
                    anno = output_field_annotations[field]
                    if not _allows_none(anno):
                        raise RuntimeError(
                            f"predict: VLM returned None for non-Optional output "
                            f"field {field!r} (declared type: {anno}). The VLM "
                            f"couldn't produce a valid value. Schema may be too "
                            f"complex — try simplifying the signature, or mark "
                            f"the field Optional (e.g. Optional[list[X]]) if None "
                            f"is acceptable."
                        )
                result[field] = _to_serializable(value)
            record_predict_call(_RawPredictCall(
                signature=signature,
                instructions=instructions,
                model=str(getattr(lm, "model", lm)),
                duration_ms=_call_duration,
                usage=_call_usage,
                input=kwargs,
                output=result,
            ))
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

        from .files import _is_list_annotation, is_file_type

        fields = []
        for name, field in self.signature.output_fields.items():
            annotation = getattr(field, "annotation", str)
            field_info = {"name": name}
            if is_file_type(annotation):
                field_info["type"] = "list" if _is_list_annotation(annotation) else "str"
            elif annotation in SIMPLE_TYPES:
                field_info["type"] = annotation.__name__
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
        from .files import _is_list_annotation, is_file_type

        raw = result.output
        if isinstance(raw, dict):
            for name in output_field_names:
                if name not in raw:
                    continue
                field = self.signature.output_fields.get(name)
                if field is None or not is_file_type(field.annotation):
                    continue
                val = raw[name]
                if _is_list_annotation(field.annotation):
                    if isinstance(val, list):
                        raw[name] = [
                            {"path": v} if isinstance(v, str) else v for v in val
                        ]
                elif isinstance(val, str):
                    raw[name] = {"path": val}

        return super()._process_final_output(result, output_field_names)

    @contextmanager
    def _interpreter_context(
        self,
        execution_tools: dict[str, Callable],
        file_plan: dict[str, Any] | None = None,
    ) -> Iterator[CodeInterpreter]:
        """Yield interpreter, creating JspiInterpreter if none provided."""
        if self._interpreter is not None:
            self._inject_execution_context(self._interpreter, execution_tools)
            yield self._interpreter
        else:
            extra_read = list(file_plan["read_paths"]) if file_plan else []
            extra_write = [file_plan["write_dir"]] if file_plan and file_plan["write_dir"] else None

            # Add module host paths to read permissions
            for mod_path in self._skill_modules.values():
                extra_read.append(mod_path)

            repl = JspiInterpreter(
                tools=execution_tools,
                output_fields=self._get_output_fields_info(),
                allowed_domains=self._allowed_domains,
                skill_packages=self._skill_packages or None,
                debug=self._debug,
                extra_read_paths=extra_read or None,
                extra_write_paths=extra_write,
            )
            try:
                yield repl
            finally:
                repl.shutdown()

    def _prepare_execution_tools(self) -> dict[str, Callable]:
        """Return only user-provided tools (including predict).

        Override parent to skip the default llm_query and llm_query_batched tools.
        """
        return {name: tool.func for name, tool in self._user_tools.items()}

    @contextmanager
    def _lm_context(self):
        """Set the configured LM as active context and capture it for predict() calls."""
        if self._lm is not None:
            with dspy.context(lm=self._lm):
                self._context_lm = dspy.settings.lm
                try:
                    yield
                finally:
                    self._context_lm = None
        else:
            self._context_lm = dspy.settings.lm
            try:
                yield
            finally:
                self._context_lm = None

    def forward(self, **kwargs: Any) -> dspy.Prediction:
        """Execute the RLM with captured context LM for thread-safe predict calls.

        Handles File inputs by mounting them into the sandbox,
        and File outputs by syncing them back to the host.
        """
        with self._lm_context():
            file_plan, kwargs = self._prepare_file_io(kwargs)
            return self._forward_traced(file_plan, **kwargs)

    def _build_run_trace(
        self,
        status: Literal["completed", "max_iterations", "error"],
        steps: list[IterationStep],
        lm: Any,
        sub_lm: Any,
        lm_hist_start: int,
        sub_hist_start: int,
        run_start: float,
    ) -> RunTrace:
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
            iterations=len(steps),
            max_iterations=self.max_iterations,
            duration_ms=ms_since(run_start),
            usage=LMUsage(main=main_usage, **({"sub": sub_usage} if sub_usage else {})),
            steps=steps,
        )

    async def aforward(self, **kwargs: Any) -> dspy.Prediction:
        """Async version of forward(). Sets the LM context for async execution.

        Uses aexecute() on the interpreter so asyncio.gather can run
        multiple rollouts concurrently.
        """
        self._context_lm = dspy.settings.lm

        try:
            file_plan, kwargs = self._prepare_file_io(kwargs)
            with self._lm_context():
                return await self._aforward_traced(file_plan, **kwargs)
        finally:
            self._context_lm = None

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
        repl.aexecute() (available on JspiInterpreter) so the event loop
        stays free for other coroutines.
        """
        from dspy.predict.rlm import _strip_code_fences

        variables_info = [variable.format() for variable in variables]
        pred = await self.generate_action.acall(
            variables_info=variables_info,
            repl_history=history,
            iteration=f"{iteration + 1}/{self.max_iterations}",
        )
        if self.verbose:
            import logging as _logging

            _logging.getLogger("dspy.predict.rlm").info(
                f"RLM iteration {iteration + 1}/{self.max_iterations}\n"
                f"Reasoning: {pred.reasoning}\nCode:\n{pred.code}"
            )

        try:
            code = _strip_code_fences(pred.code)
            if hasattr(repl, "aexecute"):
                result = await repl.aexecute(code, variables=dict(input_args))
            else:
                result = repl.execute(code, variables=dict(input_args))
        except Exception as e:
            result = f"[Error] {e}"

        return self._process_execution_result(pred, result, history, output_field_names)

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
            input_args, input_file_fields, output_file_fields, self._output_dir
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
        self, repl: JspiInterpreter, file_plan: dict[str, Any]
    ) -> None:
        """Mount input files, skill modules, and create output dirs in the sandbox."""
        repl._ensure_deno_process()

        for host_path, virtual_path in file_plan["mounts"]:
            repl.mount_file_at(host_path, virtual_path)

        for virtual_dir in file_plan["output_dirs"]:
            repl.mkdir_p(virtual_dir)

        # Mount skill modules into /sandbox/lib/ so the RLM can import them
        if self._skill_modules:
            repl.mkdir_p("/sandbox/lib")
            for mod_name, host_path in self._skill_modules.items():
                repl.mount_file_at(host_path, f"/sandbox/lib/{mod_name}.py")
            repl.execute("import sys; sys.path.insert(0, '/sandbox/lib')")

    def _sync_output_files(
        self,
        repl: JspiInterpreter,
        prediction: dspy.Prediction,
        output_file_fields: dict[str, str],
        file_plan: dict[str, Any],
    ) -> None:
        """Sync output files from sandbox to host and populate prediction fields.

        The RLM submits the sandbox path(s) it wrote to. We sync those files
        back to the host and replace the prediction value with File objects.
        """
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
                if submitted_path and isinstance(submitted_path, str) and submitted_path.startswith("/sandbox/"):
                    basename = os.path.basename(submitted_path)
                    host_path = os.path.join(host_dir, basename)
                    os.makedirs(host_dir, exist_ok=True)
                    repl.sync_file_to(submitted_path, host_path)
                    setattr(prediction, field_name, File(path=host_path))
                else:
                    # Fallback: list the output dir and sync everything
                    virtual_files = repl.list_dir(virtual_dir)
                    if virtual_files:
                        os.makedirs(host_dir, exist_ok=True)
                        for vpath in virtual_files:
                            rel = os.path.relpath(vpath, virtual_dir)
                            hp = os.path.join(host_dir, rel)
                            os.makedirs(os.path.dirname(hp), exist_ok=True)
                            repl.sync_file_to(vpath, hp)
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
                if virtual_files:
                    os.makedirs(host_dir, exist_ok=True)
                    for vpath in virtual_files:
                        rel = os.path.relpath(vpath, virtual_dir)
                        hp = os.path.join(host_dir, rel)
                        os.makedirs(os.path.dirname(hp), exist_ok=True)
                        repl.sync_file_to(vpath, hp)
                        result_files.append(File(path=hp))
                setattr(prediction, field_name, result_files)

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
        lm_hist_start = snapshot_lm_history_len(lm)
        sub_lm = self._sub_lm
        sub_hist_start = snapshot_lm_history_len(sub_lm) if sub_lm and sub_lm is not lm else 0
        steps: list[IterationStep] = []

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
                    history = REPLHistory()

                    for iteration in range(self.max_iterations):
                        iter_start = time.perf_counter()
                        result = self._execute_iteration(
                            repl, variables, history, iteration, input_args, output_field_names
                        )

                        # Extract step data from the new history entry
                        new_history = result if isinstance(result, REPLHistory) else None
                        if new_history and len(new_history.entries) > len(history.entries):
                            entry = new_history.entries[-1]
                        elif isinstance(result, dspy.Prediction) and hasattr(result, "trajectory"):
                            traj = result.trajectory
                            entry_data = traj[-1] if traj else {}
                            from dspy.primitives.repl_types import REPLEntry

                            entry = REPLEntry(
                                reasoning=entry_data.get("reasoning", ""),
                                code=entry_data.get("code", ""),
                                output=entry_data.get("output", ""),
                            )
                        else:
                            entry = None

                        full_output = entry.output if entry else ""
                        if len(full_output) > 5000:
                            prompt_output = (
                                full_output[:5000]
                                + f"\n... (truncated to 5000/{len(full_output):,} chars)"
                            )
                        else:
                            prompt_output = full_output

                        step = IterationStep(
                            iteration=iteration + 1,
                            reasoning=entry.reasoning if entry else "",
                            code=entry.code if entry else "",
                            output=prompt_output,
                            untruncated_output=full_output,
                            error=full_output.startswith(("[Error]", "[Type Error]")),
                            duration_ms=ms_since(iter_start),
                            tool_calls=drain_tool_calls(),
                            predict_calls=drain_predict_calls(),
                        )
                        steps.append(step)

                        if isinstance(result, dspy.Prediction):
                            status = "completed"
                            prediction = result
                            if output_file_fields:
                                self._sync_output_files(
                                    repl, prediction, output_file_fields, file_plan
                                )
                            break
                        history = result
                    else:
                        prediction = self._extract_fallback(
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
                    return prediction
                finally:
                    reset_tool_call_collector(tool_token)
                    reset_predict_call_collector(predict_token)
        except Exception as exc:
            exc.trace = self._build_run_trace(
                status="error",
                steps=steps,
                lm=lm,
                sub_lm=sub_lm,
                lm_hist_start=lm_hist_start,
                sub_hist_start=sub_hist_start,
                run_start=run_start,
            )
            raise
        finally:
            if file_plan:
                self.generate_action, self.extract = orig_action, orig_extract

    async def _aforward_traced(
        self, file_plan: dict[str, Any] | None, **input_args: Any
    ) -> dspy.Prediction:
        """Execute aforward() with tracing and optional file I/O."""
        _, output_file_fields = scan_file_fields(self.signature) if file_plan else (None, {})

        orig_action, orig_extract = self.generate_action, self.extract
        if file_plan:
            self.generate_action, self.extract = self._build_signatures_with_files(
                file_plan["instructions"]
            )

        run_start = time.perf_counter()
        lm = dspy.settings.lm
        lm_hist_start = snapshot_lm_history_len(lm)
        sub_lm = self._sub_lm
        sub_hist_start = snapshot_lm_history_len(sub_lm) if sub_lm and sub_lm is not lm else 0
        steps: list[IterationStep] = []

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
                    history = REPLHistory()

                    for iteration in range(self.max_iterations):
                        iter_start = time.perf_counter()
                        result = await self._aexecute_iteration(
                            repl, variables, history, iteration, input_args, output_field_names
                        )

                        new_history = result if isinstance(result, REPLHistory) else None
                        if new_history and len(new_history.entries) > len(history.entries):
                            entry = new_history.entries[-1]
                        elif isinstance(result, dspy.Prediction) and hasattr(result, "trajectory"):
                            traj = result.trajectory
                            entry_data = traj[-1] if traj else {}
                            from dspy.primitives.repl_types import REPLEntry

                            entry = REPLEntry(
                                reasoning=entry_data.get("reasoning", ""),
                                code=entry_data.get("code", ""),
                                output=entry_data.get("output", ""),
                            )
                        else:
                            entry = None

                        full_output = entry.output if entry else ""
                        if len(full_output) > 5000:
                            prompt_output = (
                                full_output[:5000]
                                + f"\n... (truncated to 5000/{len(full_output):,} chars)"
                            )
                        else:
                            prompt_output = full_output

                        step = IterationStep(
                            iteration=iteration + 1,
                            reasoning=entry.reasoning if entry else "",
                            code=entry.code if entry else "",
                            output=prompt_output,
                            untruncated_output=full_output,
                            error=full_output.startswith(("[Error]", "[Type Error]")),
                            duration_ms=ms_since(iter_start),
                            tool_calls=drain_tool_calls(),
                            predict_calls=drain_predict_calls(),
                        )
                        steps.append(step)

                        if isinstance(result, dspy.Prediction):
                            status = "completed"
                            prediction = result
                            if output_file_fields:
                                self._sync_output_files(
                                    repl, prediction, output_file_fields, file_plan
                                )
                            break
                        history = result
                    else:
                        prediction = await self._aextract_fallback(
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
                    return prediction
                finally:
                    reset_tool_call_collector(tool_token)
                    reset_predict_call_collector(predict_token)
        except Exception as exc:
            exc.trace = self._build_run_trace(
                status="error",
                steps=steps,
                lm=lm,
                sub_lm=sub_lm,
                lm_hist_start=lm_hist_start,
                sub_hist_start=sub_hist_start,
                run_start=run_start,
            )
            raise
        finally:
            if file_plan:
                self.generate_action, self.extract = orig_action, orig_extract

    def _build_signatures_with_files(
        self, file_instructions: str
    ) -> tuple[dspy.Predict, dspy.Predict]:
        """Build signatures with file instructions.

        File output fields are replaced with str in the signature so the RLM
        submits a path string, not a JSON object. The framework wraps the
        path in File after syncing.
        """
        from .files import (
            _is_list_annotation,
            is_input_file_type,
            is_output_file_type,
        )

        # Replace file-typed fields with str/list[str] for the RLM's view
        modified_sig = self.signature
        for name, field in self.signature.output_fields.items():
            if is_output_file_type(field.annotation):
                replacement = list[str] if _is_list_annotation(field.annotation) else str
                modified_sig = modified_sig.with_updated_fields(
                    name,
                    desc=f"{field.json_schema_extra.get('desc', '')} "
                    f"(submit the sandbox path you wrote to)",
                    type_=replacement,
                )
        for name, field in self.signature.input_fields.items():
            if is_input_file_type(field.annotation):
                replacement = list[str] if _is_list_annotation(field.annotation) else str
                modified_sig = modified_sig.with_updated_fields(
                    name,
                    desc=field.json_schema_extra.get("desc", ""),
                    type_=replacement,
                )

        raw_tools = {name: tool.func for name, tool in self._user_tools.items()}
        action_sig, extract_sig = build_rlm_signatures(
            modified_sig,
            PREDICT_RLM_INSTRUCTIONS,
            raw_tools,
            self._format_tool_docs,
            skill_instructions=self._skill_instructions,
            file_instructions=file_instructions,
        )
        return dspy.Predict(action_sig), dspy.Predict(extract_sig)

    def _build_signatures(self) -> tuple[Signature, Signature]:
        """Build action and extract signatures with predict documentation."""
        raw_tools = {name: tool.func for name, tool in self._user_tools.items()}
        return build_rlm_signatures(
            self.signature,
            PREDICT_RLM_INSTRUCTIONS,
            raw_tools,
            self._format_tool_docs,
            skill_instructions=self._skill_instructions,
        )
