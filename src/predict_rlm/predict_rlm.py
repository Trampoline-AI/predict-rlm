"""PredictRLM - RLM subclass with predict tool for DSPy signatures."""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, List, Optional

import dspy
from dspy.primitives.code_interpreter import CodeInterpreter
from pydantic import create_model

from ._shared import build_rlm_signatures, format_tool_docs_full
from .files import File, build_file_plan, scan_file_fields
from .interpreter import JspiInterpreter
from .rlm_skills import Skill, merge_skills

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

        # Handle primitive types
        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "object": dict,
            "array": list,
        }
        return type_map.get(info.get("type", "string"), str)

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

You have access to a Python REPL environment. Write code in ```repl blocks and it will be executed. You will see the output, then write more code based on what you learned. This is an iterative, interactive process — explore your data, plan your approach, and build up your answer step by step across multiple iterations.

## Available

- Variables: {inputs} (your input data)
- `await predict(signature, *, instructions=None, **kwargs)` — your primary analysis tool (async, must await)
  - signature: str with type hints, e.g. `"page: dspy.Image, question: str -> answer: str"`
  - For images, use `dspy.Image` type hint and pass URL or base64 string directly
  - instructions: optional str describing the task
  - Returns: dict with output field names as keys
  - Capacity: ~400K tokens per call — you can pass substantial data
- `print()` — ALWAYS print to see results (REPL output is truncated, so keep prints focused)
- `SUBMIT({final_output_names})` — submit final output when done
- Standard libraries: re, json, collections, math, asyncio, etc.

The REPL runs inside an async event loop — use `await` directly, not `asyncio.run()`.

## Think step by step — iterate, don't solve all at once

This is ITERATIVE. Each code block executes, you see the output, then you decide what to do next. State persists between iterations, so use variables as buffers to accumulate findings.

On your first iteration, **explore before you extract**. Print samples of your input data — check types, lengths, what the data looks like. Understand what you're working with before writing any extraction logic. Even a quick `print(type(images), len(images))` or examining the first page can save you from going down the wrong path.

After exploring, plan your approach, then execute it in small focused steps. Each REPL block should do one logical thing — if a block fails or output truncates mid-way, you lose that iteration's state. Save intermediate results to variables so you can build on them.

Use `predict()` for anything requiring understanding of meaning — it's a powerful vision-language model. Use Python for computation, formatting, and aggregation. When REPL output is truncated and you can't see all the data, pass it to `predict()` for analysis rather than trying to print it all.

## Parallelizing predict() calls

`predict()` is async and much faster when run concurrently. For independent calls, always use `asyncio.gather()`:

```repl
import asyncio
tasks = [predict("img: dspy.Image -> text: str", img=url) for url in page_urls]
results = await asyncio.gather(*tasks)
```

Use sequential iteration only when each step depends on previous results (e.g. accumulating context across pages).

## Output types

Prefer typed outputs over JSON strings. You can use typed fields, lists, and Pydantic models for complex structures:

```repl
# Typed fields
result = await predict("page: dspy.Image -> title: str, date: str, amount: float", page=url)

# Lists
result = await predict("text: str -> keywords: list[str]", text=doc)

# Pydantic for nested structures
from pydantic import BaseModel
class LineItem(BaseModel):
    description: str
    amount: float
result = await predict("page: dspy.Image -> items: list[LineItem]", page=url)
```

## Examples

Suppose you have a set of document images and need to extract key dates from each page. You might start by exploring the data, then process each page:

```repl
print(f"Number of pages: {{len(images)}}")
# Look at what we're working with
sample = await predict(
    "page: dspy.Image -> description: str",
    instructions="Briefly describe what this page contains.",
    page=images[0],
)
print(sample["description"])
```

Then in the next iteration, once you understand the format, extract from all pages in parallel:

```repl
import asyncio
tasks = [
    predict(
        "page: dspy.Image -> dates: list[str], context: str",
        instructions="Extract all dates mentioned on this page with their context.",
        page=img,
    )
    for img in images
]
results = await asyncio.gather(*tasks)
for i, r in enumerate(results):
    print(f"Page {{i}}: {{r['dates']}}")
```

---

Suppose you're processing a long document page by page and need to track requirements as you go. Since each page's meaning depends on what came before, process sequentially and build state:

```repl
findings = []
for i, img in enumerate(images):
    prior_summary = "; ".join(findings[-3:]) if findings else "None yet"
    result = await predict(
        "page: dspy.Image, prior: str -> new_requirements: list[str]",
        instructions=f"Page {{i+1}} of {{len(images)}}. Identify new requirements not already covered in prior findings.",
        page=img, prior=prior_summary,
    )
    findings.extend(result["new_requirements"])
    print(f"Page {{i+1}}: +{{len(result['new_requirements'])}} requirements (total: {{len(findings)}})")
```

---

Suppose you have 50 pages and need to find all mentions of insurance requirements. Map across all pages in parallel, then synthesize:

```repl
import asyncio
tasks = [
    predict(
        "page: dspy.Image -> insurance_items: list[str]",
        instructions="Extract any insurance requirements, coverage amounts, or policy references.",
        page=img,
    )
    for img in images
]
page_results = await asyncio.gather(*tasks)

# Collect all items with page references
all_items = []
for i, r in enumerate(page_results):
    for item in r["insurance_items"]:
        all_items.append(f"[Page {{i+1}}] {{item}}")
print(f"Found {{len(all_items)}} insurance items across {{len(images)}} pages")
```

Then synthesize into a structured answer:

```repl
combined = "\\n".join(all_items)
final = await predict(
    "raw_items: str -> requirements: list[str], total_coverage: str, sources: list[int]",
    instructions="Deduplicate and organize these insurance items. Cite source page numbers.",
    raw_items=combined,
)
print(final)
```

---

Suppose you've extracted items but aren't sure you got everything — maybe some pages had unusual formatting or small print. Run a targeted verification pass:

```repl
# Re-examine pages that returned few or no items
sparse_pages = [i for i, r in enumerate(page_results) if len(r["insurance_items"]) == 0]
print(f"Re-checking {{len(sparse_pages)}} pages that had no results")

recheck_tasks = [
    predict(
        "page: dspy.Image -> missed_items: list[str], notes: str",
        instructions="Look very carefully for any insurance-related content, including footnotes, sidebars, and fine print.",
        page=images[i],
    )
    for i in sparse_pages
]
rechecked = await asyncio.gather(*recheck_tasks)
for idx, r in zip(sparse_pages, rechecked):
    if r["missed_items"]:
        print(f"Page {{idx+1}} had missed items: {{r['missed_items']}}")
```

## When done

When you have your final answer, call `SUBMIT({final_output_names})`. Make sure you've verified your results — if something looks wrong (empty lists, zeros, unexpected values), reconsider your approach before submitting."""


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

        def _parse_image_fields(signature: str) -> dict[str, bool]:
            """Parse signature to find fields with dspy.Image type hint."""
            # Extract input part (before ->)
            inputs_part = signature.split("->")[0].strip()

            image_fields: dict[str, bool] = {}

            # Find fields with list[dspy.Image] type annotation
            list_pattern = r"(\w+)\s*:\s*(?:list|List)\s*\[\s*dspy\.Image\s*\]"
            for match in re.findall(list_pattern, inputs_part):
                image_fields[match] = True  # is_list = True

            # Find fields with dspy.Image type annotation (non-list)
            single_pattern = r"(\w+)\s*:\s*dspy\.Image(?!\s*\])"
            for match in re.findall(single_pattern, inputs_part):
                if match not in image_fields:  # Don't override list matches
                    image_fields[match] = False  # is_list = False

            return image_fields

        def _wrap_images(signature: str, kwargs: dict[str, Any]) -> dict[str, Any]:
            """Wrap values for dspy.Image typed fields."""
            image_fields = _parse_image_fields(signature)

            wrapped = {}
            for key, value in kwargs.items():
                if key in image_fields:
                    is_list = image_fields[key]
                    if is_list and isinstance(value, list):
                        wrapped[key] = [
                            dspy.Image(url=item) if isinstance(item, str) else item
                            for item in value
                        ]
                    elif not is_list and isinstance(value, str):
                        wrapped[key] = dspy.Image(url=value)
                    else:
                        wrapped[key] = value
                else:
                    wrapped[key] = value
            return wrapped

        def predict(
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

            # Auto-wrap image URLs/base64 for dspy.Image typed fields
            wrapped_kwargs = _wrap_images(signature, kwargs)

            # Reconstruct custom Pydantic types from schemas if provided
            custom_types: dict[str, type] = {}
            if pydantic_schemas:
                for name, schema in pydantic_schemas.items():
                    try:
                        built = _models_from_schema(schema)
                        custom_types.update(built)
                    except Exception as e:
                        import logging

                        logging.getLogger(__name__).warning(
                            f"Failed to reconstruct Pydantic model '{name}' from schema: {e}"
                        )
            else:
                import logging

                # Check if signature has custom types (capitalized names after colons)
                pattern = r":\s*(?:list\[|List\[|Optional\[)?([A-Z][A-Za-z0-9_]*)"
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
                    pattern = r":\s*(?:list\[|List\[|Optional\[)?([A-Z][A-Za-z0-9_]*)"
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
                        logger.warning(
                            f"Failed to resolve custom types {custom_names} in signature. "
                            f"This often happens when using asyncio.gather() with Pydantic models. "
                            f"Ensure models are defined at module/global scope, not inside functions. "
                            f"Error: {e}"
                        )
                        raise RuntimeError(
                            f"Failed to create signature '{signature}' with custom_types={list(custom_types.keys()) if custom_types else []}. "
                            f"If using Pydantic models in output types, ensure they are defined at global scope "
                            f"(not inside a function) so they can be found by the schema extractor. "
                            f"Original error: {e}"
                        ) from e

                logger.warning(
                    f"Failed to create Signature with parsed types: {e}. Falling back to string signature."
                )
                sig = signature  # Use string signature directly

            # Create predictor and run with the specified LM
            predictor = dspy.Predict(sig)
            with dspy.context(lm=lm):
                prediction = predictor(**wrapped_kwargs)

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

            return {
                field: _to_serializable(prediction[field])
                for field in prediction.keys()
                if not field.startswith("_")
            }

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
            if file_plan:
                return self._forward_with_files(file_plan, **kwargs)
            return super().forward(**kwargs)

    async def aforward(self, **kwargs: Any) -> dspy.Prediction:
        """Async version of forward(). Sets the LM context for async execution."""
        with self._lm_context():
            file_plan, kwargs = self._prepare_file_io(kwargs)
            if file_plan:
                return await self._aforward_with_files(file_plan, **kwargs)
            return await super().aforward(**kwargs)

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

    def _forward_with_files(
        self, file_plan: dict[str, Any], **input_args: Any
    ) -> dspy.Prediction:
        """Execute forward() with file I/O handling."""
        _, output_file_fields = scan_file_fields(self.signature)

        # Temporarily swap signatures to include file instructions
        orig_action, orig_extract = self.generate_action, self.extract
        self.generate_action, self.extract = self._build_signatures_with_files(
            file_plan["instructions"]
        )
        try:
            self._validate_inputs(input_args)
            output_field_names = list(self.signature.output_fields.keys())
            execution_tools = self._prepare_execution_tools()
            variables = self._build_variables(**input_args)

            with self._interpreter_context(execution_tools, file_plan) as repl:
                self._setup_sandbox_files(repl, file_plan)

                from dspy.primitives.repl_types import REPLHistory

                history = REPLHistory()
                for iteration in range(self.max_iterations):
                    result = self._execute_iteration(
                        repl, variables, history, iteration, input_args, output_field_names
                    )
                    if isinstance(result, dspy.Prediction):
                        if output_file_fields:
                            self._sync_output_files(
                                repl, result, output_file_fields, file_plan
                            )
                        return result
                    history = result

                prediction = self._extract_fallback(
                    variables, history, output_field_names
                )
                if output_file_fields:
                    self._sync_output_files(
                        repl, prediction, output_file_fields, file_plan
                    )
                return prediction
        finally:
            self.generate_action, self.extract = orig_action, orig_extract

    async def _aforward_with_files(
        self, file_plan: dict[str, Any], **input_args: Any
    ) -> dspy.Prediction:
        """Execute aforward() with file I/O handling."""
        _, output_file_fields = scan_file_fields(self.signature)

        orig_action, orig_extract = self.generate_action, self.extract
        self.generate_action, self.extract = self._build_signatures_with_files(
            file_plan["instructions"]
        )
        try:
            self._validate_inputs(input_args)
            output_field_names = list(self.signature.output_fields.keys())
            execution_tools = self._prepare_execution_tools()
            variables = self._build_variables(**input_args)

            with self._interpreter_context(execution_tools, file_plan) as repl:
                self._setup_sandbox_files(repl, file_plan)

                from dspy.primitives.repl_types import REPLHistory

                history = REPLHistory()
                for iteration in range(self.max_iterations):
                    result = await self._aexecute_iteration(
                        repl, variables, history, iteration, input_args, output_field_names
                    )
                    if isinstance(result, dspy.Prediction):
                        if output_file_fields:
                            self._sync_output_files(
                                repl, result, output_file_fields, file_plan
                            )
                        return result
                    history = result

                prediction = await self._aextract_fallback(
                    variables, history, output_field_names
                )
                if output_file_fields:
                    self._sync_output_files(
                        repl, prediction, output_file_fields, file_plan
                    )
                return prediction
        finally:
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
