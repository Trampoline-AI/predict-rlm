# API reference

## `PredictRLM`

The main class. Extends DSPy's `RLM` with a built-in `predict()` tool for
structured sub-LM calls.

```python
from predict_rlm import PredictRLM

rlm = PredictRLM(
    signature,                # DSPy signature (str or Signature class)
    lm=None,                  # Main LM — LM instance or model string
    sub_lm=None,              # LM for predict() — LM instance or model string
    max_iterations=30,        # Max REPL iterations
    max_llm_calls=50,         # Max LM calls per execution
    max_output_chars=50_000,  # Max chars from REPL output
    verbose=True,             # Print human-readable iteration trace blocks
    tools=None,               # Additional tool functions
    interpreter=None,         # Legacy interpreter compatibility option
    adapters=(),              # Input and output adapter instances
    execution=None,           # Custom root execution backend
    modules=(),               # Runtime contributions or factories
    events=(),                # Ordered lifecycle event sinks
    skills=None,              # List of Skill instances
    allowed_domains=None,     # Domains the sandbox can access
    debug=False,              # Print timestamped lifecycle diagnostics
    output_dir=None,          # Host directory for output files
)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `signature` | `type[Signature] \| str` | — | Defines inputs and outputs. A string like `"images, query -> answer"` or a `dspy.Signature` class. |
| `lm` | `dspy.LM \| str \| None` | `None` | Main LM that drives the RLM (writes and executes code). Accepts a `dspy.LM` instance or a model string like `"openai/gpt-5.4"`. If `None`, uses the current context LM from `dspy.settings.lm` or `dspy.context`. |
| `sub_lm` | `dspy.LM \| str \| None` | `None` | LM for the `predict()` tool. Accepts a `dspy.LM` instance or a model string like `"openai/gpt-5.1"`. If `None`, uses the current context LM. |
| `max_iterations` | `int` | `30` | Maximum REPL interaction iterations. Each iteration is one code → output → reasoning turn. |
| `max_llm_calls` | `int` | `50` | Maximum LM calls per execution (both outer LM and sub-LM calls count). |
| `max_output_chars` | `int` | `50_000` | Maximum characters to include from REPL output per iteration. |
| `verbose` | `bool` | `True` | Print human-readable RLM iteration blocks to stderr: reasoning, generated code, output, tool calls, errors, and `SUBMIT` payloads. Pass `False` for quiet execution. |
| `tools` | `dict[str, Callable] \| list[Callable] \| None` | `None` | Additional tool functions callable from the sandbox. Accepts a dict mapping names to callables, or a list of callables (names inferred from `__name__`). `predict` is added automatically. |
| `interpreter` | `CodeInterpreter \| None` | `None` | Legacy interpreter-shaped backend adapted to the runtime kernel. `DirectPythonBackend` belongs here. |
| `adapters` | `Sequence[InputAdapter \| OutputAdapter]` | `()` | Input and output adapter instances. Pass a list such as `adapters=[MyInputAdapter(), MyOutputAdapter()]`; each adapter's base class determines its role. |
| `execution` | `ExecutionBackend \| None` | `None` | Custom root session-based execution backend. Interpreter-shaped compatibility backends belong under `interpreter=`. |
| `modules` | `Sequence[RuntimeContribution \| RuntimeModule]` | `()` | Runtime contributions or zero-argument factories. Factories expand once at construction. |
| `events` | `Sequence[EventSink]` | `()` | Ordered lifecycle event sinks. Strict sinks are correctness-critical; non-strict sinks are best effort. |
| `skills` | `list[Skill] \| None` | `None` | [Skills](skills.md) providing domain-specific instructions, packages, and tools. Merged automatically. |
| `allowed_domains` | `list[str] \| None` | `None` | Domains/IPs the sandbox can access via network. By default, no network access. Example: `["api.example.com", "192.168.1.100:8080"]` |
| `debug` | `bool` | `False` | Print timestamped RLM and sandbox lifecycle diagnostics to stderr. Error-like debug records are colored red when the terminal supports ANSI colors. |
| `output_dir` | `str \| Path \| None` | `None` | Host directory for output files. When set, `File` output fields without an explicit path are written here. If `None`, a temp directory is used. |

Adapter names must be unique within each role; an input and output adapter may
share a name. A configured adapter with the same name as a built-in compatibility
adapter replaces that built-in only for the matching input or output role.
See [Custom path inputs](custom-path-inputs.md) for file-like boundaries and
[Custom adapters and the runtime kernel](custom-adapters.md) for advanced
lifecycle ownership and runtime composition.

### Input adapter lifecycle

`InputAdapter` instances are construction-time configuration and may be shared by
concurrent calls. Keep them stateless: put every invocation's mutable resources,
baselines, retry/idempotence flags, and cleanup state in a typed `PreparedInput`
subclass or in `RunContext`.

For each invocation, callbacks run in this order:

1. `prepare()` runs before acquisition and returns the model-visible plain value
   or declarative paths through `PreparedInput.path()`, `paths()`, or `glob()`.
   The kernel derives filesystem policy, destination claims, and backend bindings.
2. `prepare_session()` runs before backend acquisition. An adapter whose callback
   was entered is finalized even if that callback raises.
3. After acquisition, `mount()` binds the prepared value into the session.
4. `after_execution()` runs in mount order after each completed generated-code
   attempt, whether it succeeded or raised. It excludes framework bootstrap code
   and is a durability hook, not execution telemetry. Cancelled attempts do not
   invoke it.
5. `finalize()` runs exactly once in reverse preparation order, before session
   finalization and release. Cancellation first quiesces generated execution so
   finalization can perform its last durable flush.

Errors from `after_execution()` are fatal because durability is no longer known.
Finalization errors preserve the original execution/cancellation error as primary,
are attached to it, and make strict evidence incomplete. An adapter owns its
provider resources and should release them in `finalize()` or a LIFO
`ctx.add_cleanup()` callback; the framework owns session finalization and release.

`PreparedInput.path()` copies one host file or directory by default. Pass
`mode="mount"` for an explicit live view; unsupported backends fail rather than
silently copying. `PreparedInput.paths()` handles an explicit list, while
`PreparedInput.glob()` expands include/exclude patterns on the host and copies a
deterministically ordered filtered snapshot. `at=` is always relative to
`/sandbox`.

These helpers own the common mount implementation. Override `mount()` only for a
boundary they cannot express. Network access and stateful provider lifecycles
remain explicit advanced concerns. Reused or pooled sessions reject incompatible
per-invocation policy instead of retaining broader or stale permissions.

### Verbose, debug, and trace output

Verbose output is enabled by default for understanding the RLM's work product.
It prints colored iteration blocks to stderr: reasoning, generated code,
sandbox output, tool calls, errors, and `SUBMIT` payloads. The verbose stream
is intentionally plain text without logging prefixes. Pass `verbose=False` for
quiet execution.

`debug=True` is for diagnosing runtime behavior. It prints timestamped
`logging` records for RLM and sandbox lifecycle events such as process startup,
requests, timeouts, shutdown, and partial output captured before an error.
Debug error records are colored red.

These flags are independent and can be enabled together:

```python
rlm = PredictRLM(MySignature, debug=True)
```

Every run also attaches a structured `RunTrace` to the returned prediction as
`prediction.trace`. If sandbox code prints output and then fails, the output
printed before the exception is preserved in the failed iteration's trace output
before the formatted `[Error] ...` line.

### Usage

```python
# String signature
rlm = PredictRLM("documents, query -> answer: str", lm="openai/gpt-5.4")

# Signature class
class AnalyzeDocuments(dspy.Signature):
    """Analyze documents and produce a report."""
    documents: list[File] = dspy.InputField()
    analysis: str = dspy.OutputField()

rlm = PredictRLM(AnalyzeDocuments, lm="openai/gpt-5.4", sub_lm="openai/gpt-5.1")
result = rlm(documents=[File(path="report.pdf")])
```

### Tools

Tools are functions the RLM can call from inside the sandbox. They run on the
host, not in the WASM sandbox — use them to access databases, APIs, the
filesystem, or anything that requires native Python.

```python
def fetch_url(url: str) -> str:
    """Fetch a URL and return its content."""
    return requests.get(url).text

rlm = PredictRLM(
    "urls, query -> answer",
    lm="openai/gpt-5.4",
    tools=[fetch_url],           # list form — name inferred from __name__
    # tools={"fetch": fetch_url} # dict form — explicit name
)
```

The `predict()` tool is always added automatically. It runs a DSPy signature
against the sub-LM from within the sandbox:

```python
# Inside the sandbox, the RLM writes code like this:
result = await predict(
    "page: dspy.Image -> dates: list[str], entities: list[str]",
    instructions="Extract dates and entities from this page.",
    page=page_image,
)
```

---

## `RunTrace`

Every call to `PredictRLM` attaches a structured trace to the returned
prediction as `prediction.trace`. The trace captures the full execution history:
iterations, tool calls, predict() subcalls, token usage, and timings.

> **Note:** The trace schema is experimental and may change in future versions.

```python
from predict_rlm import PredictRLM, RunTrace

result = rlm(documents=[File(path="report.pdf")])

# Access the trace
trace: RunTrace = result.trace

print(trace.status)       # "completed" | "max_iterations" | "error"
print(trace.iterations)   # number of iterations executed
print(trace.duration_ms)  # total wall-clock time

# Token usage split by LM
print(trace.usage.main)   # TokenUsage for the outer LM
print(trace.usage.sub)    # TokenUsage for the sub-LM

# Iterate over execution steps
for step in trace.steps:
    print(f"Step {step.iteration}: {len(step.predict_calls)} predict groups, {len(step.tool_calls)} tool calls")
    if step.error:
        print(f"  Error: {step.output}")

# Export to JSON file (compact — base64 images replaced with size summaries)
trace.to_exportable_json("trace.json")

# Or get the compact JSON string
json_str = trace.to_exportable_json()

# Full data including raw base64 payloads (use for programmatic access)
data = trace.model_dump()
```

### Schema

#### `RunTrace`

| Field            | Type                                         | Description                                                                                                |
| ---------------- | -------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `status`         | `"completed" \| "max_iterations" \| "error"` | How the run ended. `completed` = SUBMIT called, `max_iterations` = extract fallback, `error` = run failed. |
| `model`          | `str`                                        | Main LM model identifier.                                                                                  |
| `sub_model`      | `str \| None`                                | Sub-LM model identifier, if different from main LM.                                                        |
| `iterations`     | `int`                                        | Total iterations executed.                                                                                 |
| `max_iterations` | `int`                                        | Maximum iterations allowed.                                                                                |
| `duration_ms`    | `int`                                        | Total wall-clock duration in milliseconds.                                                                 |
| `usage`          | `LMUsage`                                    | Token usage split by main and sub LM.                                                                      |
| `steps`          | `list[IterationStep]`                        | Per-iteration execution steps.                                                                             |

#### `LMUsage`

| Field  | Type         | Description          |
| ------ | ------------ | -------------------- |
| `main` | `TokenUsage` | Main LM token usage. |
| `sub`  | `TokenUsage` | Sub-LM token usage.  |

#### `TokenUsage`

| Field           | Type    | Description                     |
| --------------- | ------- | ------------------------------- |
| `input_tokens`  | `int`   | Total input/prompt tokens.      |
| `output_tokens` | `int`   | Total output/completion tokens. |
| `cost`          | `float` | Total cost in USD.              |

#### `IterationStep`

| Field                | Type                     | Description                                                   |
| -------------------- | ------------------------ | ------------------------------------------------------------- |
| `iteration`          | `int`                    | 1-indexed iteration number.                                   |
| `reasoning`          | `str`                    | LM reasoning for this iteration.                              |
| `code`               | `str`                    | Python code generated by the LM.                              |
| `output`             | `str`                    | Sandbox output as shown to the model, shortened by `max_output_chars`. |
| `untruncated_output` | `str`                    | Full sandbox output before prompt truncation.                 |
| `error`              | `bool`                   | `true` if code execution raised an error.                     |
| `duration_ms`        | `int`                    | Wall-clock duration of this iteration.                        |
| `tool_calls`         | `list[ToolCall]`         | Tool calls made during this iteration (excluding predict).    |
| `predict_calls`      | `list[PredictCallGroup]` | predict() subcalls, grouped by signature.                     |

#### `ToolCall`

| Field         | Type             | Description                          |
| ------------- | ---------------- | ------------------------------------ |
| `name`        | `str`            | Tool function name.                  |
| `args`        | `list[Any]`      | Positional arguments.                |
| `kwargs`      | `dict[str, Any]` | Keyword arguments.                   |
| `result`      | `Any`            | Return value from the tool.          |
| `error`       | `str \| None`    | Error message if the call failed.    |
| `duration_ms` | `int`            | Wall-clock duration in milliseconds. |

#### `PredictCallGroup`

Calls sharing the same signature, instructions, and model are grouped to reduce
trace bloat (common when using `asyncio.gather` for parallel predict calls).

| Field          | Type                      | Description                                       |
| -------------- | ------------------------- | ------------------------------------------------- |
| `signature`    | `str`                     | DSPy signature string.                            |
| `instructions` | `str \| None`             | Task instructions passed to the sub-LM.           |
| `model`        | `str`                     | Model identifier.                                 |
| `total_usage`  | `TokenUsage`              | Sum of token usage across all calls in the group. |
| `calls`        | `list[PredictCallDetail]` | Per-call metrics.                                 |

#### `PredictCallDetail`

| Field         | Type             | Description                           |
| ------------- | ---------------- | ------------------------------------- |
| `duration_ms` | `int`            | Wall-clock duration in milliseconds.  |
| `usage`       | `TokenUsage`     | Token usage for this call.            |
| `input`       | `dict[str, Any]` | Input fields passed to the sub-LM.    |
| `output`      | `dict[str, Any]` | Output fields returned by the sub-LM. |

---

## `File`

Unified file type for inputs and outputs. Behavior is determined by the field's
position in the signature.

```python
from predict_rlm import File

File(path="report.pdf")        # single file reference
File.from_dir("docs/")         # all files in a directory -> list[File]
```

### Fields

| Field  | Type          | Default | Description                                                                                                                                             |
| ------ | ------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `path` | `str \| None` | `None`  | Path to the file. For inputs, the host path copied into the sandbox. For outputs, populated after execution with the host path of the generated file. |

### Methods

| Method                | Returns      | Description                                                                      |
| --------------------- | ------------ | -------------------------------------------------------------------------------- |
| `File.from_dir(path)` | `list[File]` | Create `File` references for every file in a directory (recursive walk, sorted). |

### Input vs output behavior

As an **input field**, the file is copied from the host into the sandbox at
`/sandbox/input/{field_name}/`. The RLM can read it with standard Python file
I/O.

As an **output field**, the RLM writes files to `/sandbox/output/{field_name}/`.
After execution, the files are synced back to the host and `path` is populated
with the host path.

```python
class MySignature(dspy.Signature):
    source: File = dspy.InputField()            # copied into sandbox
    docs: list[File] = dspy.InputField()        # multiple files copied
    result: File = dspy.OutputField()           # single file synced back
    outputs: list[File] = dspy.OutputField()    # multiple files synced back
```

---

## `CtxStr`

String input marker for small-to-moderate text whose final adapter-prepared
value should be injected in full into the outer RLM prompt. Use this for
criteria, rubrics, task instructions, or other text the outer RLM should see
immediately instead of only inspecting through REPL variables.

```python
import dspy

from predict_rlm import CtxStr, PredictRLM

class Analyze(dspy.Signature):
    criteria: CtxStr = dspy.InputField(desc="Full rubric to apply")
    data: str = dspy.InputField(desc="Data available as a normal REPL variable")
    answer: str = dspy.OutputField()

rlm = PredictRLM(Analyze)
result = rlm(
    criteria="Use this rubric exactly...",
    data="This remains available as a normal REPL variable.",
)
```

Callers pass a plain `str`. The selected input adapter prepares it, and the
final prepared string is both the same-named Python variable in the sandbox and
an in-full appendix to the action and extract prompts under
`## In-Context Inputs`. This avoids relying on the ordinary variable preview.
Per-run predictors live on the run context, so concurrent calls on one
`PredictRLM` instance remain isolated. PredictRLM recognizes `CtxStr`
automatically; no extra runtime configuration is required.

`CtxStr` is input-only and currently supports class-based DSPy signatures
only. Use `field: CtxStr`, not `list[CtxStr]`, `CtxStr | None`, or string
signatures such as `"criteria: CtxStr -> answer"`.

See [How it works](how-it-works.md#signatures-file-io-and-in-context-inputs) for
how `CtxStr` composes with ordinary sandbox variables.

---

## `Skill`

Reusable bundle of instructions, packages, modules, and tools. See the
[skills guide](skills.md) for detailed usage.

```python
from predict_rlm import Skill

Skill(
    name="my-skill",                        # short identifier (required)
    instructions="How to approach...",       # injected into the RLM prompt
    packages=["pandas", "pdfplumber"],       # installed in the sandbox
    modules={"helper": "/path/to/mod.py"},   # mounted as importable modules
    tools={"my_func": my_func},             # exposed alongside predict()
)
```

### Fields

| Field          | Type                  | Default | Description                                                                                                |
| -------------- | --------------------- | ------- | ---------------------------------------------------------------------------------------------------------- |
| `name`         | `str`                 | —       | Short identifier for the skill (e.g. `"pdf-extraction"`). Required.                                        |
| `instructions` | `str`                 | `""`    | Prose instructions injected into the RLM prompt. Describes patterns, best practices, and domain knowledge. |
| `packages`     | `list[str]`           | `[]`    | PyPI packages installed in the sandbox via `micropip` before the first code execution.                     |
| `modules`      | `dict[str, str]`      | `{}`    | Python modules to mount in the sandbox. Maps import name to host filesystem path of the `.py` file.        |
| `tools`        | `dict[str, Callable]` | `{}`    | Tool functions exposed to the RLM alongside `predict()`. Can be sync or async.                             |
