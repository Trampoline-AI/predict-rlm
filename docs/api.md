# API reference

## `PredictRLM`

The main class. Extends DSPy's `RLM` with a built-in `predict()` tool for structured sub-LM calls.

```python
from predict_rlm import PredictRLM

rlm = PredictRLM(
    signature,                # DSPy signature (str or Signature class)
    lm=None,                  # Main LM — LM instance or model string
    sub_lm=None,              # LM for predict() — LM instance or model string
    max_iterations=30,        # Max REPL iterations
    max_llm_calls=50,         # Max LM calls per execution
    max_output_chars=100_000, # Max chars from REPL output
    verbose=False,            # Log detailed execution info
    tools=None,               # Additional tool functions
    skills=None,              # List of Skill instances
    allowed_domains=None,     # Domains the sandbox can access
    debug=False,              # Print REPL activity to stderr
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
| `max_output_chars` | `int` | `100_000` | Maximum characters to include from REPL output per iteration. |
| `verbose` | `bool` | `False` | Log detailed execution info. |
| `tools` | `dict[str, Callable] \| list[Callable] \| None` | `None` | Additional tool functions callable from the sandbox. Accepts a dict mapping names to callables, or a list of callables (names inferred from `__name__`). `predict` is added automatically. |
| `skills` | `list[Skill] \| None` | `None` | [Skills](skills.md) providing domain-specific instructions, packages, and tools. Merged automatically. |
| `allowed_domains` | `list[str] \| None` | `None` | Domains/IPs the sandbox can access via network. By default, no network access. Example: `["api.example.com", "192.168.1.100:8080"]` |
| `debug` | `bool` | `False` | Print REPL code, output, errors, and tool calls to stderr in real-time. |
| `output_dir` | `str \| Path \| None` | `None` | Host directory for output files. When set, `File` output fields without an explicit path are written here. If `None`, a temp directory is used. |

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

Tools are functions the RLM can call from inside the sandbox. They run on the host, not in the WASM sandbox — use them to access databases, APIs, the filesystem, or anything that requires native Python.

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

The `predict()` tool is always added automatically. It runs a DSPy signature against the sub-LM from within the sandbox:

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

Every call to `PredictRLM` attaches a structured trace to the returned prediction as `prediction.trace`. The trace captures the full execution history: iterations, tool calls, predict() subcalls, token usage, and timings.

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

| Field | Type | Description |
|---|---|---|
| `status` | `"completed" \| "max_iterations" \| "error"` | How the run ended. `completed` = SUBMIT called, `max_iterations` = extract fallback, `error` = run failed. |
| `model` | `str` | Main LM model identifier. |
| `sub_model` | `str \| None` | Sub-LM model identifier, if different from main LM. |
| `iterations` | `int` | Total iterations executed. |
| `max_iterations` | `int` | Maximum iterations allowed. |
| `duration_ms` | `int` | Total wall-clock duration in milliseconds. |
| `usage` | `LMUsage` | Token usage split by main and sub LM. |
| `steps` | `list[IterationStep]` | Per-iteration execution steps. |

#### `LMUsage`

| Field | Type | Description |
|---|---|---|
| `main` | `TokenUsage` | Main LM token usage. |
| `sub` | `TokenUsage` | Sub-LM token usage. |

#### `TokenUsage`

| Field | Type | Description |
|---|---|---|
| `input_tokens` | `int` | Total input/prompt tokens. |
| `output_tokens` | `int` | Total output/completion tokens. |
| `cost` | `float` | Total cost in USD. |

#### `IterationStep`

| Field | Type | Description |
|---|---|---|
| `iteration` | `int` | 1-indexed iteration number. |
| `reasoning` | `str` | LM reasoning for this iteration. |
| `code` | `str` | Python code generated by the LM. |
| `output` | `str` | Sandbox output as shown to the model (truncated to 5K chars). |
| `untruncated_output` | `str` | Full sandbox output before prompt truncation. |
| `error` | `bool` | `true` if code execution raised an error. |
| `duration_ms` | `int` | Wall-clock duration of this iteration. |
| `tool_calls` | `list[ToolCall]` | Tool calls made during this iteration (excluding predict). |
| `predict_calls` | `list[PredictCallGroup]` | predict() subcalls, grouped by signature. |

#### `ToolCall`

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Tool function name. |
| `args` | `list[Any]` | Positional arguments. |
| `kwargs` | `dict[str, Any]` | Keyword arguments. |
| `result` | `Any` | Return value from the tool. |
| `error` | `str \| None` | Error message if the call failed. |
| `duration_ms` | `int` | Wall-clock duration in milliseconds. |

#### `PredictCallGroup`

Calls sharing the same signature, instructions, and model are grouped to reduce trace bloat (common when using `asyncio.gather` for parallel predict calls).

| Field | Type | Description |
|---|---|---|
| `signature` | `str` | DSPy signature string. |
| `instructions` | `str \| None` | Task instructions passed to the sub-LM. |
| `model` | `str` | Model identifier. |
| `total_usage` | `TokenUsage` | Sum of token usage across all calls in the group. |
| `calls` | `list[PredictCallDetail]` | Per-call metrics. |

#### `PredictCallDetail`

| Field | Type | Description |
|---|---|---|
| `duration_ms` | `int` | Wall-clock duration in milliseconds. |
| `usage` | `TokenUsage` | Token usage for this call. |
| `input` | `dict[str, Any]` | Input fields passed to the sub-LM. |
| `output` | `dict[str, Any]` | Output fields returned by the sub-LM. |

---

## `File`

Unified file type for inputs and outputs. Behavior is determined by the field's position in the signature.

```python
from predict_rlm import File

File(path="report.pdf")        # single file reference
File.from_dir("docs/")         # all files in a directory -> list[File]
```

### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `path` | `str \| None` | `None` | Path to the file. For inputs, the host path to mount into the sandbox. For outputs, populated after execution with the host path of the generated file. |

### Methods

| Method | Returns | Description |
|---|---|---|
| `File.from_dir(path)` | `list[File]` | Create `File` references for every file in a directory (recursive walk, sorted). |

### Input vs output behavior

As an **input field**, the file is mounted from the host into the sandbox at `/sandbox/input/{field_name}/`. The RLM can read it with standard Python file I/O.

As an **output field**, the RLM writes files to `/sandbox/output/{field_name}/`. After execution, the files are synced back to the host and `path` is populated with the host path.

```python
class MySignature(dspy.Signature):
    source: File = dspy.InputField()            # mounted into sandbox
    docs: list[File] = dspy.InputField()        # multiple files mounted
    result: File = dspy.OutputField()           # single file synced back
    outputs: list[File] = dspy.OutputField()    # multiple files synced back
```

---

## `Skill`

Reusable bundle of instructions, packages, modules, and tools. See the [skills guide](skills.md) for detailed usage.

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

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | — | Short identifier for the skill (e.g. `"pdf-extraction"`). Required. |
| `instructions` | `str` | `""` | Prose instructions injected into the RLM prompt. Describes patterns, best practices, and domain knowledge. |
| `packages` | `list[str]` | `[]` | PyPI packages installed in the sandbox via `micropip` before the first code execution. |
| `modules` | `dict[str, str]` | `{}` | Python modules to mount in the sandbox. Maps import name to host filesystem path of the `.py` file. |
| `tools` | `dict[str, Callable]` | `{}` | Tool functions exposed to the RLM alongside `predict()`. Can be sync or async. |
