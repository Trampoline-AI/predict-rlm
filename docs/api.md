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

| Parameter          | Type                                             | Default  | Description                                                                                                                                                                                                       |
| ------------------ | ------------------------------------------------ | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `signature`        | `type[Signature] \| str`                         | —        | Defines inputs and outputs. A string like `"images, query -> answer"` or a `dspy.Signature` class.                                                                                                                |
| `lm`               | `dspy.LM \| str \| None`                         | `None`   | Main LM that drives the RLM (writes and executes code). Accepts a `dspy.LM` instance or a model string like `"openai/gpt-5.4"`. If `None`, uses the current context LM from `dspy.settings.lm` or `dspy.context`. |
| `sub_lm`           | `dspy.LM \| str \| None`                         | `None`   | LM for the `predict()` tool. Accepts a `dspy.LM` instance or a model string like `"openai/gpt-5.1"`. If `None`, uses the current context LM.                                                                      |
| `max_iterations`   | `int`                                            | `30`     | Maximum REPL interaction iterations. Each iteration is one code → output → reasoning turn.                                                                                                                        |
| `max_llm_calls`    | `int`                                            | `50`     | Maximum LM calls per execution (both outer LM and sub-LM calls count).                                                                                                                                            |
| `max_output_chars` | `int`                                            | `50_000` | Maximum characters to include from REPL output per iteration.                                                                                                                                                     |
| `verbose`          | `bool`                                           | `True`   | Print human-readable RLM iteration blocks to stderr: reasoning, generated code, output, tool calls, errors, and `SUBMIT` payloads. Pass `False` for quiet execution.                                              |
| `tools`            | `dict[str, Callable] \| list[Callable] \| None`  | `None`   | Additional tool functions callable from the sandbox. Accepts a dict mapping names to callables, or a list of callables (names inferred from `__name__`). `predict` is added automatically.                        |
| `interpreter`      | `CodeInterpreter \| None`                        | `None`   | Legacy interpreter-shaped backend adapted to the runtime kernel. `DirectPythonBackend` belongs here.                                                                                                              |
| `adapters`         | `Sequence[InputAdapter \| OutputAdapter]`        | `()`     | Input and output adapter instances. Pass a list such as `adapters=[MyInputAdapter(), MyOutputAdapter()]`; each adapter's base class determines its role.                                                          |
| `execution`        | `ExecutionBackend \| None`                       | `None`   | Custom root session-based execution backend. Interpreter-shaped compatibility backends belong under `interpreter=`.                                                                                               |
| `modules`          | `Sequence[RuntimeContribution \| RuntimeModule]` | `()`     | Runtime contributions or zero-argument factories. Factories expand once at construction.                                                                                                                          |
| `events`           | `Sequence[EventSink]`                            | `()`     | Ordered lifecycle event sinks. Strict sinks are correctness-critical; non-strict sinks are best effort.                                                                                                           |
| `skills`           | `list[Skill] \| None`                            | `None`   | [Skills](skills.md) providing domain-specific instructions, packages, and tools. Merged automatically.                                                                                                            |
| `allowed_domains`  | `list[str] \| None`                              | `None`   | Domains/IPs the sandbox can access via network. By default, no network access. Example: `["api.example.com", "192.168.1.100:8080"]`                                                                               |
| `debug`            | `bool`                                           | `False`  | Print timestamped RLM and sandbox lifecycle diagnostics to stderr. Error-like debug records are colored red when the terminal supports ANSI colors.                                                               |
| `output_dir`       | `str \| Path \| None`                            | `None`   | Host directory for output files. When set, `File` output fields without an explicit path are written here. If `None`, a temp directory is used.                                                                   |

Adapter names must be unique within each role; an input and output adapter may
share a name. A configured adapter with the same name as a built-in
compatibility adapter replaces that built-in only for the matching input or
output role. See [Custom path inputs](custom-path-inputs.md) for file-like
boundaries and [Custom adapters and the runtime kernel](custom-adapters.md) for
advanced lifecycle ownership and runtime composition.

### Input adapter lifecycle

`InputAdapter` instances are construction-time configuration and may be shared
by concurrent calls. Keep them stateless: put every invocation's mutable
resources, baselines, retry/idempotence flags, and cleanup state in a typed
`PreparedInput` subclass or in `RunContext`.

#### Choose a binding pattern

**Pattern 1: the backend opens the resource from a description.** Use this when an
ID, URI, or host path contains everything the backend needs. For example, a volume ID
and a target sandbox path may be enough to attach a cloud volume. `prepare()` records
that description. Once the execution environment is ready, the kernel gives it to
the backend, which opens the resource and returns the path visible inside the
sandbox. The adapter does not override `open()` or `bind()`.

In the API, that description is an `Artifact` stored in
`PreparedInput.artifacts`. The default `bind()` passes it to
`session.mount(artifact)`. The backend uses the artifact's `kind` and metadata to
decide how to open it.

For example, an adapter and backend can agree on this cloud-volume description:

```python
sandbox_path = f"/sandbox/input/{field.name}"
artifact = Artifact(
    id=f"{field.name}-volume",
    kind="cloud.volume",
    metadata={
        "volume_id": value.volume_id,
        "sandbox_path": sandbox_path,
    },
)
return PreparedInput(
    model_value=sandbox_path,
    artifacts=(artifact,),
    sandbox_roots=(SandboxRootReservation(sandbox_path),),
)
```

`Artifact.id` identifies this attachment within the call. `Artifact.kind` tells the
backend which attachment contract to use. Keys such as `volume_id` belong to that
adapter/backend contract; `sandbox_path` is the planned destination used by the
kernel and backend. Reserving the same path prevents another input or output from
claiming it.

**Pattern 2: the adapter opens the resource first.** Use this when opening the
resource requires a provider client, login, lease, or other live handle:

1. `prepare()` records the resource configuration and planned sandbox path.
2. `open()` obtains the live handle and stores it in `ctx.state`.
3. `bind()` gives the handle and configuration to the active execution session.
4. The session attaches the resource and returns its sandbox path.
5. `bind()` returns that path to the RLM in `BoundInput.model_value`.

`bind()` does not call `ExecutionBackend` directly. It receives the `session` created
by that backend and uses the methods available on that session. See the complete
[remote workspace example](custom-adapters.md#advanced-input-example).

The hooks below are listed in execution order. In every signature, `self` is the
adapter instance configured when `PredictRLM` was constructed. It may be shared by
concurrent calls, so store per-call mutable state in `prepared` or `ctx`, not on
`self`.

#### 1. `prepare()`

```python
async def prepare(
    self,
    field: FieldDescriptor,
    value: AdapterValue | list[AdapterValue] | None,
    ctx: RunContext,
) -> PreparedInput: ...
```

Arguments:

- `field` describes the matched signature input;
- `value` is the caller-provided value; and
- `ctx` stores mutable state and cleanup callbacks for this `PredictRLM` call.

**Backend/session access:** None. `prepare()` runs before backend acquisition. Return
declarative requirements in `PreparedInput` instead of calling a backend or session.

Every input adapter implements `prepare()`. Use it to choose the input value the
RLM receives, add input-specific instructions, or declare host resources that
should appear in the sandbox. It returns a `PreparedInput`.

For an ordinary value, return `PreparedInput(model_value=value)`. For host
filesystem resources, return one of these declarations:

- `PreparedInput.path()` maps one host file or directory into the sandbox;
- `.paths()` maps an explicit list of host files; and
- `.glob()` selects and maps a filtered file tree from one host root.

Copy is the default. Pass `mode="mount"` to `path()` for an explicit live directory
view; unsupported backends fail rather than silently copying. `at=` chooses the
destination relative to `/sandbox`. The kernel derives the permissions, destination
claims, backend bindings, and model-visible sandbox paths.

For example, an adapter for `S3File` can download the object to a host path and
return `PreparedInput.path(local_path)`. The RLM then receives the corresponding
sandbox path instead of the `S3File` object.

#### 2. `open()`

```python
async def open(
    self,
    field: FieldDescriptor,
    prepared: PreparedInput,
    ctx: RunContext,
    backend: ExecutionBackend,
) -> None: ...
```

Arguments:

- `field` and `ctx` are the same objects passed to `prepare()`;
- `prepared` is the value returned by `prepare()`; and
- `backend` is the selected execution backend.

**Backend/session access:** Use `backend` to check backend compatibility or choose
backend-specific provider setup before acquisition. No `ExecutionSession` exists
yet, and the adapter must not acquire one; session acquisition belongs to the kernel.

`open()` returns `None`. It may mutate `ctx`, but should not mutate `field`,
`prepared`, or `backend`. Store an opened client, lease, or synchronization handle in
`ctx.state` under an adapter-owned key; `bind()`, `after_execution()`, and
`finalize()` receive the same `ctx` and can retrieve it. Release the resource in
`finalize()`, or register a LIFO cleanup callback with `ctx.add_cleanup()` when it
must also cover partial setup.

Use this hook when later hooks need a resource kept open for the whole call, such as
a provider client, writable-workspace lease, or synchronization handle. Do not
override `open()` when `prepare()` fully describes the input.

#### 3. `bind()`

```python
async def bind(
    self,
    field: FieldDescriptor,
    prepared: PreparedInput,
    ctx: RunContext,
    session: ExecutionSession,
) -> BoundInput: ...
```

Arguments:

- `field`, `prepared`, and `ctx` are carried forward from preparation; and
- `session` is the active execution session acquired by the kernel.

**Backend/session access:** Use `session` to make the prepared input available in the
execution environment, either through `ExecutionSession` methods or a custom session
capability. Do not finalize or release it; the kernel owns the session. The returned
`BoundInput.model_value` becomes the input value the RLM receives.

In Pattern 1, inherit the default `bind()`. In Pattern 2, override it to call the
backend-specific method exposed by `session`. In both cases, the execution session
itself is never passed to the RLM.

#### 4. `after_execution()`

```python
async def after_execution(
    self,
    field: FieldDescriptor,
    prepared: PreparedInput,
    ctx: RunContext,
    session: ExecutionSession,
    result: ExecutionResult | None,
    error: BaseException | None,
) -> None: ...
```

Arguments:

- `field`, `prepared`, and `ctx` identify the input and its call-local state;
- `session` is the active execution session;
- `result` contains the completed generated-code result on success; and
- `error` contains the raised exception on failure.

**Backend/session access:** Use `session` to read or synchronize state after the code
attempt, such as copying changed workspace files to durable storage. The session is
still kernel-owned and must not be finalized or released by the adapter.

Use this hook to save changes after every completed block of generated code. For
example, synchronize a writable workspace so completed changes remain durable even
if a later block fails. Do not override it for read-only inputs or when only a final
save is required. This is a durability hook, not an execution-telemetry hook, and it
does not run for cancelled attempts.

#### 5. `finalize()`

```python
async def finalize(
    self,
    field: FieldDescriptor,
    prepared: PreparedInput,
    ctx: RunContext,
    session: ExecutionSession | None,
    error: BaseException | None,
) -> None: ...
```

Arguments:

- `field`, `prepared`, and `ctx` identify the input and its call-local state;
- `session` is the active execution session, or `None` if acquisition did not finish;
  and
- `error` is the error that ended the call, or `None` after success.

**Backend/session access:** When `session` is available, use it for a final read or
synchronization before releasing adapter-owned resources. Do not finalize or release
the session itself; the kernel does that after this hook returns.

Use this hook to perform the final save and release resources opened for the call,
such as a workspace lease, provider client, or temporary directory. It runs exactly
once in reverse preparation order after generated execution stops and before the
kernel releases the execution session.

#### Failure and cleanup guarantees

Errors from `after_execution()` are fatal because durability is no longer known.
Finalization errors preserve the original execution or cancellation error as the
primary error and make strict evidence incomplete. An adapter owns its provider
resources and should release them in `finalize()` or a LIFO `ctx.add_cleanup()`
callback; the framework owns session finalization and release.

Reused or pooled sessions reject incompatible per-call policy instead of retaining
broader or stale permissions.

### Verbose, debug, and trace output

Verbose output is enabled by default for understanding the RLM's work product.
It prints colored iteration blocks to stderr: reasoning, generated code, sandbox
output, tool calls, errors, and `SUBMIT` payloads. The verbose stream is
intentionally plain text without logging prefixes. Pass `verbose=False` for
quiet execution.

`debug=True` is for diagnosing runtime behavior. It prints timestamped `logging`
records for RLM and sandbox lifecycle events such as process startup, requests,
timeouts, shutdown, and partial output captured before an error. Debug error
records are colored red.

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

| Field                | Type                     | Description                                                            |
| -------------------- | ------------------------ | ---------------------------------------------------------------------- |
| `iteration`          | `int`                    | 1-indexed iteration number.                                            |
| `reasoning`          | `str`                    | LM reasoning for this iteration.                                       |
| `code`               | `str`                    | Python code generated by the LM.                                       |
| `output`             | `str`                    | Sandbox output as shown to the model, shortened by `max_output_chars`. |
| `untruncated_output` | `str`                    | Full sandbox output before prompt truncation.                          |
| `error`              | `bool`                   | `true` if code execution raised an error.                              |
| `duration_ms`        | `int`                    | Wall-clock duration of this iteration.                                 |
| `tool_calls`         | `list[ToolCall]`         | Tool calls made during this iteration (excluding predict).             |
| `predict_calls`      | `list[PredictCallGroup]` | predict() subcalls, grouped by signature.                              |

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

| Field  | Type          | Default | Description                                                                                                                                           |
| ------ | ------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
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

`CtxStr` is input-only and currently supports class-based DSPy signatures only.
Use `field: CtxStr`, not `list[CtxStr]`, `CtxStr | None`, or string signatures
such as `"criteria: CtxStr -> answer"`.

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
