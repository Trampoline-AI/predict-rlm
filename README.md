> [!NOTE]
> Read the launch post for our optimized
> [SpreadsheetBench skill](https://x.com/GabLesperance/status/2048072367876735415).

# predict-rlm

Many LLM workflows are too complex for one prompt and too adaptive for a fixed
chain.

**predict-rlm gives models a runtime for those workflows:** inspect files, keep
state, branch, call focused sub-models, use tools, manage large context through
code, and return typed output.

You define the inputs, outputs, tools, and operating procedure. The model writes
and executes Python in a sandboxed REPL, adapting as it discovers evidence.

Use it when you know the outcome you want, but not the exact path.

Based on the [Recursive Language Models](https://arxiv.org/abs/2512.24601v1)
paper by [Alex L. Zhang](https://x.com/a1zhang),
[Tim Kraska](https://x.com/tim_kraska), and
[Omar Khattab](https://x.com/lateinteraction) from MIT CSAIL.<br/>

<br>
<p align="center">
  <a href="https://github.com/Trampoline-AI/predict-rlm/actions/workflows/tests.yml"><img src="https://img.shields.io/github/actions/workflow/status/Trampoline-AI/predict-rlm/tests.yml?label=Tests" alt="Tests"></a>
  <a href="https://codecov.io/gh/Trampoline-AI/predict-rlm"><img src="https://img.shields.io/codecov/c/github/Trampoline-AI/predict-rlm?token=NNS3R7OIT2&color=brightgreen&label=codecov" alt="codecov"></a>
  <a href="https://pypi.org/project/predict-rlm/"><img src="https://img.shields.io/pypi/v/predict-rlm?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/predict-rlm/"><img src="https://img.shields.io/pypi/pyversions/predict-rlm" alt="Python"></a>
  <a href="https://pypi.org/project/predict-rlm/"><img src="https://img.shields.io/pypi/dm/predict-rlm" alt="PyPI downloads"></a>
  <a href="https://discord.gg/BAkd288sGN"><img src="https://img.shields.io/badge/Discord-Join-5865F2?style=flat&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/Trampoline-AI/predict-rlm"><img src="https://img.shields.io/github/stars/trampoline-ai/predict-rlm?cacheSeconds=3600" alt="GitHub stars"></a>
  <br/>
  crafted  with ♥ in MTL · NYC · FLP<br>by <a href="https://trampoline.ai">Trampoline AI</a>
</p>

## When to use it

predict-rlm is a good fit when the model needs to explore, reason, and adapt
before it can produce the final output:

- codebase analysis and investigations
- document review, redaction, extraction, and comparison
- log analysis and incident-style evidence gathering
- spreadsheet and financial-model workflows
- audits, compliance review, and messy data transformation
- multi-file synthesis with typed outputs and readable traces

It is probably not the right tool for simple chat completions, one-shot
classification, deterministic ETL, or tiny prompts where a direct LLM call is
already enough.

## Installation

```bash
uv add predict-rlm
```

Optional extras are available for adjacent tooling:

```bash
# GEPA optimization support
uv add "predict-rlm[gepa]"

# Codex-backed DSPy LM and the `codex-lm` CLI
uv add "predict-rlm[codex-lm]"

# Docker Sandboxes backend support
uv add "predict-rlm[sbx]"
```

With the Codex LM extra installed, import `CodexLM` or use the script. The
vendored backend supports current Codex model slugs including `gpt-5.5`.

```python
from dspy_codex_lm import CodexLM
```

```bash
codex-lm auth list
codex-lm usage
```

## Why RLMs?

<p align="center">
  <img src="https://raw.githubusercontent.com/Trampoline-AI/predict-rlm/main/docs/bitter_lesson_spectrum.svg" alt="Bitter Lesson Spectrum — from hand-written prompts to RLMs" width="680"/>
</p>

- **Avoid context rot** — The outer LM works through files, variables, and tool
  calls instead of trying to keep every detail in one prompt. Large inputs stay
  as file paths and metadata until the runtime needs to inspect them.
- **Adaptive execution inside a defined procedure** — A signature gives the
  workflow a contract, while the REPL lets the model branch, retry, verify, and
  accumulate state across iterations.
- **Focused sub-model calls** — `predict()` lets the runtime spin up typed
  DSPy signatures for narrow perception and extraction tasks, including
  multimodal calls with `dspy.Image`.
- **Readable trajectories** — Every run records generated code, output, tool
  calls, `predict()` subcalls, timings, token usage, errors, and final `SUBMIT`
  payloads, so you can inspect what happened instead of guessing.
- **Optimization-ready traces** — The same traces that help humans debug a run
  can feed tools like [GEPA](https://gepa-ai.github.io/gepa) to improve RLM
  strategies from scored examples.

## Features

<p align="center">
  <img src="https://raw.githubusercontent.com/Trampoline-AI/predict-rlm/main/docs/harness_vs_rlm.svg" alt="Classic harness vs RLM architecture" width="600"/>
</p>

- **Multimodal** — process images and rendered document pages through sub-LM
  calls using native provider multimodal APIs.
- **Async tool calling** — native RLM async support in the WASM sandbox,
  enabling concurrent sub-LM invocations and tool calls.
- **Skills & tools** — bundle domain instructions, PyPI packages, sandbox
  modules, and host-side tools for reusable task capabilities.
- **Simple file I/O** — pass local files and mutable workspaces as typed inputs,
  and return generated artifacts through `File` outputs.
- **Structured sub-LM calls** — native Pydantic and DSPy signature support for
  type-safe sub-LM invocations with structured outputs.

## Demos

| Description                                                                                                                                              | Input / Output                                                                                                                   | Preview                                                                                                                                                                                                                          |
| -------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Document Analysis](examples/document_analysis/) — Analyze documents and extract key dates, entities, and financial information into a structured report | **Input:** PDFs<br>**Output:** Structured briefing report ([example output](examples/document_analysis/sample/output/report.md)) | <a href="examples/document_analysis/sample/output/report.md"><img src="https://raw.githubusercontent.com/Trampoline-AI/predict-rlm/main/examples/document_analysis/sample/output/screenshot.png" width="280"></a>                |
| [Document Redaction](examples/document_redaction/) — Redact PII from PDFs based on a policy, then verify the redactions visually                         | **Input:** PDFs<br>**Output:** Redacted PDFs ([example output](examples/document_redaction/sample/output/output.md))             | <a href="examples/document_redaction/sample/output/output.md"><img src="https://raw.githubusercontent.com/Trampoline-AI/predict-rlm/main/examples/document_redaction/sample/output/screenshot.png" width="280"></a>              |
| [Invoice Processing](examples/invoice_processing/) — Extract vendor info, line items, and totals from PDF invoices into a consolidated Excel spreadsheet | **Input:** PDF invoices<br>**Output:** Excel spreadsheet ([example output](examples/invoice_processing/sample/output/))          | <a href="examples/invoice_processing/sample/output/output.md"><img src="https://raw.githubusercontent.com/Trampoline-AI/predict-rlm/main/examples/invoice_processing/sample/output/screenshot.png" width="280"></a>              |
| [Contract Comparison](examples/contract_comparison/) — Compare two contract versions and produce a structured diff report with per-section analysis      | **Input:** 2 PDF contracts<br>**Output:** Structured diff report ([example output](examples/contract_comparison/sample/output/)) | <a href="examples/contract_comparison/sample/output/comparison-report.md"><img src="https://raw.githubusercontent.com/Trampoline-AI/predict-rlm/main/examples/contract_comparison/sample/output/screenshot.png" width="280"></a> |

## Quick start

### With your coding agent

Install the [predict-rlm skill](.agents/skills/rlm/SKILL.md) in Claude Code,
Codex, Cursor, or any compatible coding agent:

```bash
npx skills add Trampoline-AI/predict-rlm
```

Then ask your agent to build an RLM:

```
❯ /rlm build an RLM that extracts line items from PDF invoices into a spreadsheet
```

### Quick Example

```python
import dspy
from predict_rlm import File, PredictRLM

class AnalyzeImages(dspy.Signature):
    """Analyze images and answer the query. Load each image as a base64 data
    URI and use predict() with dspy.Image to extract visual information."""
    images: list[File] = dspy.InputField()
    query: str = dspy.InputField()
    answer: str = dspy.OutputField()

rlm = PredictRLM(
    AnalyzeImages,
    lm="openai/gpt-5.4",
    sub_lm="openai/gpt-5.1",
)

result = rlm(
    images=[File(path="page.png")],
    query="Extract all visible text, then count each letter A-Z (case-insensitive).",
)

print(result.answer)
```

### Observability

Every `PredictRLM` call returns a structured `prediction.trace` with iterations,
code, output, tool calls, `predict()` subcalls, timings, token usage, and errors.
Human-readable colored trace blocks are printed to stderr by default; pass
`verbose=False` for quiet execution. Use `debug=True` for timestamped RLM and
sandbox lifecycle diagnostics; error-like debug records are colored red.

### Optional: Docker Sandboxes backend

JSPI/Deno/Pyodide remains the default sandbox. After installing
`predict-rlm[sbx]`, use Docker Sandboxes (`sbx`) when you want an explicit
opt-in Linux Python runner:

```bash
brew install docker/tap/sbx
sbx login
```

```python
from predict_rlm import PredictRLM, SbxConfig, SbxPool

rlm = PredictRLM(
    "question -> answer",
    sandbox_backend="sbx",
    sbx_config=SbxConfig(name="my-predict-rlm-sbx"),
)
```

By default, `SbxConfig` passes the explicit non-Docker shell template
`docker.io/docker/sandbox-templates:shell` to `sbx create`. Pass a custom
`template="..."` to override it, or `template=None` to omit `--template` and use
Docker's CLI default behavior.

For throughput-sensitive evals or optimization loops, create a pool of prewarmed
runners and pass it explicitly:

```python
with SbxPool(size=4, config=SbxConfig()) as pool:
    rlm = PredictRLM(
        "question -> answer",
        sandbox_backend="sbx",
        sbx_pool=pool,
    )
```

The backend mounts only a per-run staging directory under `.predict_rlm_sbx/` by
default, preserving model-facing paths such as `/sandbox/input/...` and
`/sandbox/output/...` without exposing the rest of the repo workspace. Use
`SbxConfig(extra_workspaces=[...])` only when the sandbox needs explicit
additional host mounts. Real `sbx` integration tests are skipped by default; run
them with `PREDICT_RLM_RUN_SBX_TESTS=1 uv run pytest -m sbx` after the CLI is
installed and logged in.

The native execution stack uses a persistent supervisor plus a persistent Python kernel so successful iterations preserve full REPL state. Per-iteration timeouts are recoverable when the backend can interrupt execution cleanly; native hard-kill fallback restores only a pre-timeout pickleable snapshot and tells the RLM which globals / imports were lost.

See [predict-rlm Architecture](ARCHITECTURE.md) for the component model,
timeout behavior, and shared backend contracts.

### Using the spreadsheet skill

The optimized spreadsheet skill is built in. Import it and pass it through
`skills=[spreadsheet]` so the RLM gets the spreadsheet-specific instructions,
`openpyxl`, `pandas`, `formulas`, and the `formula_eval` verification module
inside its sandbox.

```python
import dspy

from predict_rlm import File, PredictRLM
from predict_rlm.skills import spreadsheet


class UpdateWorkbook(dspy.Signature):
    """Update the workbook following the request.

    Use openpyxl for workbook edits, Excel formulas for derived values, and
    verify formulas before returning the final .xlsx file.
    """

    workbook: File = dspy.InputField(desc="Input .xlsx workbook")
    request: str = dspy.InputField(desc="Requested spreadsheet changes")
    updated_workbook: File = dspy.OutputField(desc="Updated .xlsx workbook")


rlm = PredictRLM(
    UpdateWorkbook,
    lm="openai/gpt-5.4",
    sub_lm="openai/gpt-5.1",
    skills=[spreadsheet],
)

result = rlm(
    workbook=File(path="model.xlsx"),
    request="Add a Summary sheet with revenue by quarter and formulas for totals.",
)

print(result.updated_workbook.path)
```

For workflows that combine source documents and spreadsheets, compose skills:

```python
from predict_rlm.skills import pdf, spreadsheet

rlm = PredictRLM(ProcessInvoices, skills=[pdf, spreadsheet])
```

## Lifecycle callbacks

Hook into the RLM iteration loop to broadcast progress to a UI, write
structured logs, or feed an observability pipeline. `PredictRLM` extends
DSPy's existing callback contract (`dspy.utils.callback.BaseCallback`)
with two RLM-specific handlers:

| Handler | Fires | Receives |
|---|---|---|
| `on_rlm_iteration_start` | Before the action LM is called for iteration N | `call_id`, `instance`, `iteration`, `max_iterations` |
| `on_rlm_iteration_end` | After iteration N finishes (code executed, `IterationStep` built — or an error was raised) | `call_id`, `instance`, `iteration`, `step: IterationStep \| None`, `is_final: bool`, `exception: Exception \| None` |

`call_id` matches the parent module's `on_module_start/end` ID, so events
correlate cleanly with DSPy's own callback events when you invoke the RLM
through DSPy's public module call path (`rlm(...)` or `await rlm.acall(...)`).
Existing `BaseCallback` subclasses keep working unchanged — handlers we
call are opt-in via `getattr`.

**Async-aware.** If you use `await rlm.acall(...)` your handlers may be
coroutines and they will be awaited. Sync `rlm(...)` calls sync handlers;
if it encounters an async handler the coroutine is closed and a warning
is logged.

**Failure-isolated.** Handler exceptions are logged and swallowed — a
broken callback can never break the run.

### Broadcasting a "loading" status to a websocket

```python
import json
from dspy.utils.callback import BaseCallback
from predict_rlm import IterationStep, PredictRLM

class ProgressBroadcaster(BaseCallback):
    def __init__(self, websocket):
        self.ws = websocket

    async def on_rlm_iteration_start(self, *, iteration, max_iterations, **_):
        await self.ws.send_json({
            "type": "iteration_start",
            "iteration": iteration,
            "max_iterations": max_iterations,
        })

    async def on_rlm_iteration_end(
        self, *, iteration, step: IterationStep | None, is_final, exception, **_
    ):
        await self.ws.send_json({
            "type": "iteration_end",
            "iteration": iteration,
            "is_final": is_final,
            "step": step.model_dump(mode="json") if step else None,
            "error": str(exception) if exception else None,
        })

rlm = PredictRLM("query -> answer")
rlm.callbacks = [ProgressBroadcaster(ws)]
result = await rlm.acall(query="...")
```

Register globally instead with `dspy.configure(callbacks=[...])` and the
same handlers fire for every `PredictRLM` instance.

## Next steps

- [How it works](docs/how-it-works.md) — understand the sandbox, REPL loop,
  signatures, and file I/O
- [Architecture](ARCHITECTURE.md) — compare backend component and
  state/recovery semantics
- [API reference](docs/api.md) — constructor params for `PredictRLM`, `File`,
  and `Skill`
- [Skills](docs/skills.md) — define, compose, and mount custom skills
- [RLM-GEPA](src/rlm_gepa/README.md) — optimize RLM skills from traces and
  configure `AgentSpec`
- [Examples](examples/) — end-to-end demos with setup instructions
