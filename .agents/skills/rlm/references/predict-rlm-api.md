# PredictRLM API Reference

Use this reference to keep generated code aligned with the package API.

## Core Imports

```python
from predict_rlm import File, PredictRLM, Skill
from predict_rlm.skills import docx, pdf, spreadsheet
```

## PredictRLM

```python
PredictRLM(
    signature: type[Signature] | str,
    lm: dspy.LM | str | None = None,
    sub_lm: dspy.LM | str | None = None,
    max_iterations: int = 30,
    max_llm_calls: int = 50,
    max_output_chars: int = 100_000,
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
)
```

Both `lm` and `sub_lm` accept a model string or a `dspy.LM` instance. If `lm` is
omitted, the current `dspy.context(lm=...)` LM is used.

## File I/O

Use `File` for large inputs and generated artifacts.

- Input fields mount host files under `/sandbox/input/{field_name}/`.
- Output fields sync files from `/sandbox/output/{field_name}/` back to the host.

## Skills

```python
Skill(
    name="my-skill",
    instructions="How to approach the domain...",
    packages=["pandas", "openpyxl"],
    modules={"helper": "/path/to/helper.py"},
    tools={"fetch": fetch_fn},
)
```

Skills bundle reusable instructions, sandbox packages, mounted modules, and
host-side tools. When skills are composed, instructions concatenate, packages
deduplicate, and tool-name conflicts raise errors.

Built-in skills:

| Skill | Import | Packages | Modules | Purpose |
| --- | --- | --- | --- | --- |
| `pdf` | `from predict_rlm.skills import pdf` | `pymupdf` | - | Read, render, modify, and redact PDFs |
| `spreadsheet` | `from predict_rlm.skills import spreadsheet` | `openpyxl`, `pandas`, `formulas` | `formula_eval` | Build and modify Excel workbooks |
| `docx` | `from predict_rlm.skills import docx` | `python-docx` | `md2docx` | Read, write, and modify Word documents |

## Host-Side Tools

Tools are host-side callables the RLM can invoke from the sandbox. Use them for
operations that need host access, authenticated APIs, databases, native
libraries, or filesystem-heavy work.

```python
async def fetch_exchange_rate(currency: str, date: str) -> str:
    """Fetch the exchange rate for a currency on a given date.

    Args:
        currency: ISO currency code, e.g. "EUR".
        date: Date in YYYY-MM-DD format.

    Returns:
        JSON string with exchange-rate data.
    """
    ...
```

Pass tools directly via `tools={"name": fn}` or bundle reusable tools in a
`Skill`.

## predict() Inside The Sandbox

The RLM can call `predict()` for sub-LM perception or extraction. Each call gets
its own context window.

```python
result = await predict(
    "image: dspy.Image -> items: list[Item]",
    instructions="Extract all line items from this invoice page",
    image=page_image,
)
```

Use `dspy.Image` for multimodal image inputs.

## CodexLM

When the user wants PredictRLM to run on Codex/ChatGPT subscription auth instead
of ordinary OpenAI API keys, use `predict-rlm[codex-lm]` and `CodexLM`.

```bash
uv add "predict-rlm[codex-lm]"
uv run codex-lm auth login default
uv run codex-lm auth status
uv run codex-lm smoke-test --model gpt-5.5
```

```python
from dspy_codex_lm import CodexLM
from predict_rlm import PredictRLM

rlm = PredictRLM(
    MySignature,
    lm=CodexLM(model="gpt-5.5"),
    sub_lm=CodexLM(model="gpt-5.5"),
)
```

`CodexLM` uses Codex/ChatGPT subscription auth, not ordinary OpenAI API keys.
Routing is strict: unsupported Codex slugs should fail instead of silently
falling back.
