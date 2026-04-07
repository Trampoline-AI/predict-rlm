# predict-rlm

[![Tests](https://img.shields.io/github/actions/workflow/status/Trampoline-AI/predict-rlm/tests.yml?label=Tests)](https://github.com/Trampoline-AI/predict-rlm/actions/workflows/tests.yml)
[![codecov](https://img.shields.io/codecov/c/github/Trampoline-AI/predict-rlm?token=NNS3R7OIT2&color=brightgreen&label=codecov)](https://codecov.io/gh/Trampoline-AI/predict-rlm)
[![PyPI](https://img.shields.io/pypi/v/predict-rlm?color=blue)](https://pypi.org/project/predict-rlm/)
[![Python](https://img.shields.io/pypi/pyversions/predict-rlm)](https://pypi.org/project/predict-rlm/)
[![Discord](https://img.shields.io/badge/Discord-Join-5865F2?style=flat&logo=discord&logoColor=white)](https://discord.gg/BAkd288sGN)
[![GitHub stars](https://img.shields.io/github/stars/trampoline-ai/predict-rlm)](https://github.com/Trampoline-AI/predict-rlm)

Production-grade [Recursive Language Models](https://arxiv.org/abs/2512.24601v1).<br/>
Based on the paper by [Alex L. Zhang](https://x.com/a1zhang), [Tim Kraska](https://x.com/tim_kraska), and [Omar Khattab](https://x.com/lateinteraction) from the Stanford NLP lab.

## Installation

```bash
uv add predict-rlm
```

Or with pip:

```bash
pip install predict-rlm
```


## Why RLMs?

Think of an RLM as a **callable, pre-configured agent**. Like Claude Code or Cursor, it can autonomously explore context, write and execute code, call tools, inspect results, and iterate until the task is done. Unlike a chat agent, an RLM is a **function** — you define its inputs, outputs, and tools, then call it from your code. It returns structured data, not chat messages.

This makes RLMs ideal for tasks that are:

- **Specific and repeatable** — tasks with a well-defined SOP and a known desired outcome. Think of an RLM as a Claude Code that's been purpose-built for one task — with the right tools, the right instructions, and a tuned workflow that reliably produces the result you want. You define the procedure once, and the RLM follows it every time.
- **Context-heavy** — too much data to fit in a single prompt. The RLM selectively loads what it needs via tools, working through documents page by page rather than stuffing everything into one call.
- **Multi-step** — require exploring, extracting, computing, and synthesizing. The RLM writes code to orchestrate these steps, parallelizing where possible (e.g. processing 50 pages concurrently with `asyncio.gather()`).
- **Action-oriented** — need to make changes, not just read. By giving the RLM tools that modify state (redact text, call APIs, write files), it becomes an autonomous executor — not just an analyzer.
- **Iterative** — the RLM can inspect its own results, catch errors, retry with different approaches, and verify its work before submitting. It self-corrects in ways a single LLM call cannot.

## What is predict-rlm?

`predict-rlm` extends DSPy's RLM with a built-in `predict()` tool — a sub-LM the RLM can call from within its sandbox to perform language understanding, vision analysis, and structured extraction via DSPy signatures.

The architecture is two-level:

1. **The outer LLM** (the RLM itself) writes and executes Python code in a sandboxed REPL. It plans, orchestrates, and iterates.
2. **The sub-LM** (via `predict()`) handles perception and extraction — analyzing images, understanding text, and returning typed results.

The sub-LM supports `dspy.Image` type hints, which means `predict()` calls can pass images (as URLs or base64) directly to a vision-capable model. This makes RLMs **natively multimodal** — the outer LLM renders a PDF page to an image, passes it to `predict()`, and gets back structured data. The RLM itself doesn't need to be a vision model; it delegates visual understanding to the sub-LM.

The outer LLM decides *what* to look at and *when*; the sub-LM decides *what it sees*. This separation is key to context management — the outer LLM's context stays small (code + tool results), while context-heavy work like reading a full page image or analyzing a long text block is offloaded to `predict()` calls. Each `predict()` call gets its own context window with the sub-LM, so the RLM can process far more total data than any single LLM call could hold.

## Demos

| Description | Input / Output | Preview |
|---|---|---|
| [Document Analysis](examples/document_analysis/) — Analyze documents and extract key dates, entities, and financial information into a structured report | **Input:** PDFs<br>**Output:** Structured briefing report ([example output](examples/document_analysis/sample/output/report.md)) | <a href="examples/document_analysis/sample/output/report.md"><img src="https://raw.githubusercontent.com/Trampoline-AI/predict-rlm/main/examples/document_analysis/sample/output/screenshot.png" width="280"></a> |
| [Document Redaction](examples/document_redaction/) — Redact PII from PDFs based on a policy, then verify the redactions visually | **Input:** PDFs<br>**Output:** Redacted PDFs ([example output](examples/document_redaction/sample/output/output.md)) | <a href="examples/document_redaction/sample/output/output.md"><img src="https://raw.githubusercontent.com/Trampoline-AI/predict-rlm/main/examples/document_redaction/sample/output/screenshot.png" width="280"></a> |
| [Invoice Processing](examples/invoice_processing/) — Extract vendor info, line items, and totals from PDF invoices into a consolidated Excel spreadsheet | **Input:** PDF invoices<br>**Output:** Excel spreadsheet ([example output](examples/invoice_processing/sample/output/)) | <a href="examples/invoice_processing/sample/output/output.md"><img src="https://raw.githubusercontent.com/Trampoline-AI/predict-rlm/main/examples/invoice_processing/sample/output/screenshot.png" width="280"></a> |
| [Contract Comparison](examples/contract_comparison/) — Compare two contract versions and produce a structured diff report with per-section analysis | **Input:** 2 PDF contracts<br>**Output:** Structured diff report ([example output](examples/contract_comparison/sample/output/)) | <a href="examples/contract_comparison/sample/output/comparison-report.md"><img src="https://raw.githubusercontent.com/Trampoline-AI/predict-rlm/main/examples/contract_comparison/sample/output/screenshot.png" width="280"></a> |

## Quick start

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

### Use it with your coding agent

Add the [predict-rlm agent skill](skills/create-rlm/SKILL.md) to Claude Code, Codex, Cursor, or any compatible coding agent:

```bash
npx skills add Trampoline-AI/predict-rlm
```

Your agent will then know how to build RLMs using predict-rlm — including the file structure, signatures, tools, and skills patterns.

## Features

- **Built-in `predict()` tool** — call a sub-LM from inside the sandbox with DSPy signatures and type hints
- **JSPI-enabled WASM sandbox** — concurrent async tool execution via Pyodide with `asyncio.gather()`
- **Structured outputs** — Pydantic models, typed fields, and lists as output types
- **Custom tools** — give the RLM tools that read, write, or modify external state
- **Skills** — composable bundles of instructions, PyPI packages, and tools for domain-specific tasks
- **Multimodal** — sub-LM calls support `dspy.Image`, so the RLM can analyze images, PDFs, screenshots, etc. without the outer LLM needing vision capabilities
- **Optimizable** — built on DSPy, so optimizers can tune prompts and few-shot examples automatically. Inference-time scaling techniques like [GEPA](https://arxiv.org/abs/2504.00294) push accuracy further by generating and selecting among multiple candidate solutions

## How it works

1. You define **inputs**, **outputs**, and **tools** — what the RLM receives, what it should produce, and what actions it can take
2. The outer LLM writes Python code in a sandboxed Pyodide/WASM REPL
3. Inside the sandbox, it calls `await predict(signature, **kwargs)` to invoke the sub-LM for understanding and extraction
4. It iterates — exploring data, calling tools, building up intermediate results, and handling errors
5. When done, it calls `SUBMIT()` with the final structured output

Each iteration is a REPL turn: the LLM sees the output of its previous code, decides what to do next, and writes more code. State persists between iterations, so it can accumulate findings across many steps.

### Signatures and file I/O

The DSPy signature defines the **inputs**, **outputs**, and **strategy** (via the docstring). Use `File` for file-typed fields — input files are mounted into the sandbox, output files are synced back (see [API](#file) for details).

```python
from predict_rlm import File, PredictRLM, Skill

class AnalyzeDocuments(dspy.Signature):
    """Analyze documents and produce a structured report.

    1. Survey the documents — file names, page counts, document types
    2. Render pages as images and use predict() to extract content
    3. Produce the report following the criteria's format
    """
    documents: list[File] = dspy.InputField()
    analysis: DocumentAnalysis = dspy.OutputField()

pdf_skill = Skill(
    name="pdf",
    instructions="Use pymupdf to open and render PDF pages...",
    packages=["pymupdf"],
)

rlm = PredictRLM(
    AnalyzeDocuments,
    lm="openai/gpt-5.4",
    sub_lm="openai/gpt-5.1",
    skills=[pdf_skill],
)

documents = [File(path="report.pdf"), File(path="appendix.pdf")]
result = rlm(documents=documents)
```

Inside the sandbox, the RLM autonomously decides which pages to load and when:

```python
# The RLM writes code like this — you don't write this, the LLM does:
import pymupdf, base64, asyncio

doc = pymupdf.open(documents[0])
images = [
    "data:image/png;base64,"
    + base64.b64encode(
        doc[i].get_pixmap(dpi=200).tobytes("png")
    ).decode()
    for i in range(3)
]
results = await asyncio.gather(*[
    predict("page: dspy.Image -> dates: list[str]", page=img)
    for img in images
])
```

## Skills

Skills extend what an RLM can do inside its sandbox — adding PyPI packages, instructions, modules, and tools. predict-rlm ships built-in skills for PDFs, spreadsheets, and Word documents, and you can define your own.

```python
from predict_rlm.skills import pdf, spreadsheet

rlm = PredictRLM(MySignature, skills=[pdf, spreadsheet])
```

See the [skills documentation](docs/skills.md) for details on defining, composing, and mounting custom skills.

## Examples

Each example has its own README with setup and usage instructions. See the individual directories:

- [Document Analysis](examples/document_analysis/) — Extract key dates, entities, and financial info into a structured report
- [Document Redaction](examples/document_redaction/) — Redact PII from PDFs and verify the redactions visually
- [Invoice Processing](examples/invoice_processing/) — Extract line items and totals into an Excel spreadsheet
- [Contract Comparison](examples/contract_comparison/) — Produce a structured diff report between contract versions

## API

See the [API reference](docs/api.md) for `PredictRLM`, `File`, and `Skill`.
