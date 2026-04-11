# predict-rlm
Production focused port of RLMs that allows the LM to call its sub-lm with [DSPy](https://dspy.ai) signatures. Define your inputs, outputs, and tools — the model handles its own control flow. Get fully interpretable trajectories and performance that scales directly with model improvements. Without context rot.

Based on the [Recursive Language Models](https://arxiv.org/abs/2512.24601v1) paper by [Alex L. Zhang](https://x.com/a1zhang), [Tim Kraska](https://x.com/tim_kraska), and [Omar Khattab](https://x.com/lateinteraction) from the Stanford NLP lab.<br/>

<br>
<p align="center">
  <a href="https://github.com/Trampoline-AI/predict-rlm/actions/workflows/tests.yml"><img src="https://img.shields.io/github/actions/workflow/status/Trampoline-AI/predict-rlm/tests.yml?label=Tests" alt="Tests"></a>
  <a href="https://codecov.io/gh/Trampoline-AI/predict-rlm"><img src="https://img.shields.io/codecov/c/github/Trampoline-AI/predict-rlm?token=NNS3R7OIT2&color=brightgreen&label=codecov" alt="codecov"></a>
  <a href="https://pypi.org/project/predict-rlm/"><img src="https://img.shields.io/pypi/v/predict-rlm?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/predict-rlm/"><img src="https://img.shields.io/pypi/pyversions/predict-rlm" alt="Python"></a>
  <a href="https://discord.gg/BAkd288sGN"><img src="https://img.shields.io/badge/Discord-Join-5865F2?style=flat&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/Trampoline-AI/predict-rlm"><img src="https://img.shields.io/github/stars/trampoline-ai/predict-rlm?cacheSeconds=3600" alt="GitHub stars"></a>
  <br/>
  crafted  with ♥ in MTL · NYC · FLP<br>by <a href="https://trampoline.ai">Trampoline AI</a> 
</p>

## Installation

```bash
uv add predict-rlm
```

## Why RLMs?

<p align="center">
  <img src="https://raw.githubusercontent.com/Trampoline-AI/predict-rlm/main/docs/bitter_lesson_spectrum.svg" alt="Bitter Lesson Spectrum — from hand-written prompts to RLMs" width="680"/>
</p>

- **Avoid context rot** —  The root LM only interacts with its context programmatically through the REPL, staying well within its comfortable operating range — enabling complex, long-horizon tasks that would otherwise cause models to silently degrade.
- **Bitter lesson-proof: RLMs improve as LMs improve** — Unlike harnesses, which can cap or constrain the base model's capabilities, the performance, speed, and cost of RLM calls correlate directly with improvements to base model capabilities. [If the base model handles 10M tokens tomorrow, the RLM handles 100M.](https://alexzhang13.github.io/blog/2025/rlm/)
- **Symbolic reasoning & recursion** — like algebra, RLMs express the *structure* of computation rather than performing each operation individually; a single line can represent 1M sub-calls — in direct contrast to agents like Claude Code that must mechanically emit each sub-agent call one at a time.
- **Interpretability** — RLM trajectories are fully readable: you can trace every peek, chunk, sub-call, and verification step the model takes. This not only reveals *how* the model decomposed a problem, but provides concrete optimization signals which tools like [GEPA](https://gepa-ai.github.io/gepa) can ingest to evolve the RLM's strategies.
- **Ideal for improving performance per token** — RLMs allow small models to punch way above their weight (RLM(GPT-5-mini) outperforms base GPT-5) providing great opportunities for reducing costs or stretching limited compute budgets without sacrificing quality.

## Features

<p align="center">
  <img src="https://raw.githubusercontent.com/Trampoline-AI/predict-rlm/main/docs/harness_vs_rlm.svg" alt="Classic harness vs RLM architecture" width="600"/>
</p>


- **Multimodal** — process images, documents, audio, and video through sub-LM calls using native provider multimodal APIs. 
- **Async tool calling** — native RLM async support in the WASM sandbox, enabling concurrent sub-LM invocations and tool calls
- **Prompt-optimized skills & tools** — predic-rlm skills comes tested and optimized to ensure maximum LM interoperability and performance, bundling instructions, PyPI packages, and tools for domain-specific tasks
- **Simple file I/O** — pass local or cloud files as typed inputs and outputs via `File`, keeping interop with your existing data pipelines straightforward. (S3 files support soon)
- **Structured sub-LM calls** — native Pydantic and DSPy signature support for type-safe sub-LM invocations with structured outputs

## Demos

| Description | Input / Output | Preview |
|---|---|---|
| [Document Analysis](examples/document_analysis/) — Analyze documents and extract key dates, entities, and financial information into a structured report | **Input:** PDFs<br>**Output:** Structured briefing report ([example output](examples/document_analysis/sample/output/report.md)) | <a href="examples/document_analysis/sample/output/report.md"><img src="https://raw.githubusercontent.com/Trampoline-AI/predict-rlm/main/examples/document_analysis/sample/output/screenshot.png" width="280"></a> |
| [Document Redaction](examples/document_redaction/) — Redact PII from PDFs based on a policy, then verify the redactions visually | **Input:** PDFs<br>**Output:** Redacted PDFs ([example output](examples/document_redaction/sample/output/output.md)) | <a href="examples/document_redaction/sample/output/output.md"><img src="https://raw.githubusercontent.com/Trampoline-AI/predict-rlm/main/examples/document_redaction/sample/output/screenshot.png" width="280"></a> |
| [Invoice Processing](examples/invoice_processing/) — Extract vendor info, line items, and totals from PDF invoices into a consolidated Excel spreadsheet | **Input:** PDF invoices<br>**Output:** Excel spreadsheet ([example output](examples/invoice_processing/sample/output/)) | <a href="examples/invoice_processing/sample/output/output.md"><img src="https://raw.githubusercontent.com/Trampoline-AI/predict-rlm/main/examples/invoice_processing/sample/output/screenshot.png" width="280"></a> |
| [Contract Comparison](examples/contract_comparison/) — Compare two contract versions and produce a structured diff report with per-section analysis | **Input:** 2 PDF contracts<br>**Output:** Structured diff report ([example output](examples/contract_comparison/sample/output/)) | <a href="examples/contract_comparison/sample/output/comparison-report.md"><img src="https://raw.githubusercontent.com/Trampoline-AI/predict-rlm/main/examples/contract_comparison/sample/output/screenshot.png" width="280"></a> |

## Quick start

### With your coding agent

Install the [predict-rlm skill](.agents/skills/rlm/SKILL.md) in Claude Code, Codex, Cursor, or any compatible coding agent:

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

## Next steps

- [How it works](docs/how-it-works.md) — understand the sandbox, REPL loop, signatures, and file I/O
- [API reference](docs/api.md) — constructor params for `PredictRLM`, `File`, and `Skill`
- [Skills](docs/skills.md) — define, compose, and mount custom skills
- [Examples](examples/) — end-to-end demos with setup instructions
