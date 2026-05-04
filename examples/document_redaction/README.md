# Document Redaction

Redact PII from PDFs based on a policy, then verify the redactions visually.

## Setup

```bash
git clone https://github.com/Trampoline-AI/predict-rlm.git
cd predict-rlm
uv sync --extra examples
export OPENAI_API_KEY=sk-...
```

## Usage

```bash
# Run with the included sample PDF
uv run examples/document_redaction/run.py

# Pass your own files
uv run examples/document_redaction/run.py /path/to/doc.pdf
uv run examples/document_redaction/run.py /path/to/docs/

# Custom redaction criteria
uv run examples/document_redaction/run.py --criteria "Redact all financial amounts and account numbers"

# With debug output (prints REPL code and tool calls)
uv run examples/document_redaction/run.py --debug
```

### Options

| Flag               | Default          | Description                   |
| ------------------ | ---------------- | ----------------------------- |
| `--model`          | `openai/gpt-5.4` | Main LM                       |
| `--sub-lm-model`   | `openai/gpt-5.1` | Sub-LM for `predict()` calls  |
| `--max-iterations` | `30`             | Max REPL iterations           |
| `--criteria`       | PII redaction    | What to redact                |
| `--debug`          | off              | Print REPL activity to stderr |

Outputs (redacted PDFs + report) are saved to `output/{timestamp}/` inside this directory.

## How it works

The RLM is an **autonomous executor that modifies files**. It inspects pages, identifies sensitive content, applies redactions, and then re-inspects the pages to verify the redactions worked.

1. **Scans pages in parallel** — renders pages as images and fans out `predict()` calls via `asyncio.gather()` to identify text matching the redaction criteria
2. **Applies redactions** — uses pymupdf's `search_for()` and `add_redact_annot()` to black out identified strings
3. **Handles non-text content** — for signatures, logos, or images, it estimates bounding box coordinates and redacts by area
4. **Verifies** — re-renders redacted pages and confirms the sensitive content is gone

Redacted PDFs are written to a `list[File]` output and synced back to the host automatically.

## Sample output

The [`sample/`](sample/) directory contains a 6-page mock employment agreement and the [redaction output](sample/output/output.md) — 96 redactions across 6 categories.

|               | Main LM (`gpt-5.4`) | Sub-LM (`gpt-5.1`) |
| ------------- | ------------------- | ------------------ |
| Calls         | 6                   | 20                 |
| Input tokens  | 55,432              | 19,572             |
| Output tokens | 5,866               | 6,905              |
| Cost          | $0.14               | $0.09              |

**6 pages fully redacted in ~2 minutes for $0.24 total.**

## Structure

| File                           | Purpose                                                 |
| ------------------------------ | ------------------------------------------------------- |
| [`schema.py`](schema.py)       | Pydantic models for redaction results                   |
| [`signature.py`](signature.py) | DSPy Signature with redaction instructions              |
| [`skills.py`](skills.py)       | Custom redaction skill (pymupdf redaction API patterns) |
| [`service.py`](service.py)     | DSPy Module wiring PredictRLM with skills               |
| [`run.py`](run.py)             | CLI entry point                                         |
