# Invoice Processing

Extract vendor info, line items, and totals from PDF invoices into a
consolidated Excel spreadsheet.

## Setup

```bash
git clone https://github.com/Trampoline-AI/predict-rlm.git
cd predict-rlm
uv sync --extra examples --extra codex-lm
uv run codex-lm auth import default
# Or log in instead of importing existing Codex credentials:
# uv run codex-lm auth login default --device-auth
uv run codex-lm auth use default
```

The runner uses the bundled `dspy_codex_lm.CodexLM` directly with your ChatGPT
subscription. No OpenAI API key or external `codex-lm` execution wrapper is needed.

## Usage

```bash
# Run with the included sample invoices
uv run examples/invoice_processing/run.py

# Pass your own files
uv run examples/invoice_processing/run.py invoice1.pdf invoice2.pdf
uv run examples/invoice_processing/run.py /path/to/invoices/

# Suppress verbose RLM trace blocks
uv run examples/invoice_processing/run.py --quiet

# With timestamped lifecycle diagnostics
uv run examples/invoice_processing/run.py --debug
```

### Options

| Flag               | Default          | Description                   |
| ------------------ | ---------------- | ----------------------------- |
| `--model`          | `gpt-5.5`        | Main Codex model ID (no provider prefix) |
| `--sub-lm-model`   | `gpt-5.4-mini`   | Codex sub-LM for `predict()` calls |
| `--max-iterations` | `30`             | Max REPL iterations           |
| `--quiet`          | off              | Suppress RLM reasoning, code, output, tool calls, errors, and submit blocks |
| `--debug`          | off              | Print timestamped RLM and sandbox lifecycle diagnostics to stderr |

Excel workbooks are saved to `output/{timestamp}/` inside this directory.

Every CLI or service invocation automatically saves `trace.json` and
`evidence.json` to `.run/<run_id>/` inside this example directory, including
failed or cancelled runs. `<run_id>` is the evidence run ID; `trace.json` is
JSON `null` when no trace is available. `.run/` is ignored by Git by default.

## Sample output

The [`sample/`](sample/) directory contains 2 PDF invoices and the
[extracted output](sample/output/) — structured data plus a consolidated Excel
workbook.

These are historical sample results; recorded model labels and costs are
unchanged and do not describe the current Codex defaults.

## Structure

| File                           | Purpose                                               |
| ------------------------------ | ----------------------------------------------------- |
| [`schema.py`](schema.py)       | Pydantic models for invoice data (line items, totals) |
| [`signature.py`](signature.py) | DSPy Signature with extraction instructions           |
| [`service.py`](service.py)     | DSPy Module wiring PredictRLM with skills             |
| [`run.py`](run.py)             | CLI entry point                                       |
