# Contract Comparison

Compare two contract versions and produce a structured diff report with
per-section analysis.

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

Requires at least 2 PDF files.

```bash
# Run with the included sample contracts
uv run examples/contract_comparison/run.py

# Pass your own files
uv run examples/contract_comparison/run.py v1.pdf v2.pdf
uv run examples/contract_comparison/run.py /path/to/contracts/

# Suppress verbose RLM trace blocks
uv run examples/contract_comparison/run.py --quiet

# With timestamped lifecycle diagnostics
uv run examples/contract_comparison/run.py --debug
```

### Options

| Flag               | Default          | Description                   |
| ------------------ | ---------------- | ----------------------------- |
| `--model`          | `gpt-5.6-terra`  | Main Codex model ID (no provider prefix) |
| `--sub-lm-model`   | `gpt-5.6-terra`  | Codex sub-LM for `predict()` calls |
| `--max-iterations` | `30`             | Max REPL iterations           |
| `--quiet`          | off              | Suppress RLM reasoning, code, output, tool calls, errors, and submit blocks |
| `--debug`          | off              | Print timestamped RLM and sandbox lifecycle diagnostics to stderr |

Outputs are saved to `output/{timestamp}/` inside this directory.

Every CLI or service invocation automatically saves `trace.json` and
`evidence.json` to `.run/<run_id>/` inside this example directory, including
failed or cancelled runs. `<run_id>` is the evidence run ID; `trace.json` is
JSON `null` when no trace is available. `.run/` is ignored by Git by default.

## Sample output

The [`sample/`](sample/) directory contains two versions of a microFIT contract
(45 pages total) and the
[comparison report](sample/output/comparison-report.md).

These are historical sample results; recorded model labels and costs are
unchanged and do not describe the current Codex defaults.

## Structure

| File                           | Purpose                                                         |
| ------------------------------ | --------------------------------------------------------------- |
| [`schema.py`](schema.py)       | Pydantic models for comparison results (diffs, key differences) |
| [`signature.py`](signature.py) | DSPy Signature with comparison instructions                     |
| [`service.py`](service.py)     | DSPy Module wiring PredictRLM with skills                       |
| [`run.py`](run.py)             | CLI entry point                                                 |
