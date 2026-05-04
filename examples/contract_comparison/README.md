# Contract Comparison

Compare two contract versions and produce a structured diff report with per-section analysis.

## Setup

```bash
git clone https://github.com/Trampoline-AI/predict-rlm.git
cd predict-rlm
uv sync --extra examples
export OPENAI_API_KEY=sk-...
```

## Usage

Requires at least 2 PDF files.

```bash
# Run with the included sample contracts
uv run examples/contract_comparison/run.py

# Pass your own files
uv run examples/contract_comparison/run.py v1.pdf v2.pdf
uv run examples/contract_comparison/run.py /path/to/contracts/

# With debug output (prints REPL code and tool calls)
uv run examples/contract_comparison/run.py --debug
```

### Options

| Flag               | Default          | Description                   |
| ------------------ | ---------------- | ----------------------------- |
| `--model`          | `openai/gpt-5.4` | Main LM                       |
| `--sub-lm-model`   | `openai/gpt-5.1` | Sub-LM for `predict()` calls  |
| `--max-iterations` | `30`             | Max REPL iterations           |
| `--debug`          | off              | Print REPL activity to stderr |

Outputs are saved to `output/{timestamp}/` inside this directory.

## Sample output

The [`sample/`](sample/) directory contains two versions of a microFIT contract (45 pages total) and the [comparison report](sample/output/comparison-report.md).

## Structure

| File                           | Purpose                                                         |
| ------------------------------ | --------------------------------------------------------------- |
| [`schema.py`](schema.py)       | Pydantic models for comparison results (diffs, key differences) |
| [`signature.py`](signature.py) | DSPy Signature with comparison instructions                     |
| [`service.py`](service.py)     | DSPy Module wiring PredictRLM with skills                       |
| [`run.py`](run.py)             | CLI entry point                                                 |
