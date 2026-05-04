# Document Analysis

Analyze documents and extract key dates, entities, and financial information into a structured report.

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
uv run examples/document_analysis/run.py

# Pass your own files
uv run examples/document_analysis/run.py /path/to/doc.pdf
uv run examples/document_analysis/run.py /path/to/docs/

# With debug output (prints REPL code and tool calls)
uv run examples/document_analysis/run.py --debug
```

### Options

| Flag               | Default          | Description                   |
| ------------------ | ---------------- | ----------------------------- |
| `--model`          | `openai/gpt-5.4` | Main LM                       |
| `--sub-lm-model`   | `openai/gpt-5.1` | Sub-LM for `predict()` calls  |
| `--max-iterations` | `30`             | Max REPL iterations           |
| `--debug`          | off              | Print REPL activity to stderr |

Outputs are saved to `output/{timestamp}/` inside this directory.

## How it works

The RLM receives `File` references as input. The files are mounted into the sandbox, and the RLM opens them directly with pymupdf. The RLM **manages its own context window** — given a 200-page document set, it doesn't process everything at once. Instead, it:

1. **Surveys** the documents — checks file names and page counts to understand the structure
2. **Samples** strategically — renders a few pages to understand the format and identify where key information lives
3. **Extracts in parallel** — uses `asyncio.gather()` to send multiple pages to `predict()` concurrently
4. **Synthesizes** — aggregates findings across pages, deduplicates, and produces the final structured output

## Sample output

The [`sample/`](sample/) directory contains a 136-page airport parking management RFP and the [output report](sample/output/report.md) produced by the RLM.

|               | Main LM (`gpt-5.4`) | Sub-LM (`gpt-5.1`) |
| ------------- | ------------------- | ------------------ |
| Calls         | 8                   | 63                 |
| Input tokens  | 93,571              | 69,986             |
| Output tokens | 9,241               | 21,274             |
| Cost          | $0.22               | $0.30              |

**136 pages analyzed in ~4 minutes for $0.52 total ($0.004/page).**

## Structure

| File                           | Purpose                                                  |
| ------------------------------ | -------------------------------------------------------- |
| [`schema.py`](schema.py)       | Pydantic models for the output (dates, entities, report) |
| [`signature.py`](signature.py) | DSPy Signature with task instructions                    |
| [`service.py`](service.py)     | DSPy Module wiring PredictRLM with skills                |
| [`run.py`](run.py)             | CLI entry point                                          |
