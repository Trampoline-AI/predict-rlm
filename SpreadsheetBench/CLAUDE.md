# SpreadsheetBench

Benchmark for evaluating spreadsheet manipulation tasks. We use an RLM (Recursive Language Model) via `predict-rlm` instead of direct LLM API calls.

## Setup

```bash
uv venv && uv pip install tqdm pandas openpyxl openai docker numpy predict-rlm
```

Always use `uv` for package management (not pip directly).

## Project Structure

- `spreadsheet_rlm/` — RLM package (signature, service) that takes a spreadsheet File + instruction and outputs a modified spreadsheet File
- `run_bench.py` — Runner script that iterates dataset cases through the RLM
- `evaluate_rlm.py` — Evaluation script that only scores tasks with outputs (fixes upstream bug where `evaluation/evaluation.py` compares input vs answer instead of output vs answer)
- `evaluation/` — Upstream evaluation code (do not modify)
- `inference/` — Upstream inference code (not used; we bypass it with the RLM)
- `data/` — Datasets with tar.gz archives. Extract with `tar -xzf` before use.

## One-Time Setup: Recalculate Answer Files

The benchmark answer files contain Excel formulas without cached values. openpyxl reads `None` for unevaluated formulas, causing false mismatches during evaluation. Run LibreOffice on the answer files once to cache computed values:

```bash
uv run python evaluation/open_spreadsheet.py --dir_path data/sample_data_200/spreadsheet --recursive
```

This modifies the answer files in-place. Do the same for any other dataset you plan to evaluate on (`spreadsheetbench_verified_400`, `all_data_912_v0.1`).

## Running the Benchmark

Three steps, in order:

### 1. Run inference with RLM

```bash
uv run python run_bench.py --dataset sample_data_200 --limit 5
```

Key flags: `--limit N` to cap tasks, `--overwrite` to redo existing outputs.

Outputs go to `data/{dataset}/outputs/single_rlm/`.

Verbose and debug logging is always enabled and written to per-test-case log files under `logs/run_YYYYMMDD_HHMMSS/`. The terminal only shows tqdm progress and the run stats summary.

### 2. Recalculate formulas with LibreOffice

Required because the RLM writes Excel formulas, and openpyxl can only read cached values. LibreOffice opens each file to force recalculation.

```bash
uv run python evaluation/open_spreadsheet.py --dir_path data/sample_data_200/outputs/single_rlm
```

### 3. Evaluate

Pass `--log_dir` pointing to the same run log folder to get per-test-case eval logs with pass/fail reasons.

```bash
uv run python evaluate_rlm.py --setting single --model rlm --dataset sample_data_200 --log_dir logs/run_20260407_141500
```

Results saved to `outputs/eval_single_rlm.json`. Prints soft/hard restriction averages and pass counts.

## Logs

Each run creates a folder under `logs/`, with a subfolder per task containing run and eval logs for each test case:

```
logs/run_20260407_141500/
├── summary.log              # Run stats: cost, tokens, duration
├── 59196/
│   ├── run_1.log            # RLM verbose/debug output for test case 1
│   ├── run_2.log            # RLM output for test case 2
│   ├── run_3.log            # RLM output for test case 3
│   ├── eval_1.log           # Evaluation result: PASS/FAIL + reason
│   ├── eval_2.log           # e.g. "FAIL\nreason: Value difference at cell H4: ..."
│   └── eval_3.log
├── 99-24/
│   ├── run_1.log
│   ├── run_2.log
│   ├── run_3.log
│   ├── eval_1.log
│   ├── eval_2.log
│   └── eval_3.log
└── ...
```

- `run_{idx}.log` — RLM verbose iteration reasoning/code and debug sandbox output
- `eval_{idx}.log` — PASS or FAIL with the specific cell mismatch reason
- `summary.log` — per-LM call counts, token usage, cost breakdown, duration

## Datasets

- `sample_data_200` — 200 tasks, good for testing
- `spreadsheetbench_verified_400` — 400 expert-verified tasks
- `all_data_912_v0.1` — Full 912-task benchmark

Each task has 3 test cases (input/answer pairs). Evaluation uses OJ-style scoring: soft = avg pass rate, hard = all 3 must pass.

## RLM Configuration

Models are configured in `run_bench.py` as `LM` and `SUB_LM` constants. The RLM uses the built-in `spreadsheet` skill (openpyxl, pandas, formulas) in a sandboxed Pyodide environment.
