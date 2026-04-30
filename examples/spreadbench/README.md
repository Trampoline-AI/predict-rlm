# SpreadsheetBench RLM-GEPA

Release artifact for the SpreadsheetBench RLM-GEPA numbers in
[`blog_post.md`](blog_post.md). It contains the SpreadsheetBench RLM, dataset
prep, held-out eval CLI, optimizer CLI, and tracked archives for the release
runs.

## Results

All results are on the held-out 400-task SpreadsheetBench Verified set. The
optimized skill was trained only on the non-Verified 512-task remainder.

| Executor | Seed soft | Seed hard | Optimized soft | Optimized hard |
| --- | ---: | ---: | ---: | ---: |
| `gpt-5.4`, low | 0.8710 | 0.7800 | 0.8980 | 0.8150 |
| `gpt-5.4`, medium | 0.8750 | 0.7950 | 0.9259 | 0.8500 |
| `gpt-5.5`, low | 0.9092 | 0.8500 | 0.9288 | 0.8775 |
| `gpt-5.5`, medium | 0.9142 | 0.8600 | 0.9411 | 0.8925 |

## Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/)
- Deno v2, used by the PredictRLM Pyodide/WASM sandbox
- LibreOffice on `PATH`, used for spreadsheet recalculation/rendering
- LiteLLM-compatible provider keys, for example `OPENAI_API_KEY` and/or
  `ANTHROPIC_API_KEY`

Run commands below from `examples/spreadbench`:

```bash
cd examples/spreadbench
uv sync
```

## Set up the data

The source archives are checked in:

- `data/spreadsheetbench_912_v0.1.tar.gz`
- `data/spreadsheetbench_verified_400.tar.gz`

Prepare extracted data and splits:

```bash
make dataset
# equivalent:
# uv run python -m spreadsheet_rlm.bench.prepare_dataset
```

This creates:

- `data/trainset/`: 512 non-Verified tasks, used for optimization.
- `data/testset`: symlink to the 400-task Verified held-out set, used for
  release metrics.

## Unpack release traces

Run artifacts are tracked as tarballs under [`run_artifacts/`](run_artifacts/)
because expanded traces are large and `runs/` is gitignored.

```bash
mkdir -p runs
for archive in run_artifacts/*.tar.gz; do
  tar -xzf "$archive" -C runs
done
```

The main optimization run for the release plots and winning skill is:

```bash
export RUN_DIR=runs/optimize_20260421_135558_gpt-5.4-mini__sub-gpt-5.4-mini__prop-claude-sonnet-4-6
```

Its winning candidate is candidate `4` (`best_idx = 4`).

Useful artifact commands:

```bash
uv run rlm-gepa stats "$RUN_DIR"
uv run rlm-gepa plot "$RUN_DIR"
```

## Re-run the winning held-out eval

This re-evaluates candidate `4` from the release optimization run on the
held-out Verified set using the strongest reported executor setting:

```bash
uv run rlm-gepa eval \
  --dataset testset \
  --run-dir "$RUN_DIR" \
  --cand-idx 4 \
  --lm openai/gpt-5.5 \
  --reasoning-effort medium \
  --sub-lm openai/gpt-5.4-mini
```

Expected aggregate: `0.9411` soft / `0.8925` hard.

For a cheap smoke check, add `--limit 5 --concurrency 5`.

## Re-run other release evals

Use the same candidate and swap executor settings:

```bash
# gpt-5.5 low
uv run rlm-gepa eval --dataset testset \
  --run-dir "$RUN_DIR" \
  --cand-idx 4 --lm openai/gpt-5.5 --reasoning-effort low \
  --sub-lm openai/gpt-5.4-mini

# gpt-5.4 medium
uv run rlm-gepa eval --dataset testset \
  --run-dir "$RUN_DIR" \
  --cand-idx 4 --lm openai/gpt-5.4 --reasoning-effort medium \
  --sub-lm openai/gpt-5.4-mini

# gpt-5.4 low
uv run rlm-gepa eval --dataset testset \
  --run-dir "$RUN_DIR" \
  --cand-idx 4 --lm openai/gpt-5.4 --reasoning-effort low \
  --sub-lm openai/gpt-5.4-mini
```

To run the seed baseline, omit `--run-dir` and `--cand-idx`.

## Re-run optimization

The release optimization run used the parameters recorded in
`optimization_summary.json`:

```bash
uv run rlm-gepa optimize \
  --executor-lm openai/gpt-5.4-mini \
  --executor-sub-lm openai/gpt-5.4-mini \
  --executor-reasoning-effort low \
  --executor-sub-lm-reasoning-effort none \
  --proposer-lm anthropic/claude-sonnet-4-6 \
  --proposer-sub-lm openai/gpt-5.4-mini \
  --proposer-reasoning-effort medium \
  --proposer-sub-lm-reasoning-effort medium \
  --train-dataset trainset \
  --val-ratio 0.2 \
  --cases-per-task 1 \
  --max-metric-calls 6500 \
  --minibatch-size 50 \
  --concurrency 30 \
  --max-iterations 50 \
  --task-timeout 300 \
  --proposer-timeout 1800
```

Before spending model calls, run:

```bash
uv run rlm-gepa optimize --check
```

Resume a stopped run by increasing the total metric-call cap:

```bash
uv run rlm-gepa optimize \
  --resume \
  --run-dir runs/<optimize-run> \
  --max-metric-calls <new-total-cap>
```

Create `runs/<optimize-run>/gepa.stop` to request a graceful stop.

## Key paths

- `spreadsheet_rlm/agent/`: PredictRLM module, signature, and seed spreadsheet
  skill.
- `spreadsheet_rlm/tools/`: LibreOffice recalculation and workbook rendering tools.
- `spreadsheet_rlm/bench/`: dataset loading, scoring, eval CLI, candidate extraction.
- `spreadsheet_rlm/gepa/`: optimizer project config and CLI.
- `run_artifacts/`: tracked release eval/optimization archives.
- `blog_assets/`: release plots.

## Development checks

From the repo root:

```bash
uv run pytest examples/spreadbench/tests -q
uv run ruff check examples/spreadbench/spreadsheet_rlm examples/spreadbench/tests
uv run pytest tests/test_rlm_gepa.py -q
uv run ruff check src/rlm_gepa tests/test_rlm_gepa.py
```
