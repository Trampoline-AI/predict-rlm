# SpreadsheetBench RLM-GEPA

This example packages the SpreadsheetBench RLM used for the RLM-GEPA release. It
contains the spreadsheet agent, host-side workbook tools, dataset preparation,
held-out evaluation, and GEPA optimization wiring.

The release write-up is in [`blog_post.md`](blog_post.md). The plot images used
by that article are in [`blog_assets/`](blog_assets/).

## Layout

- `spreadsheet_rlm/agent/`: `PredictRLM` module, DSPy signature, and seed
  spreadsheet skill.
- `spreadsheet_rlm/tools/`: host-side LibreOffice recalculation and workbook
  rendering tools.
- `spreadsheet_rlm/bench/`: SpreadsheetBench dataset loading, scoring, eval CLI,
  and candidate extraction.
- `spreadsheet_rlm/gepa/`: project-specific `RLMGepaProject`, defaults, and CLI
  wiring.
- `data/`: checked-in source archives plus generated `trainset/` and `testset/`
  folders.
- `runs/`: local eval and optimize artifacts. The directory contents are
  gitignored.
- `blog_post.md`: article describing the RLM-GEPA result and methodology.
- `blog_assets/`: checked-in article plots.

## Results

The optimized skill was evolved on the non-Verified SpreadsheetBench remainder
using a cheap executor, then evaluated unchanged on the held-out 400-task
SpreadsheetBench Verified set.

| Executor          | Seed soft | Seed hard | Optimized soft | Optimized hard |
| ----------------- | --------: | --------: | -------------: | -------------: |
| `gpt-5.4`, low    |    0.8710 |    0.7800 |         0.8980 |         0.8150 |
| `gpt-5.4`, medium |    0.8750 |    0.7950 |         0.9259 |         0.8500 |
| `gpt-5.5`, low    |    0.9092 |    0.8500 |         0.9288 |         0.8775 |
| `gpt-5.5`, medium |    0.9142 |    0.8600 |         0.9411 |         0.8925 |

See [`blog_post.md`](blog_post.md) for the full interpretation, costs, and
candidate lineage plots.

## Setup

Run commands from this directory unless noted otherwise. That keeps optimize
artifacts under `examples/spreadbench/runs/`.

```bash
cd examples/spreadbench
uv sync
```

Runtime requirements:

- Python 3.11+
- Deno v2 for the Pyodide/WASM sandbox
- LibreOffice available on `PATH` for workbook recalculation/rendering
- Provider API keys expected by LiteLLM, such as `OPENAI_API_KEY` or
  `ANTHROPIC_API_KEY`

If running from the repository root, prefix commands with
`uv run --project examples/spreadbench` and pass an explicit `--run-dir` if you
want optimize artifacts to stay under `examples/spreadbench/runs/`.

## Dataset

The checked-in archives are the source of truth:

- `data/spreadsheetbench_912_v0.1.tar.gz`
- `data/spreadsheetbench_verified_400.tar.gz`

Prepare extracted folders and splits with either command:

```bash
make dataset
```

```bash
uv run python -m spreadsheet_rlm.bench.prepare_dataset
```

This creates:

- `data/trainset/`: 512 tasks from the 912-task set after removing Verified IDs.
- `data/testset`: symlink to the 400-task Verified held-out set.

The optimizer uses `trainset` by default. The release metrics use `testset` only
for held-out evaluation.

## Evaluate

Run a seed held-out eval:

```bash
uv run rlm-gepa eval \
  --dataset testset \
  --lm openai/gpt-5.5 \
  --reasoning-effort medium \
  --sub-lm openai/gpt-5.4-mini
```

Run a small smoke eval:

```bash
uv run rlm-gepa eval \
  --dataset testset \
  --limit 5 \
  --concurrency 5 \
  --lm openai/gpt-5.4 \
  --reasoning-effort low \
  --sub-lm openai/gpt-5.4-mini
```

Evaluate the best validation candidate from an optimize run:

```bash
uv run rlm-gepa eval \
  --dataset testset \
  --run-dir runs/<optimize-run> \
  --lm openai/gpt-5.5 \
  --reasoning-effort medium \
  --sub-lm openai/gpt-5.4-mini
```

Evaluate a specific candidate:

```bash
uv run rlm-gepa eval \
  --dataset testset \
  --run-dir runs/<optimize-run> \
  --cand-idx 4 \
  --lm openai/gpt-5.5 \
  --reasoning-effort medium \
  --sub-lm openai/gpt-5.4-mini
```

Useful eval flags:

- `--only skill`: use the evolved skill with the seed signature.
- `--only signature`: use the evolved signature with the seed skill.
- `--task-ids 123,456`: evaluate selected task IDs.
- `--task-ids @tasks.txt`: read one task ID per line.
- `--cases-per-task N`: cap cases per task, with `0` meaning all cases.
- `--output runs/<eval-run>`: write `eval.json` and logs to a chosen directory.
- `--no-logs`: skip per-case RLM logs.
- `--cache`: enable DSPy LM caching. Caching is disabled by default.

Eval output includes `eval.json`, aggregate cost logs, task traces, and optional
per-case logs under `cases/`.

## Optimize

Check model credentials and dataset wiring before spending calls:

```bash
uv run rlm-gepa optimize --check
```

Run optimization with the default SpreadsheetBench config:

```bash
uv run rlm-gepa optimize \
  --max-metric-calls 2000 \
  --minibatch-size 50 \
  --concurrency 30
```

The RLM proposer is the default proposer. Merge proposals are opt-in:

```bash
uv run rlm-gepa optimize \
  --merge-proposer \
  --max-merge-attempts 12 \
  --max-metric-calls 3000
```

Override executor and proposer models explicitly:

```bash
uv run rlm-gepa optimize \
  --executor-lm openai/gpt-5.4-mini \
  --executor-sub-lm openai/gpt-5.4-mini \
  --executor-reasoning-effort low \
  --executor-sub-lm-reasoning-effort none \
  --proposer-lm anthropic/claude-sonnet-4-6 \
  --proposer-sub-lm openai/gpt-5.4-mini \
  --proposer-reasoning-effort medium \
  --proposer-sub-lm-reasoning-effort medium
```

Resume a stopped run by increasing the total metric-call cap:

```bash
uv run rlm-gepa optimize \
  --resume \
  --run-dir runs/<optimize-run> \
  --max-metric-calls 5014
```

To request a graceful stop, create `gepa.stop` inside the run directory. The
optimizer checks that file through GEPA's file stopper.

```bash
touch runs/<optimize-run>/gepa.stop
```

Verbose RLM logs can be streamed during eval/optimization with `--verbose-rlm`.

## Codex-LM Routing

When `dspy-codex-lm` is installed, `--codex-lm` routes OpenAI-family `dspy.LM`
construction through the Codex-LM adapter. Use `--no-codex-lm` to disable this.

With a local editable checkout:

```bash
uv run --with-editable /path/to/dspy-codex-lm rlm-gepa optimize \
  --codex-lm \
  --verbose-rlm \
  --merge-proposer \
  --proposer-lm openai/gpt-5.5 \
  --proposer-reasoning-effort medium \
  --proposer-sub-lm openai/gpt-5.5 \
  --proposer-sub-lm-reasoning-effort medium
```

From the repository root, use the project form:

```bash
uv run --project examples/spreadbench \
  --with-editable /path/to/dspy-codex-lm \
  python -m spreadsheet_rlm.gepa optimize --codex-lm
```

Use repeated `--codex-lm-exclude <substring>` flags to leave specific model
names unpatched.

## Inspect Runs

Render all stats tables:

```bash
uv run rlm-gepa stats runs/<optimize-run>
```

Render one table:

```bash
uv run rlm-gepa stats runs/<optimize-run> --table iterations
uv run rlm-gepa stats runs/<optimize-run> --table merges
uv run rlm-gepa stats runs/<optimize-run> --table candidates
uv run rlm-gepa stats runs/<optimize-run> --table costs
```

Use Markdown output for copy/paste:

```bash
uv run rlm-gepa stats runs/<optimize-run> --table candidates --format markdown
```

Write PNG plots:

```bash
uv run rlm-gepa plot runs/<optimize-run>
```

Plots are written under `runs/<optimize-run>/plots/` unless `--output` is set.

Important optimize artifacts:

- `run_metadata.json`: resolved config and invocation metadata.
- `gepa_state.bin`: GEPA-owned checkpoint and candidate state.
- `rlm_merge_state.json`: RLM merge proposer bookkeeping when merges are
  enabled.
- `optimization_summary.json`: final summary for completed runs.
- `all_candidates.json`: final candidate texts and validation scores for
  completed runs.
- `cost_log.jsonl`: append-only cost events used by stats.
- `task_traces/`: per-task executor traces.
- `proposer_traces/`: normal and merge proposer traces.

During in-progress runs, the shared stats and plot readers operate on artifact
files directly. Eval needs the SpreadsheetBench agent implementation to rerun
candidates.

## Candidate Workflow

Use `stats --table candidates` to find candidate indices and validation scores.
If `--cand-idx` is omitted, eval extracts the candidate with the highest mean
validation score from `gepa_state.bin`.

Typical held-out rerun after optimization:

```bash
uv run rlm-gepa stats runs/<optimize-run> --table candidates
uv run rlm-gepa eval \
  --dataset testset \
  --run-dir runs/<optimize-run> \
  --lm openai/gpt-5.5 \
  --reasoning-effort medium \
  --sub-lm openai/gpt-5.4-mini
```

Pass `--cand-idx <N>` to force a specific candidate instead of the validation
best.

## Archive Runs

Run artifacts can be large and are intentionally gitignored. The release ships
tracked archives in [`run_artifacts/`](run_artifacts/) so the supporting
eval and optimize runs can be restored without committing expanded traces.

Restore all bundled run artifacts from this directory:

```bash
mkdir -p runs
for archive in run_artifacts/*.tar.gz; do
  tar -xzf "$archive" -C runs
done
```

Archive a new completed run for tracking from this directory:

```bash
COPYFILE_DISABLE=1 tar -czf run_artifacts/<archive-name>.tar.gz -C runs <optimize-run>
```

Restore it later:

```bash
tar -xzf run_artifacts/<archive-name>.tar.gz -C runs
```

Keep the whole run directory together if you need resume, stats, plots, or
candidate extraction. The checkpoint, merge state, trace logs, and cost log are
cross-referenced by path and candidate index.

## Development Checks

Run dev checks from the repository root so pytest and ruff use the root dev
environment:

```bash
cd ../..
uv run pytest examples/spreadbench/tests -q
uv run ruff check examples/spreadbench/spreadsheet_rlm examples/spreadbench/tests
uv run pytest tests/test_rlm_gepa.py -q
uv run ruff check src/rlm_gepa tests/test_rlm_gepa.py
```
