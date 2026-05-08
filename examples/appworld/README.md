# AppWorld RLM

PredictRLM + RLM-GEPA scaffold for AppWorld tasks.

## Why the runner is isolated

AppWorld currently depends on `pydantic>=1.9,<2`, while `predict-rlm` uses
Pydantic v2. Keep this example's PredictRLM/GEPA process in the normal repo
environment and run AppWorld in a separate Python 3.11/3.12 environment.

## Setup

This example uses two Python environments because AppWorld requires Pydantic v1
while `predict-rlm` uses Pydantic v2:

- `.venv/`: the normal `uv sync` environment for PredictRLM/RLM-GEPA.
- `.appworld-venv/`: an isolated AppWorld runtime used only by the task runner.

Run the setup script:

```bash
make setup
# or: scripts/setup_appworld_data.sh
```

The script runs `uv sync`, creates `.appworld-venv/`, installs AppWorld, unpacks
AppWorld's bundled runtime files, and downloads the dataset under `data/`. The
AppWorld environment and dataset are intentionally gitignored, so a fresh clone
must run setup before live AppWorld evals.

## Dataset semantics

- `train`: official AppWorld train split, minus the groups held out for validation.
- `validation`: deterministic group-level holdout carved out of official train.
- `test_normal` / `test_challenge`: held-out reporting splits only.

### Train / validation split

AppWorld task IDs have a generator prefix and a case suffix. For example, these
are three task cases from the same generator group:

```text
82e2fac_1
82e2fac_2
82e2fac_3
└─────┘
 group
```

Validation randomly selects whole generator groups from the official train split,
not individual cases. With the defaults (`val_ratio=0.20`, `seed=13`), the local
official train pool has 30 groups / 90 cases. Six groups are selected for
validation, producing 18 validation cases and leaving 24 groups / 72 cases for
training.

In code, the split is:

```python
groups = sorted({example.group_id for example in train_examples})
rng = random.Random(seed)
rng.shuffle(groups)
val_size = max(1, int(round(len(groups) * val_ratio)))
val_groups = set(groups[:val_size])

train = [ex for ex in train_examples if ex.group_id not in val_groups]
validation = [ex for ex in train_examples if ex.group_id in val_groups]
```

Holding out whole groups avoids sibling-task leakage. We intentionally do not put
`82e2fac_1` in train and `82e2fac_2` in validation, because those sibling cases
come from the same task generator and can share structure, API patterns, and
failure modes.

## AppWorld RLM interface

We instantiate AppWorld as a stateful external environment for a PredictRLM
agent. The benchmark state, task data, and final scorer remain AppWorld's; the
RLM only supplies the policy that reads task context, calls AppWorld APIs, and
submits the final answer.

This adapter has one compatibility layer. PredictRLM/GEPA runs in this repo's
normal environment, while AppWorld runs in `.appworld-venv/` behind a local
JSONL worker because AppWorld depends on Pydantic v1. This isolates dependencies
and process lifecycle; it is not intended to change the benchmark semantics.

The model-facing action space is intentionally small and host-bound:

```text
list_appworld_apps()
show_appworld_api_descriptions(app_name)
show_appworld_api_doc(app_name, api_name)
search_appworld_api_docs(query)
call_appworld_api(app_name, api_name, kwargs_json)
```

The model discovers apps and API schemas through the documentation tools, then
calls an API with `call_appworld_api`. `kwargs_json` must be a JSON object string
containing the documented API parameters:

```text
call_appworld_api("spotify", "login", "{\"username\": \"...\", \"password\": \"...\"}")
```

The model never supplies `task_id`; each exposed tool closes over the current
task in the host. The adapter does not expose generated `app__api` wrappers such
as `spotify__login` or `venmo__search`.

The prompt and few-shot protocol are matched where possible. Task instructions
come from local `specs.json`, and the supervisor profile context that stock
AppWorld prompts provide is passed into the RLM signature separately. For ICL,
the repo stores only the official demo task IDs (`82e2fac_1`, `29caf6f_1`,
`d0b1f43_1`) and loads tutorial task instructions from the user's local AppWorld
download at runtime. It does not check in worked demo traces, `demos.json`
content, generated app state, ground truth, or reference answers.

The completion protocol is also adapted to the RLM interface. The RLM terminates
with `SUBMIT(answer=value)` for answer tasks or `SUBMIT()` for state-change-only
tasks. Immediately before harness scoring, the wrapper passes that optional raw
answer value to AppWorld's required `supervisor.complete_task({"answer": value})`
call through a host-only completion path; bare `SUBMIT()` sends `{}`.

The model is not given the evaluator, reference answers, score feedback,
`run_appworld_program`, generated direct `app__api` wrappers, or cleanup tools
such as `close_appworld_task`. Scoring happens only after the final AppWorld
state is produced. After each example, the host closes both the AppWorld task and
the JSONL worker process to avoid file-descriptor leaks during concurrent
evaluation.

References:

- Official AppWorld function-calling config: typed direct functions,
  `direct_function_separator="__"`, API predictor, demo task IDs, and
  `demos.json`: [`experiments/configs/simplified_function_calling_agent/openai/gpt-5-2025-08-07-high-reasoning/test_normal.jsonnet#L20-L49`](https://github.com/stonybrooknlp/appworld/blob/a072b7a86e7c1d5b1d7175659d750ebb9b79f10a/experiments/configs/simplified_function_calling_agent/openai/gpt-5-2025-08-07-high-reasoning/test_normal.jsonnet#L20-L49).
- Official AppWorld API docs can be rendered in function-calling format:
  [`src/appworld/collections/api_docs.py#L168-L182`](https://github.com/stonybrooknlp/appworld/blob/a072b7a86e7c1d5b1d7175659d750ebb9b79f10a/src/appworld/collections/api_docs.py#L168-L182).
- HALO's AppWorld demo identifies the same harness surfaces—agent loop, API
  predictor, instructions, demos, and config templates—as the main improvement
  targets: [`demo/appworld/README.md#L193-L203`](https://github.com/context-labs/halo/blob/93371385f4bb12743d4c3c7c6c57f22cabdb2af7/demo/appworld/README.md#L193-L203).

These differences should be disclosed when comparing our numbers to AppWorld or
HALO. PredictRLM uses a documentation-plus-generic-caller interface rather than
native provider-side function schemas or generated direct wrappers.

## Reporting metrics

AppWorld reports two official aggregate completion metrics:

- **Task Goal Completion (TGC)**: case-level pass rate across all task cases.
- **Scenario Goal Completion (SGC)**: group-level pass rate across task-generator
  scenarios. The scenario/group (SGC) passes only when all of its cases pass.

Equivalent formula:

```python
TGC = average(case_success for every task case)
SGC = average(min(case_successes) for each scenario_group)
```

For example, `test_normal` contains 168 task cases grouped into 56 scenarios
(three cases per scenario):

```text
group A: 3/3 case passes -> contributes 3 TGC successes, SGC pass
group B: 2/3 case passes -> contributes 2 TGC successes, SGC fail
group C: 1/3 case passes -> contributes 1 TGC success,  SGC fail
group D: 0/3 case passes -> contributes 0 TGC successes, SGC fail

TGC = case successes / total cases = (3 + 2 + 1 + 0) / 12 = 50%
SGC = fully passed groups / total groups = 1 / 4 = 25%
```

This matches official AppWorld's evaluator implementation, where
`scenario_goal_completion` averages `min(scores)` over each scenario group:
[`src/appworld/evaluator.py#L352-L358`](https://github.com/stonybrooknlp/appworld/blob/a072b7a86e7c1d5b1d7175659d750ebb9b79f10a/src/appworld/evaluator.py#L352-L358).
Similarly, HALO uses the same evaluation logic:
[`demo/appworld/src/appworld/evaluator.py#L352-L358`](https://github.com/context-labs/halo/blob/93371385f4bb12743d4c3c7c6c57f22cabdb2af7/demo/appworld/src/appworld/evaluator.py#L352-L358).

The local `rlm-gepa stats` `soft` / `hard` fields are case-level averages. Use
SGC for comparisons to AppWorld/HALO charts labeled "Scenario Goal Completion".

## Results

### Baselines

| Model | Soft | Hard |
| --- | ---: | ---: |
| GPT-5.4-low | 0.720 | 0.720 (121/168) |
| GPT-5.5-low | 0.899 | 0.899 (151/168) |
| Sonnet-4.6-low | 0.631 | 0.631 (106/168) |

| Model | TGC | SGC | Cost / task |
| --- | ---: | ---: | ---: |
| GPT-5.4-low | 72.0% (121/168) | 50.0% (28/56) | $0.26 ($43.85) |
| GPT-5.5-low | 89.9% (151/168) | 80.4% (45/56) | $0.35 ($58.72) |
| Sonnet-4.6-low | 63.1% (106/168) | 44.6% (25/56) | $0.73 ($123.19) |

## Commands

Optimize and eval use default concurrency 30. Pass `--concurrency` only when a
run needs a different worker count.

```bash
uv run rlm-gepa optimize --check --data-root data --val-ratio 0.2
uv run rlm-gepa optimize --data-root data --max-metric-calls 100 --minibatch-size 5
uv run rlm-gepa eval --dataset test_normal --data-root data --limit 5
uv run rlm-gepa eval --dataset test_challenge --data-root data --run-dir runs/<run> --cand-idx <idx>
```

The default test suite uses tiny fixture split files and does not require
AppWorld, Deno, or API keys.
