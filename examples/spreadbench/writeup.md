
## Baselines

### gpt-5.4 (effort = low)
```
uv run --env-file .env.development python examples/spreadbench/scripts/eval.py \
  --lm openai/gpt-5.4 \
  --reasoning_effort low \
  -sub_lm openai/gpt-5.1 \
  --dataset testset \
  --concurrency 30 \
  --max_iterations 50 \
  --task_timeout 300
```

```
- eval_20260413_141857/ + eval_20260413_141857.json  
  - Completed 400/400  
  - gpt-5.4, low, prompt_length=405, hard 0.7025  
  - examples/spreadbench/runs/eval_20260413_141857.json:14
```


```
============================================================
EVALUATION COMPLETE
============================================================
Prompt:         seed
Model:          openai/gpt-5.4 (reasoning_effort=low)
Dataset:        testset
Tasks:          400
Duration:       38m 50s
Soft (avg):     0.7607
Hard (avg):     0.7025  (281/400 all passing)

Costs:
  main     openai/gpt-5.4                    1490 calls  11,219,566 in /  1,114,790 out  $  26.6412
  sub      openai/gpt-5.1                       0 calls           0 in /          0 out  $   0.0000
  total                                                                                  $  26.6412
```

### mercury-2 (effort = medium)

```
uv run --env-file .env.development python examples/spreadbench/scripts/eval.py \
  --lm openai/mercury-2 \
  --reasoning_effort medium \
  --sub_lm openai/gpt-5.1 \
  --dataset testset \
  --concurrency 30 \
  --max_iterations 50 \
  --task_timeout 300
```

```
- eval_20260413_151349/ + eval_20260413_151349.json  gemini/gemini-3.1-pro-preview
  - Completed 400/400  
  - mercury-2, medium, prompt_length=405, hard 0.27  
  - examples/spreadbench/runs/eval_20260413_151349.json:3
```

```
============================================================
EVALUATION COMPLETE
============================================================
Prompt:         seed
Model:          openai/mercury-2 (reasoning_effort=medium)
Dataset:        testset
Tasks:          400
Duration:       51m 41s
Soft (avg):     0.4330
Hard (avg):     0.2700  (108/400 all passing)

Costs:
  main     openai/mercury-2                  1790 calls  11,732,377 in /    936,771 out  $   3.6357
  sub      openai/gpt-5.1                       0 calls           0 in /          0 out  $   0.0000
  total                                                                                  $   3.6357
```

## GEPA Prompt Compilation (Optimization)

```
uv run --env-file .env.development python examples/spreadbench/scripts/optimize.py \
  --lm openai/mercury-2 \
  --reasoning_effort low \
  --sub_lm openai/gpt-5.1 \
  --reflection_lm anthropic/claude-opus-4-6 \
  --train_dataset trainset \
  --val_ratio 0.20 \
  --val_limit 50 \
  --cases_per_task 1 \
  --max_metric_calls 2000 \
  --minibatch_size 20 \
  --seed 42 \
  --module_selector round_robin \
  --rlm_proposer \
  --proposer_max_iterations 40 \
  --concurrency 30 \
  --max_iterations 50 \
  --task_timeout 300
```

```
  Final result
 
  ┌─────────────────┬──────────────────────┐
  │     metric      │        value         │
  ├─────────────────┼──────────────────────┤
  │ duration        │ 7h04m                │
  ├─────────────────┼──────────────────────┤
  │ candidates      │ 22                   │
  ├─────────────────┼──────────────────────┤
  │ metric calls    │ 2035 / 2000          │
  ├─────────────────┼──────────────────────┤
  │ best val (n=50) │ 0.6810               │
  ├─────────────────┼──────────────────────┤
  │ best idx        │ cand[12]             │
  ├─────────────────┼──────────────────────┤
  │ Δ vs seed       │ +0.1148 (+20.3% rel) │
  ├─────────────────┼──────────────────────┤
  │ seed val        │ 0.5662               │
  └─────────────────┴──────────────────────┘

  Costs (final)

  ┌──────────┬───────────┬────────┬───────┬───────┬─────────┐
  │   role   │   model   │ calls  │  in   │  out  │    $    │
  ├──────────┼───────────┼────────┼───────┼───────┼─────────┤
  │ main     │ mercury-2 │ 10,000 │ 86.6M │ 4.32M │ $24.89  │
  ├──────────┼───────────┼────────┼───────┼───────┼─────────┤
  │ sub      │ gpt-5.1   │ 0      │ 0     │ 0     │ $0.00   │
  ├──────────┼───────────┼────────┼───────┼───────┼─────────┤
  │ proposer │ opus-4.6  │ 634    │ 14.5M │ 295K  │ $87.42  │
  ├──────────┼───────────┼────────┼───────┼───────┼─────────┤
  │ rollout  │           │        │       │       │ $24.89  │
  ├──────────┼───────────┼────────┼───────┼───────┼─────────┤
  │ optimize │           │        │       │       │ $87.42  │
  ├──────────┼───────────┼────────┼───────┼───────┼─────────┤
  │ total    │           │        │       │       │ $112.31 │
  └──────────┴───────────┴────────┴───────┴───────┴─────────┘
```

## Evaluating the Compiled Program

### mercury-2 (effort = low)
```
uv run --env-file .env.development python examples/spreadbench/scripts/eval.py \
  --lm openai/mercury-2 \
  --reasoning_effort low \
  --run_dir examples/spreadbench/runs/optimize_20260413_234031
```

```
============================================================
EVALUATION COMPLETE
============================================================
Signature:      optimize_20260413_234031#sig (6699 chars)
Skill:          optimize_20260413_234031#skill (7594 chars)
Model:          openai/mercury-2  (reasoning_effort=low)
Dataset:        testset
Tasks:          400
Duration:       29m 25s
Soft (avg):     0.6222
Hard (avg):     0.4075  (163/400 all passing)

Costs:
  main     openai/mercury-2                  2209 calls  21,180,689 in /  1,114,912 out  $   6.1314
  sub      openai/gpt-5.1                       0 calls           0 in /          0 out  $   0.0000
  total                                                                                  $   6.1314

Saved to:       /Users/gabriel/Workspace/trampoline-ai/predict-rlm.spreadbench-release/examples/spreadbench/runs/eval_20260414_072012.json
Per-case logs:  /Users/gabriel/Workspace/trampoline-ai/predict-rlm.spreadbench-release/examples/spreadbench/runs/eval_20260414_072012
```

### mercury-2 (effort = medium)
```
============================================================
EVALUATION COMPLETE
============================================================
Signature:      optimize_20260413_234031#sig (6699 chars)
Skill:          optimize_20260413_234031#skill (7594 chars)
Model:          openai/mercury-2  (reasoning_effort=medium)
Dataset:        testset
Tasks:          400
Duration:       31m 39s
Soft (avg):     0.6532
Hard (avg):     0.4600  (184/400 all passing)

Costs:
  main     openai/mercury-2                  1937 calls  18,605,807 in /  1,296,783 out  $   5.6240
  sub      openai/gpt-5.1                       0 calls           0 in /          0 out  $   0.0000
  total                                                                                  $   5.6240

Saved to:       /Users/gabriel/Workspace/trampoline-ai/predict-rlm.spreadbench-release/examples/spreadbench/runs/eval_20260414_075046.json
Per-case logs:  /Users/gabriel/Workspace/trampoline-ai/predict-rlm.spreadbench-release/examples/spreadbench/runs/eval_20260414_075046
```

### gpt-5.4 (effort = none)

```
uv run --env-file .env.development python examples/spreadbench/scripts/eval.py \
  --lm openai/gpt-5.4 \
  --reasoning_effort none \
  --run_dir examples/spreadbench/runs/optimize_20260413_234031
```

```
============================================================
EVALUATION COMPLETE
============================================================
Signature:      optimize_20260413_234031#sig (6699 chars)
Skill:          optimize_20260413_234031#skill (7594 chars)
Model:          openai/gpt-5.4  (reasoning_effort=none)
Dataset:        testset
Tasks:          400
Duration:       31m 26s
Soft (avg):     0.9071
Hard (avg):     0.8400  (336/400 all passing)

Costs:
  main     openai/gpt-5.4                    1449 calls  15,416,175 in /    916,703 out  $  27.9140
  sub      openai/gpt-5.1                       0 calls           0 in /          0 out  $   0.0000
  total                                                                                  $  27.9140

Saved to:       /Users/gabriel/Workspace/trampoline-ai/predict-rlm.spreadbench-release/examples/spreadbench/runs/eval_20260414_104504.json
Per-case logs:  /Users/gabriel/Workspace/trampoline-ai/predict-rlm.spreadbench-release/examples/spreadbench/runs/eval_20260414_104504
```

### gpt-5.4 (effort = low)

```
uv run --env-file .env.development python examples/spreadbench/scripts/eval.py \
  --lm openai/gpt-5.4 \
  --reasoning_effort low \
  --run_dir examples/spreadbench/runs/optimize_20260413_234031
```

```
============================================================
EVALUATION COMPLETE
============================================================
Signature:      optimize_20260413_234031#sig (6699 chars)
Skill:          optimize_20260413_234031#skill (7594 chars)
Model:          openai/gpt-5.4  (reasoning_effort=low)
Dataset:        testset
Tasks:          400
Duration:       31m 4s
Soft (avg):     0.9087
Hard (avg):     0.8400  (336/400 all passing)

Costs:
  main     openai/gpt-5.4                    1472 calls  16,240,370 in /  1,415,269 out  $  36.2515
  sub      openai/gpt-5.1                       0 calls           0 in /          0 out  $   0.0000
  total                                                                                  $  36.2515

Saved to:       /Users/gabriel/Workspace/trampoline-ai/predict-rlm.spreadbench-release/examples/spreadbench/runs/eval_20260414_082428.json
Per-case logs:  /Users/gabriel/Workspace/trampoline-ai/predict-rlm.spreadbench-release/examples/spreadbench/runs/eval_20260414_082428
```

### gpt-5.4 (effort = medium)

```
uv run --env-file .env.development python examples/spreadbench/scripts/eval.py \
  --lm openai/gpt-5.4 \
  --reasoning_effort medium \
  --run_dir examples/spreadbench/runs/optimize_20260413_234031
```

```
============================================================
EVALUATION COMPLETE
============================================================
Signature:      optimize_20260413_234031#sig (6699 chars)
Skill:          optimize_20260413_234031#skill (7594 chars)
Model:          openai/gpt-5.4  (reasoning_effort=medium)
Dataset:        testset
Tasks:          400
Duration:       35m 7s
Soft (avg):     0.8977
Hard (avg):     0.8425  (337/400 all passing)

Costs:
  main     openai/gpt-5.4                    1557 calls  17,731,565 in /  2,343,635 out  $  50.6966
  sub      openai/gpt-5.1                       0 calls           0 in /          0 out  $   0.0000
  total                                                                                  $  50.6966

Saved to:       /Users/gabriel/Workspace/trampoline-ai/predict-rlm.spreadbench-release/examples/spreadbench/runs/eval_20260414_100724.json
Per-case logs:  /Users/gabriel/Workspace/trampoline-ai/predict-rlm.spreadbench-release/examples/spreadbench/runs/eval_20260414_100724
```
