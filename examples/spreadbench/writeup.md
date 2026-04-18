## Methodology notes (scratch — flesh out before publishing)

These are bullets to make sure we address the most likely reviewer criticisms
about our train/test split and the way we used SpreadsheetBench. Not writeup
copy yet — just flags to remember.

### On training against SpreadsheetBench at all

- **Split we used**: 912 total → held out the verified 400 as testset, optimized
  GEPA against the other 512 as trainset (`val_ratio=0.20`, `val_limit=50`
  inside the trainset). The testset was never seen during optimization.
- **Dataset vs benchmark distinction matters here.** Some published evaluation
  sets are *datasets* (HotpotQA, HoVer, GSM8K, SQuAD) — they ship with
  official train/dev/test splits and the authors designed them to be trained
  on. Others are *benchmarks* (BIG-Bench Hard, MMLU, SpreadsheetBench,
  IFBench) — they have no official train split and are intended for
  zero/few-shot evaluation. Our situation is the second kind: SpreadsheetBench
  calls itself a benchmark throughout the paper, never publishes a train/test
  split, and §2.1 says explicitly "We define the dataset of a benchmark as
  D = {(q_i, c_i, a_i)}" — singular D, no split.
- **Standard prompt-optimization practice on eval-only benchmarks**: when no
  official train split exists, researchers create one from the benchmark's
  task pool and hold out a disjoint subset for evaluation. Concrete citation:
  the GEPA paper (Agrawal et al., ICLR 2026, §4 Evaluation, p.8) explicitly
  says *"We adopt a standard train/validation/test split. Optimizers have
  full access to the train split, including text and labels, for program
  tuning. [...] We evaluate on six benchmarks—AIME-2025, LiveBench-Math,
  HotpotQA, IFBench, HoVer, and PUPA."*
- **Four of GEPA's six benchmarks are direct analogs to our setup** —
  eval-only benchmarks with no official train split that the GEPA authors
  had to partition themselves:
  - **AIME-2025** (Balunović et al. 2025): 30 contest problems from the
    American Invitational Mathematics Examination 2025. Eval-only. No
    published training data. GEPA invented its own split.
  - **LiveBench-Math** (White et al. 2025): "refresh" math benchmark,
    eval-only by design.
  - **IFBench** (Pyatkin et al. 2025, arXiv 2507.02833): **verified
    eval-only** — the paper says *"The final benchmark consists of 300
    prompts"* and provides no train/val/test split within IFBench itself.
    (The "29 IFTrain constraints" released alongside IFBench are
    supplementary training material, *not* an official IFBench split.)
  - **PUPA** (Li et al. 2025, PAPILLON, arXiv 2410.17127): **verified no
    shipped splits** — 901 total instances (237 PUPA-TNB + 664 PUPA-New).
    The paper says *"We use PUPA-TNB for pipeline and model comparisons,
    and PUPA-New for any optimization we perform"* and ad-hoc samples
    150/150 for its own MIPRO-v2 setup, leaving 364 examples unassigned.
    No canonical split is defined for downstream use.
- **The other two GEPA benchmarks DO have official splits**: HotpotQA
  (Yang et al. 2018) and HoVer (Jiang et al. 2020). When GEPA uses them
  it's using the official train splits — the sanctioned setup. **Do NOT
  cite HotpotQA or HoVer as precedent for our case** — they're the
  opposite of our situation.
- **But a benchmark purist will still push back**: "SpreadsheetBench is
  published as a benchmark for zero-shot evaluation; any subset used for
  optimization is contamination." Acknowledge the tension in the writeup
  instead of ignoring it. Lead with the seed-vs-optimized delta so the
  reader can see what GEPA actually added on top of the base agent.
- **Prompt optimization ≠ fine-tuning**: GEPA evolves a natural-language
  artifact (signature docstring + skill instructions). The base model is
  unchanged. Say this explicitly.
- **Scope the claim honestly**: the evolved prompt is
  SpreadsheetBench-distribution-specific. It will not necessarily transfer to
  other spreadsheet task formats without re-optimization.

### On the specific 512 / 400 split (the more pointed criticism)

- **The 400 is the *verified* subset, not a random half.** The 512 trainset is
  by construction the tasks that the curators *rejected* during the
  verification pass — the messier, more ambiguous, more error-prone ones.
- **Framing: the split *disadvantages* us**, it doesn't advantage us. Our
  training distribution is strictly noisier than our eval distribution. A
  random 512/400 split would probably be easier to GEPA against. Say this
  explicitly in the writeup — it flips the default assumption that "training
  set must have been cherry-picked".
- **Preempt the leakage concern**: run a near-duplicate check between the 512
  trainset and the 400 testset (instruction-text hash match + cosine
  similarity on task text). Report "zero structural overlaps" if that's
  what we find. Take any overlapping tasks out of the training set and re-run
  if there are any.
- **Cross-validation option if we need to strengthen the claim**: optimize on
  a random 512/400 split (ignoring the verified-vs-unverified distinction) and
  compare the optimized score. If similar, the split choice didn't buy us
  anything unfair. Only do this if the criticism lands hard.

### On benchmark pretraining contamination (the bigger problem nobody raises)

- **The paper itself flags this** in Section 2.3 ("Against Data Leakage"):
  SpreadsheetBench was scraped from public Excel forums that every frontier
  model likely saw during pretraining. The authors mitigated it by modifying
  each question (rewording, perturbing spreadsheets, changing answer
  positions). That mitigation is imperfect for modern models.
- **This is a bigger threat to any SpreadsheetBench number than our 512/400
  split**. Cite the paper's own disclosure as context. Reframes the
  conversation from "did you overfit to 512?" to "can any published
  SpreadsheetBench number be trusted given pretraining contamination?"
- Not our problem to solve, but worth one paragraph.

### Defense paragraph (drop into the final writeup's methodology section)

> We optimize against the 512-task non-verified subset of SpreadsheetBench
> and evaluate on the disjoint 400-task verified subset, which was never
> seen during optimization. SpreadsheetBench publishes no official
> train/test split, so our partition is defined de novo. This follows
> standard practice in the prompt-optimization literature: the GEPA paper
> (Agrawal et al., ICLR 2026) evaluates on six benchmarks, four of which
> (AIME-2025, LiveBench-Math, IFBench, PUPA) similarly ship without
> official training splits and require researchers to invent their own
> partition. GEPA's §4 Evaluation explicitly describes the protocol as
> *"a standard train/validation/test split"* in which *"optimizers have
> full access to the train split."* Our setup matches that protocol, with
> the additional caveat that the 400-task verified subset is the
> quality-filtered half of SpreadsheetBench — which disadvantages us
> rather than advantages us, since our training distribution is strictly
> noisier than our eval distribution.

### What to report in the final writeup

- **Seed score on the 400 testset** (ManipulateSpreadsheet seed docstring +
  seed libreoffice_spreadsheet_skill, no optimization). Measures base agent
  quality before any GEPA work.
- **Optimized score on the 400 testset** (best GEPA candidate applied).
  Measures what GEPA added.
- **Delta**. This is the claim. Make it concrete and don't oversell it.
- **Costs at both seed and optimized** (rollout cost + optimization cost).
  Cost-per-pp framing is more honest than raw scores.
- **Timeouts / failures broken down** — how much of the failure set is
  LLM-strength-bound (would improve with a better model) vs benchmark-bug /
  grader-tolerance issues (would never improve). We have this data from the
  failure analysis passes.
- **Prompt format disclosure**: we use the canonical SpreadsheetBench
  single-round prompt format from the paper's Figure 25 (instruction +
  answer_position + answer_sheet), with one deviation — we removed
  `instruction_type` and `spreadsheet_content` from the input fields because
  our RLM agent reads the full file via sandbox mount rather than from a
  prompt-embedded preview. Document this explicitly so the comparison to
  published numbers is apples-to-apples where possible.

---

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
