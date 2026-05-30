# Going recursive (part I): Applying RLM-GEPA to AppWorld 🌎

**TL;DR** Aligned with the
[mismanaged-genius hypothesis](https://alexzhang13.github.io/blog/2026/mgh/),
and inspired by the work of the people at
[Context Labs](https://github.com/context-labs/halo#appworld) on harness
optimization with RLMs, we set out to test what lift we could get from running
AppWorld through a simple generic
[Predict-RLM](https://github.com/Trampoline-AI/predict-rlm) interface, and how
far RLM-GEPA could push performance on the same setup.

On held-out `test_normal`, our strongest current **unoptimized** baseline
reaches **0.917 TGC / 0.839 SGC** with PredictRLM(GPT-5.5 low), above the
current public AppWorld `test_normal` leaderboard high-water mark of **0.804
SGC**;

on `test_challenge`, the same unoptimized PredictRLM(GPT-5.5 low) run reaches
**0.914 TGC / 0.820 SGC**.

AppWorld optimized RLM-GEPA lifts the strongest run to **0.940 TGC / 0.911 SGC**
on `test_normal`, a **+2.3 pp TGC / +7.2 pp SGC** gain,

and on `test_challenge` reaches **0.911 TGC / 0.849 SGC**, corresponding to a
**-0.3 pp TGC / +2.9 pp SGC** change relative to the unoptimized baseline.

## 1. Motivation

AppWorld is a benchmark for agents operating realistic app ecosystems: email,
calendar, Spotify, Venmo, shopping, todo lists, etc. Tasks require changing
state through a sequence of API calls, then submitting the final answer or
completing the task.

This makes AppWorld a natural place to test whether a generic RLM interface can
operate against the existing benchmark environment and evaluator. Aligned with
the
[mismanaged-genius hypothesis](https://alexzhang13.github.io/blog/2026/mgh/), we
set out to test what lift we could get from running AppWorld through a simple
RLM interface, and how far we could push an optimized RLM on that same task.

AppWorld agents often have exactly the shape the mismanaged-genius hypothesis
calls out: planners, routers, API selection, direct function wrappers, recovery
logic, and curated prompts. The question here is whether an RLM gives the model
a better management interface: expose the environment, preserve the evaluator,
let the LM express control flow in code with tools as functions, then optimize
the resulting RLM skill with RLM-GEPA.

Our view of RLMs is that they are a natural runtime for user-defined programs
(skills) interpreted into model-defined control flow.

Coupled with RLM-GEPA they represent a sort of
[bitter free lunch(tm)](https://x.com/lateinteraction/status/2043099113000931398?s=20)
where the task specification is concentrated in the skill as a standard
operating procedure.

Instead of encoding the agent loop as a pile of example-specific glue, expose a
small set of tools, define a skill as a standard operating procedure, and let
the model decide how to proceed.

Same task. Less harness.

## 2. AppWorld as an RLM environment

Our AppWorld adapter keeps AppWorld's task state and evaluator intact. The RLM
only supplies the policy.

The model-facing tool surface is intentionally small:

```text
list_appworld_apps()
show_appworld_api_descriptions(app_name)
show_appworld_api_doc(app_name, api_name)
search_appworld_api_docs(query)
call_appworld_api(app_name, api_name, kwargs)
```

The model discovers available apps and APIs through documentation tools, then
calls a single generic API caller:

```python
await call_appworld_api(
  "spotify",
  "login",
  {"username": "...", "password": "..."}
)
```

Completion is also adapted to the RLM interface. The model ends with
`SUBMIT(answer=value)` for answer tasks or `SUBMIT()` for state-change-only
tasks. Immediately before scoring, the host maps that into AppWorld's required
`supervisor.complete_task(...)` call.

## 3. Metrics

AppWorld reports two aggregate metrics:

- **TGC**: Task Goal Completion, the case-level pass rate.
- **SGC**: Scenario Goal Completion, the group-level pass rate.

`test_normal` has 168 task cases grouped into 56 scenarios, with three cases per
scenario. SGC is stricter: a scenario only passes if all of its cases pass.

```text
group A: 3/3 case passes -> contributes 3 TGC successes, SGC pass
group B: 2/3 case passes -> contributes 2 TGC successes, SGC fail
group C: 1/3 case passes -> contributes 1 TGC success,  SGC fail
group D: 0/3 case passes -> contributes 0 TGC successes, SGC fail
```

This matters because the public AppWorld leaderboard reports both **TGC** and
**SGC**, and SGC is the stricter headline metric for scenario-level success.

## 4. Our unoptimized baselines

Before optimizing anything with RLM-GEPA, we measured the RLM baseline across
several executor families on full `test_normal`. The relevant headline is that
PredictRLM(GPT-5.5 low) already reaches **154 / 168** task cases, or **0.917 TGC
/ 0.839 SGC**, before any AppWorld-specific skill tuning. PredictRLM(Sonnet 4.6
adaptive) reaches **0.786 SGC** under the same small
documentation-plus-generic-caller interface.

For the comparison that matters most, see the public leaderboard table in
Section 6. For the full internal baseline sweep, including cost, errors, and
timeouts, see the appendix.

## 5. RLM-GEPA / optimization methodology

We use [RLM-GEPA](https://github.com/Trampoline-AI/predict-rlm/tree/main/src/rlm_gepa),
our port of GEPA over RLMs, to optimize the AppWorld
[predict-RLM](https://github.com/Trampoline-AI/predict-rlm) skill, not model
weights. The optimizer runs on AppWorld's `train` split and selects candidates
on the held-out `dev` split; `test_normal` and `test_challenge` are reserved for
reporting. The proposer reads execution traces and evaluator feedback after each
attempt, then rewrites the AppWorld skill instructions for future runs.

The main optimized skill in the table was produced with a cheap
`gpt-5.4-mini` proxy executor and sub-LM at low / no explicit reasoning
effort, while the proposer was `gpt-5.5` high with a `gpt-5.5` high
sub-LM. That run used minibatches of **30** examples, up to **3,000**
metric calls, concurrency **10**, task timeout **300s**, proposer timeout
**900s**, merge proposer enabled, and selected candidate **12** with dev score
**0.970** after **3,182** metric calls.

We also ran a stronger-proxy optimization pass with `gpt-5.4` low as both
executor and sub-LM, again using `gpt-5.5` high for the proposer and
sub-LM. That run used the same minibatch size **30**, concurrency **10**,
**300s** task timeout, **900s** proposer timeout, merge proposer, and a lower
budget of **2,000** metric calls.

In both cases, the optimized artifact is only the skill; the held-out rows below
evaluate that skill unchanged with the named executor model.

## 6. Public leaderboard comparison

The natural comparison is the official AppWorld leaderboard, with special
attention to optimized-harness work.

Outside of the leaderboard, Context Labs' HALO is the closest conceptual
reference point for this post: it explicitly studies harness optimization with
RLMs on AppWorld, and its
[public AppWorld chart](https://github.com/context-labs/HALO#appworld) reports
peak optimized `test_normal` SGC of **0.482** for HALO(Gemini 3 Flash) and
**0.732** for HALO(Sonnet 4.6); the chart does not report TGC.

The current public `test_normal` leaderboard high-water mark is Alibaba Cloud
ApsaraLab AgentRL with Qwen3-14B at **0.869 TGC / 0.804 SGC**. It reports an
impressive (for a 14B module) **0.869 TGC / 0.804 SGC** on `test_normal`, but
drops to **0.676 TGC / 0.504 SGC** on `test_challenge`, which suggests possible
overfitting to the easier split.

Our best unoptimized [predict-RLM](https://github.com/Trampoline-AI/predict-rlm)
baseline is **0.917 TGC / 0.839 SGC** with PredictRLM(GPT-5.5 low), and the
strongest RLM-GEPA run reaches **0.940 TGC / 0.911 SGC**.

<!-- prettier-ignore -->
<!-- deno-fmt-ignore -->
| Method / model                    | `test_normal` (TGC / SGC)         | `test_challenge` (TGC / SGC)      |
| --------------------------------- | --------------------------------- | --------------------------------- |
| _LOOP_                            |                                   |                                   |
| - Qwen2.5-32B                     | 72.6% / 53.6%                     | -                                 |
|                                   |                                   |                                   |
| _ReAct + 2 SetBSR_                |                                   |                                   |
| - GPT-4o                          | 68.5% / 57.1%                     | -                                 |
|                                   |                                   |                                   |
| _IBM CUGA_                        |                                   |                                   |
| - GPT-4.1                         | 73.2% / 62.5%                     | -                                 |
|                                   |                                   |                                   |
| _HALO_, **optimized harness**     |                                   |                                   |
| - HALO(Gemini 3 Flash)            | not reported / 48.2%              | -                                 |
| - HALO(Sonnet 4.6)                | not reported / 73.2%              | -                                 |
|                                   |                                   |                                   |
| _Alibaba Cloud ApsaraLab AgentRL_ |                                   |                                   |
| - Qwen3-14B                       | 86.9% / 80.4%                     | 67.6% / 50.4%                     |
|                                   |                                   |                                   |
| _predict-RLM,_ **unoptimized**    |                                   |                                   |
| - PredictRLM(Gemini 3 Flash)      | 69.6% / 42.9%                     | 66.4% / 39.6%                     |
| - PredictRLM(GPT-5.4 low)         | 82.7% / 71.4%                     | -                                 |
| - PredictRLM(GPT-5.4 medium)      | 83.9% / 73.2%                     | -                                 |
| - PredictRLM(Sonnet 4.6)          | 88.1% / 78.6%                     | -                                 |
| - PredictRLM(GPT-5.5 low)         | 91.7% / 83.9%                     | 91.4% / 82.0%                     |
|                                   |                                   |                                   |
| _predict-RLM + RLM-GEPA_          |                                   |                                   |
| - PredictRLMGEPA(Gemini 3 Flash)† | 78.6% ↑9.0 pp / 55.4% ↑12.5 pp    | 64.0% ↓2.4 pp / 38.8% ↓0.7 pp     |
| - PredictRLMGEPA(GPT-5.4 low)     | 86.3% ↑3.6 pp / 75.0% ↑3.6 pp     | -                                 |
| - PredictRLMGEPA(GPT-5.4 medium)  | 86.3% ↑2.4 pp / 76.8% ↑3.6 pp     | -                                 |
| - PredictRLMGEPA(Sonnet 4.6)†     | 83.9% ↓4.2 pp / 71.4% ↓7.1 pp     | -                                 |
| - PredictRLMGEPA(GPT-5.5 low)†    | **94.0% ↑2.3 pp / 91.1% ↑7.2 pp** | **91.1% ↓0.3 pp / 84.9% ↑2.9 pp** |

† Optimized with a gpt54-mini proxy executor, then evaluated with the named
model.

Lift annotations are absolute percentage-point changes from the matching
unoptimized [predict-RLM](https://github.com/Trampoline-AI/predict-rlm) row.

The unoptimized rows are initial
[predict-RLM](https://github.com/Trampoline-AI/predict-rlm) baselines without
RLM-GEPA optimization. We were surprised that this baseline already beat HALO's
published optimized-harness AppWorld result for Sonnet 4.6 and the current
public `test_normal` high-water mark.

We were also encouraged by how well the RLM-GEPA gains transferred off the
optimization split. The strongest current row in this table reaches
**94.0% / 91.1%** on `test_normal` and **91.1% / 84.9%** on `test_challenge`
with PredictRLMGEPA(GPT-5.5 low).

There is one caveat: this is not yet an official leaderboard submission. The
main comparison remains `test_normal`, where the public leaderboard is most
complete. The `test_challenge` columns are included when available because they
show a different kind of robustness: the public AgentRL row drops from **80.4%**
SGC on `test_normal` to **50.4%** on `test_challenge`, while the
[predict-RLM](https://github.com/Trampoline-AI/predict-rlm) baseline and
RLM-GEPA runs stay above **82%** SGC.

## 7. Why this matters

A lot of agent work hides intelligence in the harness.

The loop decides what to inspect. The planner decides what counts as a step. The
router decides which API should be called. The wrapper shapes every tool call
before the model sees it. This can work well, but it makes progress hard to
interpret: did the model improve, or did the harness improve?

RLMs push in the opposite direction.

They make the execution trace explicit and let the model write the control flow
inside a constrained runtime. The host still owns safety, state, tools, and
scoring. But the policy is no longer a hand-authored loop pretending to be
general intelligence.

## 8. What's next?

AppWorld should be a favorable setting for RLMs: at its core, it is a
function-calling benchmark. If the task is mostly choosing APIs, inspecting
state, composing calls, and recovering from mistakes, an RLM should be a natural
fit.

In Part 2, we explore how well RLMs compete with harnesses on tasks that are
more harness-friendly & less natural for RLMs: Terminal-Bench 2.1. The question
there is whether the same pattern holds when the environment is a terminal and
the incumbent baselines are more explicitly harness-engineered.

If you want to try this yourself, check out
[predict-RLM](https://github.com/Trampoline-AI/predict-rlm) on GitHub and give
us a ⭐️

Follow me for Part 2.

with ♥ from MTL

## Appendix: Full unoptimized baseline sweep

These are our unoptimized
[predict-RLM](https://github.com/Trampoline-AI/predict-rlm) AppWorld runs on
full `test_normal`, before any AppWorld-specific RLM-GEPA skill tuning.

| Run                                  | c  | Pass    |   TGC |   SGC | Errors | Timeouts | Cost   |
| ------------------------------------ | -- | ------- | ----: | ----: | -----: | -------: | ------ |
| PredictRLM(GPT-5.5 low)              | 10 | 154/168 | 0.917 | 0.839 |      0 |        0 | $69.44 |
| PredictRLM(Sonnet 4.6 adaptive)      | 10 | 148/168 | 0.881 | 0.786 |      0 |        0 | $69.61 |
| PredictRLM(GPT-5.4 medium)           | 20 | 141/168 | 0.839 | 0.732 |      1 |        1 | $61.52 |
| PredictRLM(GPT-5.4 low)              | 30 | 139/168 | 0.827 | 0.714 |      3 |        2 | $53.84 |
| PredictRLM(Kimi K2.6 Exacto)         | 10 | 137/168 | 0.815 | 0.679 |      6 |        6 | $18.27 |
| PredictRLM(DeepSeek V4 Flash Exacto) | 10 | 121/168 | 0.720 | 0.482 |      3 |        3 | $2.26  |
| PredictRLM(Gemini 3 Flash)           | 10 | 117/168 | 0.696 | 0.429 |      8 |        8 | $18.29 |
| PredictRLM(GPT-5.4-mini low)         | 20 | 104/168 | 0.619 | 0.375 |      1 |        1 | $10.89 |
