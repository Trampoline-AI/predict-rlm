---
name: rlm-gepa
description: >
  Design, scaffold, and use RLM-GEPA optimization wiring for PredictRLM projects,
  including AgentSpec scoping, train/validation data for optimization or
  candidate selection, scoring feedback, seed candidates, GEPA project files,
  and GEPA optimize/eval CLI setup. Use when the user asks for GEPA, prompt or
  skill optimization, candidate selection from RLM traces, AgentSpec,
  RLMGepaProject, optimization metrics, or split design for optimization or
  candidate selection. Use rlm for ordinary agent-plus-evals or evaluation-only
  work without GEPA optimization or candidate selection. Do not use for modifying
  the predict-rlm repository internals; use predict-rlm-contributor for that.
---

# RLM-GEPA Optimization

RLM-GEPA optimizes reusable PredictRLM text components, usually skill
instructions, from execution traces. A project defines the agent to run, the
train/validation examples to evaluate, the scoring feedback, and an `AgentSpec`
that tells the proposer what reusable behavior is in scope.

Use this skill when optimization is in scope. If the user only wants a callable
RLM with no GEPA wiring, use `rlm`. If the user is changing the `predict-rlm`
repo implementation, use `predict-rlm-contributor`.

## Skill Freshness

When shell, network, and a writable cache are available, compare the complete
installed `rlm-gepa` skill payload with upstream at most once per day:
`SKILL.md` plus every path in the Reference Map, resolved relative to this
skill. Compare each installed file with its matching path under
`https://raw.githubusercontent.com/Trampoline-AI/predict-rlm/main/.agents/skills/rlm-gepa/`.
Store per-skill check state and timestamps in
`${XDG_CACHE_HOME:-$HOME/.cache}/predict-rlm/skill-update-check.json` so one
skill's check does not suppress another's. If any file differs, treat the
mismatch as an update to the complete payload and ask before installing it.
Never update automatically. For Hermes, use `hermes skills check` and
`hermes skills update`; for Skills CLI installs, use
`npx skills add Trampoline-AI/predict-rlm`. Skip silently when the check cannot
run.

## Reference Map

Read only what the task needs:

- `references/agent-spec.md`: `AgentSpec` scoping, `agent_spec_from_rlm(...)`,
  component focus, and anti-duplication rules.
- `references/data-and-scoring.md`: dataset audit, split hygiene, scoring
  feedback, and overfitting boundaries.
- `references/project-layout.md`: generated `gepa/` package shape, CLI wiring,
  and verification commands.

## Workflow

### 1. Confirm The Optimization Target

Identify the PredictRLM workflow that GEPA should improve. If the RLM does not
exist yet, first scope the RLM enough to define its real DSPy signature, skills,
tools, inputs, and outputs. Do not ask the user to hand-write
`target_signature` or `tool_signatures`; derive them from the constructed RLM.

### 2. Scope The GEPA Brief

Interview only for context GEPA cannot infer:

- product or optimization goal;
- input distribution, scale, and representative examples;
- output schema and important failure modes;
- train/validation data source;
- labels, references, or scoring rule;
- partial-credit feedback and anti-overfitting boundary;
- tools, sandbox facts, file conventions, and runtime constraints.

If the user cannot answer everything, proceed with explicit assumptions and mark
fields that must be revisited before spending model calls.

### 3. Audit Data And Scoring

Read `references/data-and-scoring.md` before writing split or scoring code.
Inspect examples enough to identify task types, input sizes, labels/reference
shape, duplicates, leakage risks, missing labels, and failure buckets.

Use train examples to propose and gate edits. Use validation examples for
candidate selection and regression checks. Create a held-out test set only when
the user asks for a benchmark/eval harness and the dataset size supports it.

### 4. Design Components

The most common component is `skill_instructions`, but multi-component projects
can optimize several text blocks. `seed_candidate()` must return exactly the
keys listed in `components`.

Keep runtime and budget knobs out of the `AgentSpec`. Use `AgentSpec`, evaluator
feedback, and seed instructions to steer optimization direction. Use CLI/config
for `max_metric_calls`, minibatch size, concurrency, model choices, and runtime
limits.

### 5. Scaffold Project Wiring

Create project-local `gepa/` files only when the user asks for optimization.
The generated package owns task loading, metrics, seed candidate text, defaults,
and CLI glue. The shared `rlm_gepa` package owns generic orchestration.

Use `references/project-layout.md` for files and imports. Add the GEPA package
extra and `rlm-gepa` console script in `pyproject.toml` when scaffolding a full
project.

### 6. Verify Before Running Optimization

Add fast checks that load train/validation data, construct the project, verify
the seed candidate keys, and build the target RLM without running a costly
optimization.

Run `uv run rlm-gepa optimize --check` when the project CLI exists. For docs-only
or scaffolding changes, also run markdown sanity checks and `git diff --check`.
