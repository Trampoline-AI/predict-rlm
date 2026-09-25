---
name: rlm-gepa
description: >-
  Add, evaluate, and optimize RLM-GEPA projects for an existing PredictRLM. Use
  when a user wants to improve reusable RLM instructions from execution traces,
  build a train/validation optimization harness, configure AgentSpec or
  OptimizeConfig, run optimize --check, or inspect RLM-GEPA run artifacts. For
  creating the base RLM itself, use rlm first.
compatibility: Requires Python 3.11+, Deno 2, predict-rlm[gepa] 0.8.0-alpha0 or newer, and a working PredictRLM.
metadata:
  author: Emile Riberdy
  version: "1.0"
---

# Optimize an RLM with RLM-GEPA

RLM-GEPA improves reusable text components, usually a `Skill.instructions`
string, using scored executions of a real PredictRLM. It is not a replacement
for initial RLM design. Start only after the base RLM has a concrete signature,
tools, a measurable outcome, and a working smoke path.

The system has two loops:

1. **Executor loop** — run each candidate RLM against train or validation
   examples, collecting outputs, `RunTrace` objects, scores, and feedback.
2. **Proposer loop** — use scored traces to make a surgical edit to a declared
   mutable component.

`AgentSpec` controls which transferable behaviors the proposer may improve.
`OptimizeConfig` controls budget, concurrency, models, and search behavior.
Keep those concerns separate.

# Workflow

## Step 1: Confirm optimization fit

Use RLM-GEPA when the user wants a reusable behavior to improve and can provide:

- a working PredictRLM with its real DSPy signature and tool surface;
- a text component worth optimizing, such as domain skill instructions;
- representative examples and a deterministic or carefully designed score;
- concrete feedback for imperfect outputs.

Do not optimize a vague prompt before the base RLM works. First fix functional
bugs, missing tools, weak output schemas, and unsupported sandbox dependencies.
Do not use GEPA for a one-off answer or a generic chat-agent prompt.

## Step 2: Audit data and split boundaries

Inspect examples before coding loaders or split logic. Identify input shapes,
reference-output shape, duplicates, shared source groups, missing labels, and
failure modes the score should expose.

Use consistent semantics:

- **Train** — examples available to proposer and candidate-gating work.
- **Validation** — examples used to select candidates and catch regressions.
- **Held-out test** — optional reporting-only data. Do not leak it into prompt
  text, proposer traces, candidate selection, or optimization feedback.

Prefer deterministic splits. Record the seed, grouping key, counts, and sampling
limits. Split by document, user, task family, or source whenever related rows
would otherwise leak across train and validation.

Keep the evaluator harness-side. The RLM may use safe task tools and documents,
but it must not see hidden answer keys, oracle APIs, or evaluator feedback while
solving an example.

## Step 3: Define the mutable component and AgentSpec

Build the concrete RLM first. Derive its signature and tools with
`agent_spec_from_rlm(...)`; do not duplicate those broad descriptions manually.
Add only context that the RLM cannot reveal:

- at least two distinct `use_cases` defining the transfer boundary;
- at least three nonempty `runtime_grounding_examples` groups containing stable
  runtime facts;
- `scoring_description` explaining score, partial credit, and hard failures;
- optional `counterfactual_axis_name` (`domains`, `task shapes`, `failure modes`,
  `task types`, or `problem classes`) and an anti-hack
  `domain_conventions_note`.

A useful `AgentSpec` teaches grounded behavior, not benchmark trivia. Name
actual tool contracts, library symbols, sandbox facts, and evaluator-visible
failure surfaces. Do not key instructions on task IDs, file names, row counts,
or answer artifacts.

```python
from predict_rlm import PredictRLM, Skill
from rlm_gepa import agent_spec_from_rlm


SEED_INSTRUCTIONS = "Inspect evidence before producing a grounded result."


def build_rlm(skill_instructions: str, *, lm=None, sub_lm=None, max_iterations: int = 30):
    return PredictRLM(
        AnalyzeDocuments,
        lm=lm,
        sub_lm=sub_lm,
        max_iterations=max_iterations,
        verbose=False,
        skills=[Skill(name="document-analysis", instructions=skill_instructions)],
    )


agent_spec = agent_spec_from_rlm(
    build_rlm(SEED_INSTRUCTIONS),
    use_cases=[
        "contract review with clause-level citations",
        "invoice analysis with total reconciliation",
    ],
    runtime_grounding_examples={
        "skills": ["document-analysis instructions are the mutable component"],
        "sandbox facts": ["Pyodide packages must be compatible with WASM"],
        "document behaviors": ["tables may span pages", "OCR text can be missing"],
    },
    scoring_description="Score combines correctness and citation support; feedback names missing or unsupported findings.",
    counterfactual_axis_name="task shapes",
    domain_conventions_note=(
        "Rules must transfer to unseen files; do not key behavior on file names, "
        "task IDs, or reference-answer artifacts."
    ),
)
```

## Step 4: Implement the project contract

Subclass `RLMGepaProject`. Its contract is intentionally small:

- `seed_candidate()` returns exactly the declared component keys and nonempty
  seed text.
- `load_trainset()` and `load_valset()` return nonempty sequences.
- `evaluate_example()` runs the candidate RLM, scores it, and returns a finite
  `RLMGepaExampleResult` with feedback, traces, stable `example_id`, and useful
  `rlm_inputs`.

```python
from dataclasses import dataclass
from typing import Any

from predict_rlm.trace import RunTrace
from rlm_gepa import EvaluationContext, RLMGepaExampleResult, RLMGepaProject
from .bench import load_examples, score_result



@dataclass
class EvalExample:
    example_id: str
    rlm_kwargs: dict[str, Any]
    reference: Any


class DocumentProject(RLMGepaProject):
    project_name = "document-analysis"
    components = ("skill_instructions",)
    agent_spec = agent_spec

    def seed_candidate(self) -> dict[str, str]:
        return {"skill_instructions": SEED_INSTRUCTIONS}

    def load_trainset(self) -> list[EvalExample]:
        return load_examples("train")

    def load_valset(self) -> list[EvalExample]:
        return load_examples("validation")

    async def evaluate_example(
        self,
        candidate: dict[str, str],
        example: EvalExample,
        context: EvaluationContext,
    ) -> RLMGepaExampleResult:
        rlm = build_rlm(
            candidate["skill_instructions"],
            lm=context.lm,
            sub_lm=context.sub_lm,
            max_iterations=context.max_iterations,
        )
        result = await rlm.acall(**example.rlm_kwargs)
        score, feedback = score_result(result, example.reference)
        trace: RunTrace | None = getattr(result, "trace", None)
        return RLMGepaExampleResult(
            score=score,
            feedback=feedback,
            traces=[trace] if trace is not None else [],
            rlm_inputs={"example_id": example.example_id, **example.rlm_kwargs},
            example_id=example.example_id,
        )
```

For an imperfect result, feedback must be nonempty. Good feedback identifies the
actual failing surface: page, cell, clause, tool response, assertion, timeout,
or schema mismatch. It describes what happened, not replacement prompt text.

Override `task_timeout_for_example(example, default_timeout)` or
`task_resources_for_example(example)` only when the workload genuinely has
per-example timeout or resource needs. Override `minibatch_group_id(example)`
when grouped examples must be sampled together.

## Step 5: Configure the project CLI

Keep optimization code local to the generated project:

```text
my_rlm/
├── agent/             # Existing PredictRLM signature, service, skills, tools
├── bench/             # Examples, split policy, deterministic scorer
└── gepa/
    ├── __init__.py
    ├── config.py      # OptimizeConfig defaults
    ├── project.py     # RLMGepaProject implementation
    └── cli.py         # run_project_cli wiring
```

Install the required extra and expose the project CLI:

```toml
[project]
dependencies = [
    "predict-rlm[gepa]>=0.8.0a0,<0.9",
]

[project.scripts]
rlm-gepa = "my_rlm.gepa.cli:main"
```

```python
from rlm_gepa import OptimizeConfig, run_project_cli

from .project import DocumentProject


def default_config() -> OptimizeConfig:
    return OptimizeConfig(
        max_metric_calls=1000,
        minibatch_size=25,
        concurrency=8,
        max_iterations=30,
    )


def build_project() -> DocumentProject:
    return DocumentProject()


def main() -> int:
    return run_project_cli(build_project, default_config())
```

Set model, budget, concurrency, timeout, selection, merge, and telemetry
choices in `OptimizeConfig` or explicit CLI flags. Do not hide experiment
behavior behind environment-only switches.

## Step 6: Verify before spending a budget

First validate the project without a real search:

```bash
uv run rlm-gepa optimize --check
```

Then use a small budget to validate the full loop. Only increase
`max_metric_calls`, `minibatch_size`, or `concurrency` after the trace, score,
and cost surfaces are trustworthy.

Useful commands:

```bash
# Project-specific eval surface, when the project implements it.
uv run rlm-gepa eval --dataset validation --limit 5

# Optimize after --check succeeds.
uv run rlm-gepa optimize --max-metric-calls 1000 --minibatch-size 25 --concurrency 8

# Inspect candidate quality and run cost.
uv run rlm-gepa stats runs/<run-dir> --format markdown
uv run rlm-gepa plot runs/<run-dir>
```

`plot` additionally requires the `gepa-viz` extra:
`predict-rlm[gepa,gepa-viz]`.

Optimization writes `run_metadata.json`, `gepa_state.bin`,
`optimization_summary.json`, `all_candidates.json`, task and proposer traces,
and `cost_log.jsonl` beneath the run directory. Treat those artifacts as the
evidence for accepting a candidate, not a single aggregate score.

# Guardrails

- Optimize only declared text components; do not let GEPA alter code, evaluator
  logic, data splits, or hidden answer access.
- Keep task execution and scoring deterministic where possible.
- Use the shared RLM-GEPA adapter semantics for concurrent evaluation rather
  than creating ad hoc task loops; it owns concurrency, timeouts, traces,
  progress, resources, and cost accounting.
- Use `--verbose-rlm` for human-readable RLM rollout traces and `--debug-rlm`
  for lifecycle diagnostics.
- Keep train evidence, validation selection, and held-out reporting separate.
- Treat patch merge as evidence-backed behavioral grafting from trace
  disagreement, never prompt concatenation or broad source-text import.
