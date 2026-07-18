# RLM-GEPA Project Layout

Create project-local optimization wiring only when the user asks for GEPA or
prompt/skill optimization. Start from the packaged project layout in the `rlm`
skill and add `bench/` and `gepa/` under the import package.

```text
my_rlm/
├── pyproject.toml
├── my_rlm/
│   ├── __init__.py
│   ├── agent/       # PredictRLM signature, schema, service, skills/tools
│   ├── bench/       # optional eval loaders/scoring/fixtures
│   └── gepa/
│       ├── __init__.py
│       ├── config.py
│       ├── project.py
│       ├── cli.py
│       └── __main__.py
└── tests/
```

The generated `gepa/` package owns train/validation loading, metric feedback,
seed candidate text, defaults, and CLI glue. The shared `rlm_gepa` package
provides optimizer runtime and CLI helpers.

## pyproject.toml

Add GEPA dependencies and a project-local CLI when optimization is in scope.

```toml
dependencies = [
    "predict-rlm[gepa,gepa-viz]>=0.8.0-alpha0,<0.9",
]

[project.scripts]
rlm-gepa = "my_rlm.gepa:main"

[tool.predict-rlm.generated]
predict_rlm_version = "0.8.0-alpha0"
skill_version = "3.0"
layout = "agent-tools-bench-gepa"
features = ["agent", "bench", "rlm-gepa"]
```

## Project Skeleton

```python
from dataclasses import dataclass
from typing import Any

from predict_rlm import PredictRLM, Skill
from predict_rlm.trace import RunTrace
from rlm_gepa import (
    EvaluationContext,
    RLMGepaExampleResult,
    RLMGepaProject,
    agent_spec_from_rlm,
)

from ..agent.signature import AnalyzeDocuments


SEED_SKILL_INSTRUCTIONS = "Initial domain instructions for the RLM."


@dataclass
class EvalExample:
    example_id: str
    rlm_kwargs: dict[str, Any]
    reference: Any


def build_rlm(skill_instructions: str, *, lm=None, sub_lm=None, max_iterations=30):
    return PredictRLM(
        AnalyzeDocuments,
        lm=lm,
        sub_lm=sub_lm,
        max_iterations=max_iterations,
        skills=[Skill(name="document-analysis", instructions=skill_instructions)],
    )


class MyProject(RLMGepaProject):
    project_name = "my-project"
    components = ("skill_instructions",)
    agent_spec = agent_spec_from_rlm(build_rlm(SEED_SKILL_INSTRUCTIONS), ...)

    def seed_candidate(self) -> dict[str, str]:
        return {"skill_instructions": SEED_SKILL_INSTRUCTIONS}

    def load_trainset(self):
        return [...]

    def load_valset(self):
        return [...]

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
        traces = [trace] if trace is not None else []

        return RLMGepaExampleResult(
            score=score,
            feedback=feedback,
            traces=traces,
            rlm_inputs={"example_id": example.example_id, **example.rlm_kwargs},
            example_id=example.example_id,
        )
```

## CLI

The generated `my_rlm.gepa:main` should call `run_project_cli(...)`.

```python
from rlm_gepa.cli import run_project_cli

from .config import default_config
from .project import build_project


def main() -> int:
    return run_project_cli(build_project, default_config())
```

Use `optimize --check` before a real run:

```bash
uv run rlm-gepa optimize --check
```

If `bench/` exists, expose seed, validation, and held-out evaluation through the
same CLI only when the user asks for eval commands.

For eval and optimization CLIs, route task execution through
`rlm_gepa.runtime.adapter.RLMGepaAdapter` rather than bespoke `asyncio.gather`
loops. Project-local `bench/` code owns dataset selection, candidate loading,
task setup, and `eval.json` summary shaping; the shared adapter owns concurrency,
per-task timeouts, progress display, verbose RLM logs, `task_traces/*.jsonl`,
and `cost_log.jsonl`. Write `eval.json` in the run directory so
`rlm-gepa stats <run_dir>` works for held-out evals as well as optimization runs.
