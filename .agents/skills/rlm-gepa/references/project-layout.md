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

Keep runtime and dataset defaults in `config.py`. Project-specific fields belong
on an `OptimizeConfig` subclass so the CLI can pass one resolved config to
`build_project(...)`.

```python
# my_rlm/gepa/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlm_gepa import OptimizeConfig


@dataclass
class MyGepaConfig(OptimizeConfig):
    train_data: Path = Path("my_rlm/bench/train.jsonl")
    val_data: Path = Path("my_rlm/bench/val.jsonl")

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            train_data=str(self.train_data),
            val_data=str(self.val_data),
        )
        return payload


def default_config() -> MyGepaConfig:
    return MyGepaConfig(
        executor_lm="openai/gpt-5.4-mini",
        executor_sub_lm="openai/gpt-5.4-mini",
        proposer_lm="anthropic/claude-sonnet-4-6",
        proposer_sub_lm="openai/gpt-5.4-mini",
        max_metric_calls=200,
        minibatch_size=10,
        concurrency=10,
    )
```

Put dataset conversion, the scoring boundary, candidate construction, and the
`RLMGepaProject` implementation in `project.py`. This document-analysis example
uses JSONL rows with `example_id`, `document_paths`, `required_findings`, and
`required_citations`. Adapt `load_examples(...)` and `score_result(...)` to the
project's real data and evaluator rather than hiding those decisions behind
undefined helpers.

```python
# my_rlm/gepa/project.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from predict_rlm import File, PredictRLM, Skill
from predict_rlm.skills import pdf as pdf_skill
from predict_rlm.trace import RunTrace
from rlm_gepa import (
    AgentSpec,
    EvaluationContext,
    RLMGepaExampleResult,
    RLMGepaProject,
    agent_spec_from_rlm,
)

from ..agent.signature import AnalyzeDocuments
from .config import MyGepaConfig, default_config


COMPONENT_SKILL = "skill_instructions"
SEED_SKILL_INSTRUCTIONS = """\
Review every provided document before answering. Tie each material finding to a
page or section citation, reconcile conflicting values explicitly, and state
when the documents do not contain evidence required by the requested analysis.
"""


@dataclass(frozen=True)
class EvalReference:
    required_findings: tuple[str, ...]
    required_citations: tuple[str, ...]


@dataclass(frozen=True)
class EvalExample:
    example_id: str
    document_paths: tuple[Path, ...]
    reference: EvalReference

    def rlm_kwargs(self) -> dict[str, Any]:
        return {
            "documents": [File(path=str(path)) for path in self.document_paths],
        }


def load_examples(path: Path) -> list[EvalExample]:
    examples: list[EvalExample] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            examples.append(
                EvalExample(
                    example_id=str(row["example_id"]),
                    document_paths=tuple(Path(value) for value in row["document_paths"]),
                    reference=EvalReference(
                        required_findings=tuple(row["required_findings"]),
                        required_citations=tuple(row["required_citations"]),
                    ),
                )
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid GEPA example") from exc
    return examples


def score_result(result: Any, reference: EvalReference) -> tuple[float, str]:
    """Score required findings and citations in the generated report."""
    analysis = getattr(result, "analysis", None)
    report = str(getattr(analysis, "report", "") or "")
    normalized_report = report.casefold()
    missing_findings = [
        finding
        for finding in reference.required_findings
        if finding.casefold() not in normalized_report
    ]
    missing_citations = [
        citation
        for citation in reference.required_citations
        if citation.casefold() not in normalized_report
    ]
    finding_score = _recall(reference.required_findings, missing_findings)
    citation_score = _recall(reference.required_citations, missing_citations)
    score = 0.7 * finding_score + 0.3 * citation_score

    if score == 1.0:
        return score, "All required findings and citations are present."
    feedback = [
        f"finding recall={finding_score:.3f}",
        f"citation recall={citation_score:.3f}",
    ]
    if missing_findings:
        feedback.append("missing findings: " + "; ".join(missing_findings))
    if missing_citations:
        feedback.append("missing citations: " + "; ".join(missing_citations))
    return score, "\n".join(feedback)


def _recall(expected: tuple[str, ...], missing: list[str]) -> float:
    return 1.0 if not expected else (len(expected) - len(missing)) / len(expected)


def build_rlm(
    skill_instructions: str,
    *,
    lm: Any = None,
    sub_lm: Any = None,
    max_iterations: int = 30,
    verbose: bool = False,
    debug: bool = False,
) -> PredictRLM:
    return PredictRLM(
        AnalyzeDocuments,
        lm=lm,
        sub_lm=sub_lm,
        max_iterations=max_iterations,
        verbose=verbose,
        debug=debug,
        skills=[
            pdf_skill,
            Skill(name="document-analysis", instructions=skill_instructions),
        ],
    )


def build_agent_spec() -> AgentSpec:
    return agent_spec_from_rlm(
        build_rlm(SEED_SKILL_INSTRUCTIONS),
        use_cases=[
            "contract review with clause-level citations",
            "invoice analysis with total reconciliation",
        ],
        runtime_grounding_examples={
            "skills": ["document-analysis instructions are optimized"],
            "sandbox facts": ["Pyodide filesystem paths and package limits"],
            "document behaviors": ["tables may span pages", "OCR text can be missing"],
        },
        scoring_description=(
            "Score is 70% required-finding recall and 30% required-citation "
            "recall. Feedback lists every missing expected substring."
        ),
    )


class MyProject(RLMGepaProject):
    project_name = "my-project"
    components = (COMPONENT_SKILL,)
    agent_spec = build_agent_spec()

    def __init__(self, config: MyGepaConfig):
        self.config = config

    def seed_candidate(self) -> dict[str, str]:
        return {COMPONENT_SKILL: SEED_SKILL_INSTRUCTIONS}

    def load_trainset(self) -> list[EvalExample]:
        return load_examples(self.config.train_data)

    def load_valset(self) -> list[EvalExample]:
        return load_examples(self.config.val_data)

    async def evaluate_example(
        self,
        candidate: dict[str, str],
        example: EvalExample,
        context: EvaluationContext,
    ) -> RLMGepaExampleResult:
        rlm = build_rlm(
            candidate[COMPONENT_SKILL],
            lm=context.lm,
            sub_lm=context.sub_lm,
            max_iterations=context.max_iterations,
            verbose=context.verbose_rlm,
            debug=context.debug_rlm,
        )
        result = await rlm.acall(**example.rlm_kwargs())
        score, feedback = score_result(result, example.reference)
        trace: RunTrace | None = getattr(result, "trace", None)
        return RLMGepaExampleResult(
            score=score,
            feedback=feedback,
            traces=[trace] if trace is not None else [],
            rlm_inputs={
                "example_id": example.example_id,
                "document_paths": [str(path) for path in example.document_paths],
            },
            example_id=example.example_id,
            error=None if trace is not None else "no RunTrace captured",
        )


def build_project(config: MyGepaConfig | None = None) -> RLMGepaProject:
    return MyProject(config or default_config())
```

## CLI

The generated `my_rlm.gepa:main` should call `run_project_cli(...)`.

```python
# my_rlm/gepa/cli.py
from rlm_gepa.cli import run_project_cli

from .config import default_config
from .project import build_project


def main() -> int:
    return run_project_cli(build_project, default_config())
```

Export that function from the package target used by `[project.scripts]`:

```python
# my_rlm/gepa/__init__.py
from .cli import main

__all__ = ["main"]
```

Make `python -m my_rlm.gepa` use the same entry point:

```python
# my_rlm/gepa/__main__.py
from .cli import main

raise SystemExit(main())
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
