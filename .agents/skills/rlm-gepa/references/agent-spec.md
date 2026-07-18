# AgentSpec

Prefer `agent_spec_from_rlm(...)` for new projects. The RLM stays the source of
truth for the DSPy signature, output schema, skills, and tools.

```python
from rlm_gepa import agent_spec_from_rlm

agent_spec = agent_spec_from_rlm(
    build_rlm(SEED_SKILL_INSTRUCTIONS),
    use_cases=[
        "contract review with clause-level citations",
        "invoice analysis with total reconciliation",
    ],
    runtime_grounding_examples={
        "skills": ["document-analysis skill instructions are optimized"],
        "sandbox facts": ["Pyodide filesystem paths and package limits"],
        "document behaviors": ["tables may span pages", "OCR text can be missing"],
    },
    scoring_description=(
        "Score combines answer correctness and citation support. Feedback names "
        "missing findings, unsupported citations, and extraction errors."
    ),
)
```

Do not duplicate facts `agent_spec_from_rlm(...)` can derive. Add only context
GEPA cannot infer:

- transfer use cases beyond the benchmark;
- runtime-grounding examples the proposer must preserve;
- scoring signal and evaluator feedback shape;
- anti-overfitting boundaries;
- short product or optimization framing, only when it adds useful context.

Omit `agent_type` by default. Set it only when a concise product or optimization
anchor adds information not already present in the signature, tools, or output
schema.

## Components

`components` names mutable text fields. `seed_candidate()` must return exactly
those keys.

```python
class MyProject(RLMGepaProject):
    components = ("skill_instructions",)

    def seed_candidate(self) -> dict[str, str]:
        return {"skill_instructions": SEED_SKILL_INSTRUCTIONS}
```

Override `component_focus(component_name)` when each component needs a different
proposer brief. Keep component names stable so runs and candidate artifacts are
comparable.

## Proposer Boundaries

Patch-merge/crossover should be evidence-backed behavioral grafting from train
disagreement traces. Avoid broad synthesis, prompt concatenation, source text
imports, or benchmark-specific hacks. Domain specifics belong in `AgentSpec`,
seed/domain skills, runtime-grounding examples, or evaluator feedback.
