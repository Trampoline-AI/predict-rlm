# Project Layout

Default to a grouped package. Keep the root package thin and put the callable
RLM under `agent/`. Add `tools/` and `bench/` only when the selected delivery
scope needs them.

```text
my_rlm/
├── pyproject.toml
├── my_rlm/
│   ├── __init__.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── schema.py
│   │   ├── signature.py
│   │   ├── service.py
│   │   └── skills.py    # optional custom skills
│   ├── tools/           # optional host-side tools/helpers
│   └── bench/           # optional eval dataset/scoring code
└── tests/
    └── test_smoke.py
```

Always create `pyproject.toml`, `my_rlm/__init__.py`,
`my_rlm/agent/schema.py`, `my_rlm/agent/signature.py`,
`my_rlm/agent/service.py`, `my_rlm/agent/__init__.py`, and
`tests/test_smoke.py`. Replace `my_rlm` with the import package name. Do not add
compatibility shims for old flat module names in newly generated projects.

## pyproject.toml

Record generated-with metadata so readers know the target API/layout version.
Use the current package version unless the user explicitly pins another one.

```toml
[project]
name = "my-rlm"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "predict-rlm>=0.8.0-alpha0,<0.9",
]

[tool.predict-rlm.generated]
predict_rlm_version = "0.8.0-alpha0"
skill_version = "3.0"
layout = "agent-tools-bench"
features = ["agent"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["my_rlm"]
```

For examples inside the `predict-rlm` monorepo, an editable path source is fine,
but keep the metadata table.

## Package Exports

Keep package imports deliberate. Re-export the callable service through both
package layers so callers can use `from my_rlm import DocumentAnalyzer`.

```python
# my_rlm/agent/__init__.py
from .service import DocumentAnalyzer
from .signature import AnalyzeDocuments

__all__ = ["AnalyzeDocuments", "DocumentAnalyzer"]
```

```python
# my_rlm/__init__.py
from .agent import AnalyzeDocuments, DocumentAnalyzer

__all__ = ["AnalyzeDocuments", "DocumentAnalyzer"]
```

## Schema Pattern

Define models for structured inputs and outputs. Use `Field(description=...)` so
the RLM knows what each field means.

```python
from pydantic import BaseModel, Field


class KeyDate(BaseModel):
    """A key date extracted from a document."""

    name: str = Field(description="e.g. 'Submission Deadline', 'Effective Date'")
    date: str = Field(description="ISO format date (YYYY-MM-DD)")
    time: str | None = Field(None, description="24-hour format, e.g. '14:00'")
    timezone: str | None = Field(None, description="Timezone code, e.g. 'UTC'")


class DocumentAnalysis(BaseModel):
    """Structured analysis of a document set."""

    report: str = Field(description="Full analysis as a markdown report")
    key_dates: list[KeyDate] = Field(
        default_factory=list,
        description="Important dates found in the documents",
    )
```

## Signature Pattern

The signature docstring is the RLM's operating strategy.

```python
import dspy

from predict_rlm import File

from .schema import DocumentAnalysis


class AnalyzeDocuments(dspy.Signature):
    """Analyze documents and produce a structured report.

    1. Read the report criteria.
    2. Survey the documents: file names, page counts, and document types.
    3. Gather information by rendering pages and using predict() for extraction.
    4. Produce the requested report with grounded structured fields.
    """

    documents: list[File] = dspy.InputField(desc="PDF documents to analyze")
    analysis: DocumentAnalysis = dspy.OutputField(
        desc="Structured analysis with report, key dates, and entities"
    )
```

## Service Pattern

Wrap the signature and skills in a reusable DSPy module.

```python
import dspy

from predict_rlm import File, PredictRLM
from predict_rlm.skills import pdf as pdf_skill

from .schema import DocumentAnalysis
from .signature import AnalyzeDocuments


class DocumentAnalyzer(dspy.Module):
    def __init__(
        self,
        sub_lm: dspy.LM | str | None = None,
        max_iterations: int = 30,
        verbose: bool = False,
        debug: bool = False,
    ):
        self.sub_lm = sub_lm
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.debug = debug

    async def aforward(
        self, documents: list[File], criteria: str
    ) -> DocumentAnalysis:
        signature = AnalyzeDocuments.with_instructions(
            AnalyzeDocuments.instructions + "\n\n# Task\n\n" + criteria.strip()
        )
        predictor = PredictRLM(
            signature,
            sub_lm=self.sub_lm,
            skills=[pdf_skill],
            max_iterations=self.max_iterations,
            verbose=self.verbose,
            debug=self.debug,
        )
        result = await predictor.acall(documents=documents)
        return result.analysis
```

## Chaining Pattern

Use chained RLMs only for distinct phases with different skills, budgets, or
typed artifacts.

```python
async def aforward(self, documents: list[File]):
    extractor = PredictRLM(ExtractSignature, sub_lm=self.sub_lm, skills=[pdf_skill])
    extracted = await extractor.acall(documents=documents)

    analyzer = PredictRLM(AnalyzeSignature, sub_lm=self.sub_lm, skills=[analysis_skill])
    return await analyzer.acall(data=extracted.data)
```

## Smoke Tests

Default smoke tests must be fast and must not require network access, API keys,
Deno, Pyodide, or LLM calls.

```python
def test_service_constructs():
    from my_rlm import DocumentAnalyzer

    service = DocumentAnalyzer(max_iterations=1, verbose=False, debug=False)
    assert service.max_iterations == 1


def test_signature_has_fields():
    from my_rlm.agent.signature import AnalyzeDocuments

    assert AnalyzeDocuments.input_fields
    assert AnalyzeDocuments.output_fields
```

If an end-to-end check is useful, add it as a separate integration test gated by
explicit credentials or an environment flag.
