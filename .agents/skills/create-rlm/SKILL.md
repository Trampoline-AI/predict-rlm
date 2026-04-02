---
name: create-rlm
description: Build an RLM (Recursive Language Model) with predict-rlm. Use when the user wants to create an RLM that iteratively writes and executes Python code in a sandbox to solve a task, using predict() to call a sub-LM for language understanding, vision analysis, and structured extraction.
compatibility: Requires Python 3.11+, Deno, and the predict-rlm package (built on DSPy).
metadata:
  author: trampoline-ai
  version: "0.2"
---

# Building RLMs with predict-rlm

An RLM is a callable, pre-configured agent. It autonomously explores context, writes and executes code in a sandboxed REPL, calls tools, inspects results, and iterates until the task is done. Unlike a chat agent, an RLM is a function — you define its inputs, outputs, and tools, then call it from your code. It returns structured data, not chat messages.

## Architecture

The architecture is two-level:

1. **The outer LLM** (the RLM itself) writes and executes Python code in a sandboxed Pyodide/WASM REPL. It plans, orchestrates, and iterates.
2. **The sub-LM** (via `predict()`) handles perception and extraction — analyzing images, understanding text, and returning typed results.

The sub-LM supports `dspy.Image` type hints, making RLMs natively multimodal. The outer LLM's context stays small (code + tool results), while context-heavy work is offloaded to `predict()` calls. Each `predict()` call gets its own context window with the sub-LM.

## How it works

1. You define **inputs**, **outputs**, and **skills**
2. The outer LLM writes Python code in the sandbox
3. Inside the sandbox, it calls `await predict(signature, **kwargs)` to invoke the sub-LM
4. It iterates — exploring data, using skills, building up results, handling errors
5. When done, it calls `SUBMIT()` with the final structured output

### File I/O

Use `File` for file-typed fields. Behavior is determined by position in the signature:
- **Input field**: mounts the file from host into the sandbox at `/sandbox/input/{field_name}/`
- **Output field**: syncs from `/sandbox/output/{field_name}/` back to the host

`list[File]` works for multiple inputs and multiple outputs.

## Skills

Skills are the primary way to extend what an RLM can do inside its sandbox. The sandbox starts with just Python's standard library and `predict()` — skills add **PyPI packages**, **instructions**, **modules**, and **tools** on top.

### Built-in skills library

predict-rlm ships a library of pre-built skills:

```python
from predict_rlm.skills import pdf, spreadsheet
```

| Skill | Import | Packages | Modules | What it teaches the RLM |
|---|---|---|---|---|
| **pdf** | `from predict_rlm.skills import pdf` | `pymupdf` | — | Read, render, modify, and redact PDFs |
| **spreadsheet** | `from predict_rlm.skills import spreadsheet` | `openpyxl`, `pandas`, `formulas` | `formula_eval` | Build and modify Excel workbooks with formulas and formatting |

Most RLMs only need built-in skills. Use them directly in the service:

```python
from predict_rlm.skills import pdf as pdf_skill
from predict_rlm.skills import spreadsheet as spreadsheet_skill

predictor = PredictRLM(
    MySignature,
    sub_lm=self.sub_lm,
    skills=[pdf_skill, spreadsheet_skill],
)
```

### Custom skills

Create a `skills.py` only when the RLM needs domain-specific instructions beyond what the built-in skills provide. For example, the document redaction example adds a custom `redaction_skill` alongside the built-in `pdf` skill:

```python
from predict_rlm import Skill
from predict_rlm.skills import pdf as pdf_skill

redaction_skill = Skill(
    name="redaction",
    instructions="""How to redact content from PDFs using pymupdf.

## Text redaction
Search for text, create redaction annotations, then apply:
    page = doc[page_num]
    hits = page.search_for("sensitive text")
    for rect in hits:
        page.add_redact_annot(rect, fill=(0, 0, 0))
    page.apply_redactions()
...""",
)

__all__ = ["pdf_skill", "redaction_skill"]
```

### The Skill dataclass

```python
from predict_rlm import Skill

Skill(
    name="my-skill",                          # Short identifier
    instructions="How to approach...",         # Prose injected into the RLM prompt
    packages=["pandas", "openpyxl"],           # PyPI packages installed in the sandbox
    modules={"helper": "/path/to/helper.py"},  # Python files mounted as importable modules in the sandbox
    tools={"fetch": fetch_fn},                 # Host-side callable functions exposed to the RLM
)
```

Fields:
- **`name`** — identifier for the skill
- **`instructions`** — prose guidance injected into the RLM's system prompt
- **`packages`** — PyPI packages installed in the WASM sandbox before the RLM's first code execution
- **`modules`** — Python files mounted into the sandbox, keyed by import name (e.g. `{"formula_eval": "/path/to/formula_eval.py"}` makes `import formula_eval` work in sandbox code)
- **`tools`** — host-side callable functions exposed to the RLM (see Tools section below)

Skills can also bundle **host-side tools** via their `tools=` field. This is useful when a tool is part of a broader capability — for example, a "database" skill might include instructions for writing SQL, packages for result formatting, and a `query_db` tool that runs on the host:

```python
db_skill = Skill(
    name="database",
    instructions="Write SQL queries and use query_db() to execute them...",
    packages=["pandas"],
    tools={"query_db": query_db},
)
```

When skills are composed, their tools are merged alongside instructions and packages (tool name conflicts raise errors).

## Tools

Tools are **single host-side functions** that the RLM can call from the sandbox. They are for specific operations that **cannot run inside the sandbox** — things that need host access, network calls to authenticated APIs, database queries, or interactions with system resources.

Tools are NOT for general capabilities. If the RLM needs a Python library, use a **skill** with `packages=`. If it needs domain knowledge, use a skill with `instructions=`. Tools are the escape hatch for when the sandbox isn't enough.

```python
async def fetch_exchange_rate(currency: str, date: str) -> str:
    """Fetch the exchange rate for a currency on a given date.

    Args:
        currency: ISO currency code (e.g. "EUR", "GBP")
        date: Date in YYYY-MM-DD format

    Returns:
        JSON string with the exchange rate data
    """
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"https://api.example.com/rates/{currency}/{date}")
        return resp.text
```

The RLM sees the tool's docstring and can `await` it from the sandbox. Tools can be sync or async (async preferred for I/O).

Tools can be passed in two ways:

1. **Directly to PredictRLM** — for one-off tools specific to this RLM:

```python
predictor = PredictRLM(
    MySignature,
    sub_lm=self.sub_lm,
    skills=[pdf_skill],
    tools={"fetch_exchange_rate": fetch_exchange_rate},
)
```

2. **Bundled inside a Skill** — when the tool is part of a broader capability (see Skills section above).

### When to use a Skill vs tools

| Use a Skill when... | Use `tools=` when... |
|---|---|
| The RLM needs a **package** installed in the sandbox | The function must run on the **host** (API calls, DB queries, filesystem) |
| You need to teach the RLM **how to use** something | The tool's docstring is self-explanatory |
| The knowledge is **reusable** across RLMs | It's a single specific function for one RLM |
| Example: PDF manipulation, spreadsheet building, data analysis | Example: fetch a URL, query a database, call an authenticated API |

## Recommended file structure

Organize each RLM as follows:

```
my_rlm/
├── __init__.py       # Public exports (service class, schema, signature)
├── schema.py         # Pydantic output models
├── signature.py      # DSPy Signature (inputs/outputs + strategy docstring)
├── service.py        # DSPy Module wiring signature + PredictRLM + skills
└── skills.py         # (optional) Custom skill definitions beyond built-in skills
```

**Always create**: `schema.py`, `signature.py`, `service.py`, `__init__.py`
**Create when needed**: `skills.py` (only if the RLM needs domain-specific instructions beyond built-in skills)

## 1. `schema.py` — Output models

Define Pydantic models for structured outputs. Use `Field(description=...)` so the RLM knows what each field means.

```python
from pydantic import BaseModel, Field


class KeyDate(BaseModel):
    """A key date extracted from a document."""

    name: str = Field(description="e.g. 'Submission Deadline', 'Effective Date'")
    date: str = Field(description="ISO format date (YYYY-MM-DD)")
    time: str | None = Field(
        None, description="24-hour format (HH:MM), e.g. '14:00', '09:30'"
    )
    timezone: str | None = Field(
        None, description="Timezone code, e.g. 'EST', 'EDT', 'PST', 'UTC'"
    )


class KeyEntity(BaseModel):
    """A key entity extracted from a document."""

    name: str = Field(description="Name of the person, organization, or role")
    role: str | None = Field(None, description="Role or relationship to the document")
    contact: str | None = Field(None, description="Contact info if available")


class DocumentAnalysis(BaseModel):
    """Structured analysis of a document set."""

    report: str = Field(
        description="Full analysis as a well-formatted markdown report"
    )
    key_dates: list[KeyDate] = Field(
        default_factory=list, description="Important dates found in the documents"
    )
    key_entities: list[KeyEntity] = Field(
        default_factory=list,
        description="Key people, organizations, or roles mentioned",
    )
```

## 2. `signature.py` — Inputs, outputs, and strategy

A DSPy Signature declares the RLM's interface. The docstring becomes the system instructions — tell the RLM how to approach the task step by step:

```python
import dspy

from predict_rlm import File

from .schema import DocumentAnalysis


class AnalyzeDocuments(dspy.Signature):
    """Analyze documents and produce a structured report.

    1. **Read the report criteria** (appended below) to understand what
       information to extract and in what format.

    2. **Survey the documents** to understand what you're working with:
       file names, page counts, document types.

    3. **Gather information** systematically by rendering pages as images
       and using predict() to extract content.

    4. **Produce the report** following the format specified in the criteria.
       Use tables for structured data, prose for analysis and context.
    """

    documents: list[File] = dspy.InputField(
        desc="PDF documents to analyze"
    )
    analysis: DocumentAnalysis = dspy.OutputField(
        desc="Structured analysis with markdown report, key dates, and key entities"
    )
```

## 3. `service.py` — Wiring it together

Wrap signature + skills + PredictRLM into a reusable DSPy Module:

```python
import dspy

from predict_rlm import File, PredictRLM
from predict_rlm.skills import pdf as pdf_skill

from .schema import DocumentAnalysis
from .signature import AnalyzeDocuments


class DocumentAnalyzer(dspy.Module):
    """DSPy Module that wraps AnalyzeDocuments + PredictRLM."""

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

When using multiple skills or host-side tools:

```python
from predict_rlm.skills import pdf as pdf_skill
from predict_rlm.skills import spreadsheet as spreadsheet_skill

async def aforward(self, documents: list[File]) -> MyOutput:
    predictor = PredictRLM(
        MySignature,
        sub_lm=self.sub_lm,
        skills=[pdf_skill, spreadsheet_skill],
        tools={"fetch_exchange_rate": fetch_exchange_rate},
        ...
    )
```

## PredictRLM parameters

| Parameter | Description |
|-----------|-------------|
| `signature` | DSPy Signature class or string like `"docs -> answer"` |
| `lm` | Main LM that drives the RLM (writes code, orchestrates). String or `dspy.LM` |
| `sub_lm` | LM for the `predict()` tool (sub-LM calls). String or `dspy.LM` |
| `skills` | List of `Skill` instances — built-in or custom |
| `tools` | Host-side tool functions — dict of `{name: callable}` |
| `max_iterations` | Max REPL iterations (default 30) |
| `debug` | Print REPL code, output, errors to stderr (default False) |
| `verbose` | DSPy verbose logging (default False) |
| `allowed_domains` | Network domains the sandbox can access |

Both `lm` and `sub_lm` accept a model string (e.g. `"openai/gpt-5.4"`) or a `dspy.LM` instance. If `lm` is omitted, the current context LM from `dspy.context(lm=...)` is used.

## Key imports

```python
from predict_rlm import PredictRLM, Skill, File
from predict_rlm.skills import pdf, spreadsheet
```

## Reference

See `examples/document_analysis/`, `examples/document_redaction/`, `examples/invoice_processing/`, and `examples/contract_comparison/` for complete working examples.
