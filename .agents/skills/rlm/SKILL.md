---
name: rlm
description: Plan and build an RLM (Recursive Language Model) with predict-rlm. Interactively defines inputs, outputs, skills, and architecture from a goal, then implements the code. Use when the user wants to create a new RLM or explore whether one is feasible.
compatibility: Requires Python 3.11+, Deno, and the predict-rlm package (built on DSPy).
metadata:
  author: Emile Riberdy
  version: "1.0"
---

# Build an RLM

An RLM is a callable, pre-configured agent. It autonomously explores context, writes and executes code in a sandboxed REPL, calls tools, inspects results, and iterates until the task is done. Unlike a chat agent, an RLM is a function — you define its inputs, outputs, and tools, then call it from your code. It returns structured data, not chat messages.

This skill has two phases:
1. **Plan** — interactively define the RLM with the user, research feasibility, produce a plan
2. **Build** — implement the plan as code files

**First action**: Enter plan mode using the EnterPlanMode tool.

---

# Phase 1: Plan

Work through these steps interactively. Do not skip steps or rush to the plan. Each step should involve asking the user questions and confirming alignment before moving on.

## Step 1: Goal Definition

Understand what the user wants to build.

Ask:
- What is the desired outcome? What does success look like?
- What is the input material? (documents, code, data, APIs, etc.)
- What does the output look like? (structured report, modified files, spreadsheet, etc.)

Then **validate RLM fit**. An RLM is the right tool when:
- The input is large and needs selective exploration (documents, datasets, codebases)
- The task is multi-step with tool use (extract -> transform -> validate)
- Actions modify state (redaction, form filling, generation)
- Parallel sub-LM calls are needed across many items
- File-to-file transformations (PDFs -> spreadsheets, documents -> reports)

If the task is better served by a single LLM call or a simple script, tell the user and suggest an alternative. Otherwise, proceed.

## Step 2: Input Design

Work with the user to define every input to the RLM.

For each input, determine:
- **Name** and **type**: `File`, `list[File]`, `str`, or a Pydantic model
- **Description**: what it contains and how the RLM uses it
- **Source**: user-provided file, API response, config, generated data

Key principles:
- Large content (PDFs, images, datasets) must be `File` references — the RLM accesses content on-demand through skills, keeping its context small
- Metadata (file paths, page counts, config flags) can be strings or Pydantic models
- Use `list[File]` for variable-count file inputs

Confirm the input design with the user before proceeding.

## Step 3: Output Design

Work with the user to define the structured output.

For each output field, determine:
- **Name**, **type**, and **description**
- Whether it's a Pydantic model (structured data), `File` (generated file), or primitive

Push for specificity — vague outputs lead to poor RLM performance. Sketch the Pydantic models with `Field(description=...)` annotations. Include nested models where appropriate.

Ask the user:
- What fields matter most? What would they check first?
- Are there any computed/derived fields (scores, summaries, counts)?
- Do they need output files (Excel, PDF, images)?

Confirm the output design with the user before proceeding.

## Step 4: Research

This step is **autonomous**. Tell the user you are researching, then do it.

Use web search and the Explore subagent to:

1. **Find Python packages** for the domain (e.g., `networkx` for graphs, `tree-sitter` for code parsing, `beautifulsoup4` for HTML).

2. **Check Pyodide compatibility**. The sandbox runs Pyodide (Python in WASM). Only **pure-Python wheels** or packages with **Emscripten builds** work. Search pypi.org for each package and check:
   - Does it have a `py3-none-any` wheel? (pure Python — works)
   - Does it have C extensions without Emscripten builds? (won't work in sandbox)
   - Is it in the Pyodide built-in package list? (check https://pyodide.org/en/stable/usage/packages-in-pyodide.html)

3. **Identify network needs**. Does the task require calling external APIs? If so, note the domains for `allowed_domains`.

4. **Identify host-side tool needs**. If any functionality cannot run in WASM (native binaries, C extensions, heavy computation), it must be a **host-side tool** — a Python function running on the host that the RLM calls like any other tool.

5. **Check for existing skills**. The built-in skills are:
   - `pdf` — pymupdf for PDF rendering, text extraction, manipulation
   - `spreadsheet` — openpyxl, pandas, formulas for Excel work
   - `docx` — python-docx for reading, writing, and modifying Word documents

Report findings to the user with a clear feasibility assessment. Flag any blockers.

## Step 5: Skill Design

Based on research, design the skill configuration.

### Built-in skills
List which built-in skills to use and why.

### Custom skills (if needed)
For each custom skill, define:
- **name**: short identifier
- **instructions**: prose guidance injected into the RLM's system prompt — teaches the RLM patterns and best practices. Be detailed; this is the primary way to control RLM behavior.
- **packages**: PyPI packages installed in the sandbox via micropip (must be Pyodide-compatible)
- **modules**: Python files mounted into the sandbox as importable modules
- **tools**: host-side callable functions exposed to the RLM

### Host-side tool design
For each host-side tool:
- Function name and signature with type hints
- Docstring (the RLM sees this to understand how to call it)
- What it does and why it must be host-side

Confirm the skill design with the user before proceeding.

## Step 6: Strategy and Architecture

### Signature strategy
Write the step-by-step strategy that goes in the signature's docstring. This is the RLM's playbook:
1. What to do first (survey/understand the input)
2. How to gather information (render pages, use predict() for extraction, call tools)
3. How to process and synthesize
4. What to produce and where to save output files

### Single vs chained RLMs
Evaluate whether this needs one RLM or multiple chained RLMs.

**Use a single RLM when**:
- The task is one coherent workflow
- All steps need the same context/state
- The iteration count stays reasonable (under 40)

**Use chained RLMs when**:
- There are distinct phases with different skill needs
- One phase produces artifacts consumed by another
- The combined task would exceed reasonable iteration counts
- Different phases benefit from different sub-LM models

If chaining, define each stage:
- Stage name, signature (inputs/outputs), skills, strategy
- The DAG: which stage feeds into which, with typed connections

### Configuration
- `max_iterations` estimate per RLM
- `allowed_domains` if network access is needed
- `sub_lm` recommendations (capability level needed)

## Feasibility Checklist

Before producing the final plan, verify:

- [ ] All proposed packages are Pyodide-compatible (or have host-side fallbacks)
- [ ] Network access needs are identified with specific domains
- [ ] Host-side tools are defined for anything that can't run in WASM
- [ ] Iteration count is reasonable (under 50 per RLM)
- [ ] Input sizes are manageable (or chunking strategy is defined)
- [ ] Output schemas are specific enough for reliable extraction
- [ ] The task is achievable — no unsupported capabilities assumed

## Plan Output

Write the plan to the Claude Code plan file with these sections:

1. **Overview** — one paragraph: what, why, and expected workflow
2. **File manifest** — every file to create with a one-line description
3. **Input schemas** — complete Pydantic model code for `schema.py`
4. **Output schemas** — complete Pydantic model code for `schema.py`
5. **Signature** — complete `signature.py` code with strategy docstring
6. **Skills configuration** — built-in imports + custom `Skill(...)` definitions + tool signatures
7. **Service architecture** — single RLM wiring or chained DAG:
   ```
   Stage1(documents) --[ExtractedData]--> Stage2(extracted) --[Report]--> Stage3(report)
   ```
8. **Feasibility notes** — constraints, risks, alternatives
9. **Estimated complexity** — iteration count, sub-LM calls, cost range, runtime

After writing the plan, use ExitPlanMode to get user approval. Once approved, proceed to Phase 2.

---

# Phase 2: Build

Implement the approved plan. Create all files following the patterns below.

## File structure

```
my_rlm/
├── __init__.py       # Public exports (service class, schema, signature)
├── schema.py         # Pydantic models for inputs AND outputs
├── signature.py      # DSPy Signature (inputs/outputs + strategy docstring)
├── service.py        # DSPy Module wiring signature + PredictRLM + skills
└── skills.py         # (optional) Custom skill definitions beyond built-in skills
```

**Always create**: `schema.py`, `signature.py`, `service.py`, `__init__.py`
**Create when needed**: `skills.py` (only if the RLM needs domain-specific instructions beyond built-in skills)

## schema.py — Pydantic models

Define models for structured inputs and outputs. Use `Field(description=...)` so the RLM knows what each field means.

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


class DocumentAnalysis(BaseModel):
    """Structured analysis of a document set."""

    report: str = Field(
        description="Full analysis as a well-formatted markdown report"
    )
    key_dates: list[KeyDate] = Field(
        default_factory=list, description="Important dates found in the documents"
    )
```

## signature.py — Inputs, outputs, and strategy

The docstring becomes the RLM's system instructions — tell the RLM how to approach the task step by step:

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

## service.py — Wiring it together

Wrap signature + skills + PredictRLM into a reusable DSPy Module:

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

### Chaining pattern (multiple RLMs)

```python
async def aforward(self, documents: list[File]):
    # Stage 1: Extract
    extractor = PredictRLM(ExtractSignature, sub_lm=self.sub_lm, skills=[pdf_skill])
    extracted = await extractor.acall(documents=documents)

    # Stage 2: Analyze (uses output from stage 1)
    analyzer = PredictRLM(AnalyzeSignature, sub_lm=self.sub_lm, skills=[analysis_skill])
    result = await analyzer.acall(data=extracted.data)

    return result
```

## skills.py — Custom skills

Create only when the RLM needs domain-specific instructions beyond built-in skills.

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

---

# Architecture Reference

Use this reference to ensure plans and implementations are accurate. Do not hallucinate parameters or patterns.

## How an RLM works

The architecture is two-level:

1. **The outer LLM** (the RLM itself) writes and executes Python code in a sandboxed Pyodide/WASM REPL. It plans, orchestrates, and iterates.
2. **The sub-LM** (via `predict()`) handles perception and extraction — analyzing images, understanding text, and returning typed results. Each `predict()` call gets its own context window.

The outer LLM's context stays small (code + tool results), while context-heavy work is offloaded to `predict()` calls.

## File I/O

Use `File` for file-typed fields:
- **Input field**: mounts the file from host into the sandbox at `/sandbox/input/{field_name}/`
- **Output field**: syncs from `/sandbox/output/{field_name}/` back to the host

```python
from predict_rlm import File

# Input: File(path="/absolute/path/to/file.pdf")
# Output: declared as File output field, RLM writes to /sandbox/output/<field>/
```

## PredictRLM constructor

```python
PredictRLM(
    signature: type[Signature] | str,     # DSPy signature class
    lm: dspy.LM | str | None = None,      # Main LM (code generation)
    sub_lm: dspy.LM | str | None = None,  # Sub-LM for predict() calls
    max_iterations: int = 30,
    max_llm_calls: int = 50,
    verbose: bool = False,
    tools: dict[str, Callable] | list[Callable] | None = None,
    allowed_domains: list[str] | None = None,
    skills: list[Skill] | None = None,
    debug: bool = False,
    output_dir: str | Path | None = None,
)
```

Both `lm` and `sub_lm` accept a model string (e.g. `"openai/gpt-5.4"`) or a `dspy.LM` instance. If `lm` is omitted, the current context LM from `dspy.context(lm=...)` is used.

## Skill dataclass

```python
from predict_rlm import Skill

Skill(
    name="my-skill",                          # Short identifier
    instructions="How to approach...",         # Prose injected into the RLM prompt
    packages=["pandas", "openpyxl"],           # PyPI packages installed in the sandbox
    modules={"helper": "/path/to/helper.py"},  # Python files mounted as importable modules
    tools={"fetch": fetch_fn},                 # Host-side callable functions exposed to the RLM
)
```

Skills can bundle **host-side tools** via their `tools=` field. When skills are composed, their tools are merged alongside instructions and packages (tool name conflicts raise errors).

## Built-in skills

```python
from predict_rlm.skills import pdf as pdf_skill          # pymupdf
from predict_rlm.skills import spreadsheet as spreadsheet_skill  # openpyxl, pandas, formulas
from predict_rlm.skills import docx as docx_skill        # python-docx
```

| Skill | Packages | Modules | What it teaches the RLM |
|---|---|---|---|
| **pdf** | `pymupdf` | — | Read, render, modify, and redact PDFs |
| **spreadsheet** | `openpyxl`, `pandas`, `formulas` | `formula_eval` | Build and modify Excel workbooks with formulas and formatting |
| **docx** | `python-docx` | `md2docx` | Read, write, and modify Word documents with tables, formatting, and styles |

## Tools

Tools are **host-side functions** the RLM can call from the sandbox. Use them for operations that cannot run inside the sandbox — host access, authenticated APIs, database queries, system resources.

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

Tools can be passed directly to PredictRLM via `tools={"name": fn}` or bundled inside a Skill via `tools=`.

### When to use a Skill vs tools

| Use a Skill when... | Use `tools=` when... |
|---|---|
| The RLM needs a **package** installed in the sandbox | The function must run on the **host** (API calls, DB queries, filesystem) |
| You need to teach the RLM **how to use** something | The tool's docstring is self-explanatory |
| The knowledge is **reusable** across RLMs | It's a single specific function for one RLM |

## predict() tool (inside sandbox)

The RLM can call `predict()` for sub-LM perception/extraction:
```python
result = await predict(
    "image: dspy.Image -> items: list[Item]",
    instructions="Extract all line items from this invoice page",
    image=page_image,
)
```
Each predict() call gets its own context window. Supports `dspy.Image` for multimodal.

## Key imports

```python
from predict_rlm import PredictRLM, Skill, File
from predict_rlm.skills import pdf, spreadsheet, docx
```

## WASM sandbox constraints

- Only pure-Python wheels or Pyodide built-in packages work
- No subprocess, no native binaries, no C extensions (unless Emscripten-built)
- Network access requires `allowed_domains` whitelist
- File I/O is within the sandbox filesystem
- Host-side tools bridge the gap for anything WASM can't do
