# Skills

Skills are the primary way to extend what an RLM can do inside its sandbox. The sandbox starts with just Python's standard library and `predict()` — skills add **PyPI packages**, **instructions**, **modules**, and **tools** on top.

Skills are for **general capabilities** — teaching the RLM how to use a library or approach a domain. For single specialized functions (fetch a URL, query a database, call an API), use the `tools=` parameter directly instead.

## Defining a skill

```python
from predict_rlm import Skill

pdf_skill = Skill(
    name="pdf",
    instructions="Use pymupdf to open PDFs. Render pages as images (dpi=200) for analysis with predict().",
    packages=["pymupdf"],
)

rlm = PredictRLM(
    "documents -> tables: list[dict]",
    lm="openai/gpt-5.4",
    sub_lm="openai/gpt-5.1",
    skills=[pdf_skill],
)
```

See the [API reference](api.md#skill) for the full list of `Skill` fields.

## Composing skills

Pass multiple skills and their instructions, packages, modules, and tools are merged automatically:

```python
data_skill = Skill(
    name="data-analysis",
    instructions="Use pandas for tabular data. Print df.head() to inspect before processing.",
    packages=["pandas", "openpyxl"],
)

viz_skill = Skill(
    name="visualization",
    instructions="Use matplotlib for charts. Save figures to bytes, don't call plt.show().",
    packages=["matplotlib"],
)

rlm = PredictRLM(
    "spreadsheet, query -> analysis: str, chart: bytes",
    lm="openai/gpt-5.4",
    skills=[data_skill, viz_skill],
)
```

When skills are merged:

- **Instructions** are joined with section headers: `## Skill: {name}`
- **Packages** are deduplicated (preserving order)
- **Modules** are merged — duplicate import names raise `ValueError`
- **Tools** are merged — duplicate tool names raise `ValueError`

## Sandbox modules

Skills can mount Python modules directly into the sandbox via the `modules` field. This lets you ship custom Python code that the RLM can `import` — without publishing it to PyPI.

```python
from pathlib import Path

spreadsheet_skill = Skill(
    name="spreadsheet",
    instructions="Use openpyxl to build workbooks. Use formula_eval to verify formulas.",
    packages=["openpyxl", "pandas", "formulas"],
    modules={"formula_eval": str(Path(__file__).parent / "modules" / "formula_eval.py")},
)
```

The key maps the **import name** to the **host filesystem path** of the `.py` file. Inside the sandbox, the module becomes importable:

```python
# Inside the sandbox, the RLM can write:
from formula_eval import evaluate
report = evaluate("output.xlsx")
```

## Skill tools

Skills can expose tool functions to the RLM alongside the built-in `predict()` tool. Skill tools run on the host (not in the WASM sandbox), so they can access databases, APIs, the filesystem, or anything requiring native Python.

```python
async def fetch_document(doc_id: str) -> str:
    """Fetch a document by ID and return its URL."""
    return await my_storage.get_signed_url(doc_id)

doc_skill = Skill(
    name="document-access",
    instructions="Use fetch_document(doc_id) to get document URLs.",
    tools={"fetch_document": fetch_document},
)
```

Tools can be sync or async (async preferred for I/O). Their full docstrings are extracted and formatted into the RLM's prompt, so descriptive docstrings help the RLM know when and how to call them.

## Package compatibility

The sandbox runs [Pyodide](https://pyodide.org/) (CPython compiled to WebAssembly), which supports **pure-Python packages** out of the box via micropip. Packages with C extensions only work if they ship a pre-built Pyodide wheel — [many popular ones do](https://pyodide.org/en/stable/usage/packages-in-pyodide.html) (numpy, pandas, scipy, Pillow, pymupdf, etc.), but packages that rely on system libraries without a Pyodide build (e.g. psycopg2, torch) cannot be installed in the sandbox.

For these, expose the functionality as a **host-side tool** instead — the tool runs in your normal Python environment and the RLM calls it from the sandbox via the tool bridge.

## Built-in skills

predict-rlm ships a library of pre-built skills you can use directly:

```python
from predict_rlm.skills import pdf, spreadsheet, docx

rlm = PredictRLM(MySignature, skills=[pdf, spreadsheet, docx])
```

| Skill | Import | Packages | Modules | What it teaches the RLM | |
|---|---|---|---|---|---|
| **pdf** | `from predict_rlm.skills import pdf` | `pymupdf` | — | Read, render, modify, and redact PDFs | [source](../src/predict_rlm/skills/pdf/skill.py) |
| **spreadsheet** | `from predict_rlm.skills import spreadsheet` | `openpyxl`, `pandas`, `formulas` | `formula_eval` | Build and modify Excel workbooks with formulas and formatting | [source](../src/predict_rlm/skills/spreadsheet/skill.py) |
| **docx** | `from predict_rlm.skills import docx` | `python-docx` | `md2docx` | Read, write, and modify Word documents with tables, formatting, and styles | [source](../src/predict_rlm/skills/docx/skill.py) |

### pdf ([source](../src/predict_rlm/skills/pdf/skill.py))

Uses [pymupdf](https://pymupdf.readthedocs.io/) for opening, inspecting, rendering, and modifying PDFs.

Key patterns the skill teaches the RLM:
- **Visual rendering** — always render pages as images (`dpi=200`) for analysis with `predict()`, rather than relying on raw text extraction
- **Text extraction** — use `get_text()` only for keyword searches and string matching, not for understanding page content
- **Parallel processing** — fan out `predict()` calls across multiple pages with `asyncio.gather()`
- **Document modification** — search-and-redact workflows using `search_for()` and `add_redact_annot()`
- **Metadata and structure** — access TOC, annotations, links, and page-level metadata

### spreadsheet ([source](../src/predict_rlm/skills/spreadsheet/skill.py))

Uses [openpyxl](https://openpyxl.readthedocs.io/) for building and modifying Excel workbooks, [pandas](https://pandas.pydata.org/) for data manipulation, and [formulas](https://pypi.org/project/formulas/) for formula evaluation.

Includes the [`formula_eval`](../src/predict_rlm/skills/spreadsheet/modules/formula_eval.py) sandbox module with two functions:
- **`evaluate(path)`** — evaluates all formulas in a workbook and returns a report: `{"ok": bool, "formulas": int, "errors": int, "breakdown": dict, "computed": dict}`
- **`ensure_recalc_on_open(path)`** — sets the `fullCalcOnLoad` flag so Excel recalculates formulas when opened

Key patterns the skill teaches the RLM:
- **Formulas belong in Excel** — write Excel formulas (SUM, VLOOKUP, IF), don't compute values in Python and paste static numbers
- **Two libraries** — pandas for data operations, openpyxl for presentation and formatting
- **Structured assumptions** — each driver in a dedicated cell, referenced by formulas
- **Formula error guarding** — handle `#REF!`, `#DIV/0!`, `#VALUE!`, `#N/A`, `#NAME?`
- **Verification** — use `formula_eval.evaluate()` to verify formulas before submitting

### docx ([source](../src/predict_rlm/skills/docx/skill.py))

Uses [python-docx](https://python-docx.readthedocs.io/) for reading, writing, and modifying Word documents.

Includes the [`md2docx`](../src/predict_rlm/skills/docx/modules/md2docx.py) sandbox module with three functions:
- **`add_markdown(doc, text)`** — convert markdown-formatted text into properly styled Word elements (headings, bold, italic, code, lists, tables, blockquotes)
- **`add_styled_markdown(doc, text, confidence)`** — markdown with confidence-based coloring (high=black, medium=orange, placeholder=red)
- **`setup_document()`** — return a pre-configured document (1-inch margins, Arial 11pt)

Key patterns the skill teaches the RLM:
- **Document structure** — headings (levels 0–9), bullet and numbered lists, page breaks, sections
- **Page layout** — paper size, margins, orientation, landscape pages mid-document
- **Tables** — column widths, cell merging, alternating row shading
- **Headers/footers** — static text and dynamic page numbers via field codes
- **Content extraction** — body text, table text, and combined document-order extraction
- **Hyperlinks** — via relationship and XML layers
