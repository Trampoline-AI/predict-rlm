import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import base64
    import time
    from pathlib import Path
    from typing import Annotated

    import dspy
    import marimo as mo
    import pymupdf
    from dotenv import find_dotenv, load_dotenv
    from pydantic import BaseModel, Field

    from predict_rlm import File, PredictRLM, Skill
    from predict_rlm.files import SyncedFile
    from predict_rlm.skills import pdf as pdf_skill

    REPO_ROOT = Path(__file__).parent

    load_dotenv(find_dotenv(".env.development"))


@app.cell(hide_code=True)
def _():
    _hero = f"data:image/png;base64,{base64.b64encode((REPO_ROOT / 'docs/hero.png').read_bytes()).decode()}"

    mo.vstack(
        [
            mo.md("""
    # predict-rlm

    ## Production focused Self-harnessed LM runtime that allows the model to call itself using structured outputs.
            """),
            mo.image(src=_hero, width=700),
            mo.md("""
    *Based on RLM research by [Alex Zhang](https://x.com/a1zhang), PhD student at [MIT CSAIL](https://www.csail.mit.edu/)*
            """),
        ],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Why this matters

    ### we all know that with LMs >> long traces == lower quality

    | Traditional agent / harness | What's the problem? |
    |---|---|
    | Human-coded orchestration loop | The model can't adapt its own control flow to the task |
    | State lives in chat history | Working memory is fragile and rots with context length |
    | Mostly serial tool calls | The model can't shape its own reasoning to fit the task |
    | Harness is tightly coupled to the problem | New problem? New harness. Every time. |

    **Problems that break traditional harnesses:**

    - Deep research
    - Large-scale document analysis
    - Multi-file audits and compliance reviews
    - Data extraction across hundreds of pages
    - Anything that needs to run for a long time and examine a lot of data in detail
    """)
    return


@app.cell(hide_code=True)
def _():
    cache_toggle = mo.ui.switch(value=True, label="Cache LLM calls")
    cache_toggle
    return (cache_toggle,)


@app.cell(hide_code=True)
def _():
    _svg = (REPO_ROOT / "docs/harness_vs_rlm.svg").read_text()
    mo.vstack(
        [
            mo.md(
                """
    ---

    ## Recurrent Language Models (RLMs) in a nutshell

    - The model is a **program author** inside a sandboxed runtime,
      not a chat agent glued to a human-written loop.
    - Task state lives in **Python variables** — more robust than
      depending on token history alone. No context rot.
    - The model defines its own **control flow** — loops, branches,
      retries, parallel fan-outs — whatever the task demands.

    **The sandbox is the whole environment for the LM. It contains the inputs, the tools, everything. Even the prompt is encapsulated in a variable. The code is the reasoning substrate NOT A TOOL.**

    **4 knobs → 3 knobs.** Control flow moves from _your code_ into
    _the model's weights_ (C ∈ θ).
                """
            ),
            mo.Html(
                f'<div style="display:flex;justify-content:center">{_svg}</div>'
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## How PredictRLM turns schemas into behavior

    You define a typed shape for what to do and what to return —
    a [DSPy](https://dspy.ai) **Signature**. The model figures out
    the rest.

    ```python
    class AnalyzeImages(dspy.Signature):
        "\""Analyze images and answer the query."\""
        images: list[File] = dspy.InputField()
        program: str = dspy.InputField()
        answer: str = dspy.OutputField()

    rlm = PredictRLM(AnalyzeImages, sub_lm="openai/gpt-5.1")
    result = await rlm.acall(images=[File(path="page.png")], program="Count the letters.")
    ```

    No prompt engineering. No chain wiring. The model gets a sandbox
    with the inputs mounted, writes code to solve the task, and
    returns typed results.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ---

    ## Live Demo: Count every letter in a document page

    A task that seems simple but is hard for LLMs — the RLM solves it by
    using `predict()` for perception and Python for computation.
    """)
    return


@app.cell(hide_code=True)
def _():
    DEFAULT_NUM_RETRIES = 5

    lm_selector = mo.ui.dropdown(
        options={
            "Claude Opus 4.6": dict(
                model="anthropic/claude-opus-4-6",
                num_retries=DEFAULT_NUM_RETRIES,
            ),
            "Claude Sonnet 4.6": dict(
                model="anthropic/claude-sonnet-4-6",
                num_retries=DEFAULT_NUM_RETRIES,
            ),
            "GPT-5.4": dict(
                model="openai/gpt-5.4",
                reasoning_effort="none",
                num_retries=DEFAULT_NUM_RETRIES,
            ),
            "GPT-5.4 Mini": dict(
                model="openai/gpt-5.4-mini",
                reasoning_effort="low",
                num_retries=DEFAULT_NUM_RETRIES,
            ),
        },
        value="GPT-5.4 Mini",
        label="Main LM (writes code in the REPL)",
    )

    sub_lm_selector = mo.ui.dropdown(
        options={
            "GPT-5.1": "openai/gpt-5.1",
            "GPT-5 Mini": "openai/gpt-5-mini-2025-08-07",
            "Claude Haiku 4.5": "anthropic/claude-haiku-4-5-20251001",
            "Claude Sonnet 4.6": "anthropic/claude-sonnet-4-6",
            "Gemini 3 Flash Preview": "gemini/gemini-3-flash-preview",
        },
        value="GPT-5.1",
        label="Sub-LM (handles perception via predict())",
    )

    mo.hstack([lm_selector, sub_lm_selector])
    return lm_selector, sub_lm_selector


@app.cell(hide_code=True)
def _():
    _img_path = next((REPO_ROOT / "examples/image_analysis/sample/input").glob("Screenshot*"))
    _img_data = _img_path.read_bytes()
    _uri = f"data:image/png;base64,{base64.b64encode(_img_data).decode()}"

    mo.vstack(
        [
            mo.md(f"**Input image:** `{_img_path.name}`"),
            mo.image(src=_uri, width=500),
        ]
    )

    class LetterCounts(BaseModel):
        counts: dict[str, int] = Field(
            description="Mapping of uppercase letter to its count, e.g. {'A': 5, 'B': 2}. "
            "Only include letters that appear at least once. Alphabetical order."
        )

    class AnalyzeImages(dspy.Signature):
        """Analyze multiple images and answer the query about them.

        1. **List the image files** available in the input directory. Print
           their names and file sizes.

        2. **Load each image** as a base64 data URI using Python's base64 and
           pathlib modules.

        3. **Use predict()** with dspy.Image typed inputs to analyze the images
           in the context of the query. Process multiple images in parallel with
           asyncio.gather() if there are several.

        4. **Synthesize** the findings into a single answer that addresses the
           query across all images.
        """

        images: list[File] = dspy.InputField(desc="Image files to analyze")
        program: str = dspy.InputField(desc="A question about the images")
        answer: LetterCounts = dspy.OutputField(
            desc="Letter frequency counts from the image text"
        )

    return (AnalyzeImages,)


@app.cell
def _():

    analyze_img_program = """
    What letters appear in this image, and how many times does each letter
    appear?

    For each image:
    1. Extract the visible text multiple times (at least 2-3 extractions)
    2. Compare the extractions - if they differ, extract again until consistent
    3. Only after consistent extraction, count letters programmatically

    Use prompts like "Return ONLY the exact text visible, nothing else."
    Do all counting in Python, not via predict().
    Case-insensitive. Output letter stats in alphabetical order (A-Z).
    """.strip()
    return (analyze_img_program,)


@app.cell
def _():
    ### run the demo
    return


@app.cell(hide_code=True)
async def _(AnalyzeImages, analyze_img_program, lm_selector, sub_lm_selector):
    # RUN ME <3
    _img_path = next((REPO_ROOT / "examples/image_analysis/sample/input").glob("Screenshot*"))

    _lm = dspy.LM(**lm_selector.value, cache=False)
    _sub_lm = dspy.LM(sub_lm_selector.value, cache=False)

    _rlm = PredictRLM(
        AnalyzeImages,
        sub_lm=_sub_lm,
        max_iterations=10,
        verbose=True,
    )

    _start = time.perf_counter()
    with dspy.context(lm=_lm):
        img_result = await _rlm.acall(
            images=[File(path=str(_img_path.resolve()))],
            program=analyze_img_program,
        )
    img_duration = time.perf_counter() - _start
    img_lm_history = list(_lm.history)
    img_sub_lm_history = list(_sub_lm.history)

    img_result.answer.counts
    return img_duration, img_lm_history, img_sub_lm_history


@app.cell(hide_code=True)
def _(
    img_duration,
    img_lm_history,
    img_sub_lm_history,
    lm_selector,
    sub_lm_selector,
):
    _lm_cost = sum(e.get("cost", 0) or 0 for e in img_lm_history)
    _lm_input = sum(
        e.get("usage", {}).get("prompt_tokens", 0) or 0
        for e in img_lm_history
    )
    _lm_output = sum(
        e.get("usage", {}).get("completion_tokens", 0) or 0
        for e in img_lm_history
    )

    _sub_cost = sum(e.get("cost", 0) or 0 for e in img_sub_lm_history)
    _sub_input = sum(
        e.get("usage", {}).get("prompt_tokens", 0) or 0
        for e in img_sub_lm_history
    )
    _sub_output = sum(
        e.get("usage", {}).get("completion_tokens", 0) or 0
        for e in img_sub_lm_history
    )

    _total = _lm_cost + _sub_cost
    _mins, _secs = divmod(int(img_duration), 60)

    mo.md(
        f"""
    ### Run Stats

    | | Main LM (`{lm_selector.value['model']}`) | Sub-LM (`{sub_lm_selector.value}`) |
    |---|---|---|
    | Calls | {len(img_lm_history)} | {len(img_sub_lm_history)} |
    | Input tokens | {_lm_input:,} | {_sub_input:,} |
    | Output tokens | {_lm_output:,} | {_sub_output:,} |
    | Cost | ${_lm_cost:.2f} | ${_sub_cost:.2f} |

    **Completed in {_mins}m {_secs}s for ${_total:.2f} total.**
        """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ---

    ## Skills, files, and the sandbox

    Files are first-class inputs and outputs — the sandbox mounts
    input artifacts and syncs results back when the job is done.

    **Skills** bundle domain knowledge (instructions + packages + tools)
    so the model knows *how* to work with specific file types:

    ```python
    from predict_rlm.skills import pdf, spreadsheet

    rlm = PredictRLM(MySignature, skills=[pdf, spreadsheet])
    ```

    The model doesn't have to pretend a PDF is just more text. It can
    render pages as images, call `predict()` for visual analysis, and
    use host-side tools for heavy operations — all inside the same
    execution loop.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ---

    ## Live Demo: Redact PII from a 6-page employment agreement

    The RLM reads every page, identifies all PII, applies redactions,
    and verifies the result — autonomously. Skills provide
    `render_pdf_page()` and `apply_pdf_redactions()` as host-side tools.
    """)
    return


@app.cell(hide_code=True)
def _():
    _pdf_path = REPO_ROOT / "examples/document_redaction/sample/input/PNFS-Employment-Agreement-2025.pdf"
    _doc = pymupdf.open(str(_pdf_path))
    _page_count = len(_doc)
    _pix = _doc[0].get_pixmap(dpi=150)
    _uri = f"data:image/png;base64,{base64.b64encode(_pix.tobytes('png')).decode()}"
    _doc.close()

    mo.vstack(
        [
            mo.md(f"**Input:** `{_pdf_path.name}` ({_page_count} pages)"),
            mo.image(src=_uri, width=500),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    redact_lm_selector = mo.ui.dropdown(
        options={
            "Claude Opus 4.6": dict(
                model="anthropic/claude-opus-4-6",
                num_retries=5,
            ),
            "Claude Sonnet 4.6": dict(
                model="anthropic/claude-sonnet-4-6",
                num_retries=5,
            ),
            "GPT-5.4": dict(
                model="openai/gpt-5.4",
                reasoning_effort="low",
                num_retries=5,
            ),
        },
        value="GPT-5.4",
        label="Main LM",
    )

    redact_sub_lm_selector = mo.ui.dropdown(
        options={
            "GPT-5.1": "openai/gpt-5.1",
            "GPT-5 Mini": "openai/gpt-5-mini-2025-08-07",
            "Claude Haiku 4.5": "anthropic/claude-haiku-4-5-20251001",
            "Claude Sonnet 4.6": "anthropic/claude-sonnet-4-6",
            "Gemini 3 Flash Preview": "gemini/gemini-3-flash-preview",
        },
        value="Gemini 3 Flash Preview",
        label="Sub-LM",
    )

    mo.hstack([redact_lm_selector, redact_sub_lm_selector])
    return redact_lm_selector, redact_sub_lm_selector


@app.class_definition(hide_code=True)
class RedactionTarget(BaseModel):
    """A specific piece of text identified for redaction."""

    page: int = Field(description="0-indexed page number")
    text: str = Field(description="Exact text string to redact")
    category: str = Field(
        description="e.g. 'person_name', 'phone_number', 'address', 'email', 'ssn'"
    )
    reason: str = Field(description="Why this text should be redacted")


@app.class_definition(hide_code=True)
class PageRedactionSummary(BaseModel):
    """Summary of redactions applied to a single page."""

    page: int = Field(description="0-indexed page number")
    redaction_count: int = Field(description="Number of redactions applied")
    categories: list[str] = Field(default_factory=list)


@app.class_definition(hide_code=True)
class RedactionResult(BaseModel):
    """Result of the document redaction process."""

    total_redactions: int = Field(description="Total number of redactions applied")
    page_summaries: list[PageRedactionSummary] = Field(default_factory=list)
    targets: list[RedactionTarget] = Field(default_factory=list)


@app.cell
def _redactdocuments():
    class RedactDocuments(dspy.Signature):
        """Redact sensitive information from documents based on criteria.

        1. **Read the redaction criteria** (appended below) to understand what
           types of information must be redacted.

        2. **Survey the documents** -- file names, page counts, document types.

        3. **Inspect each page** visually and identify all text matching the
           redaction criteria.

        4. **Apply redactions** using solid black fill (0, 0, 0) for all
           redaction annotations -- text and non-text elements like signatures
           or logos. If a text match fails, try a shorter or different substring.

        5. **Verify the result** by re-rendering redacted pages and confirming
           *only* the sensitive content is gone. Non-sensitive content must
           still be readable.

        6. **Save the redacted PDFs** to the output directory and **produce the
           result** with counts, per-page summaries, and targets.
        """

        documents: list[File] = dspy.InputField(desc="PDF documents to redact")
        redacted_documents: list[File] = dspy.OutputField(desc="Redacted PDF files")
        result: RedactionResult = dspy.OutputField(
            desc="Redaction result with counts and per-page summaries"
        )

    redaction_criteria = """
    Redact all personally identifiable information (PII), including:

    1. **Names** -- Full names of individuals (not company or organization names)
    2. **Contact info** -- Phone numbers, email addresses, fax numbers
    3. **Addresses** -- Street addresses, P.O. boxes (not city/state/country)
    4. **Government IDs** -- Social security numbers, tax IDs, passport numbers
    5. **Financial info** -- Bank account numbers, credit card numbers
    6. **Signatures** -- Handwritten signatures (redact the bounding area)

    Also redact any dates found in the document, in any format.
    """.strip()
    return RedactDocuments, redaction_criteria


@app.cell(hide_code=True)
async def _(
    RedactDocuments,
    cache_toggle,
    redact_lm_selector,
    redact_sub_lm_selector,
    redaction_criteria,
):
    _pdf_path = REPO_ROOT / "examples/document_redaction/sample/input/PNFS-Employment-Agreement-2025.pdf"

    def _apply_pdf_redactions(
        path: Annotated[str, SyncedFile(writeback=True)],
        redactions: list[dict],
    ) -> str:
        """Apply redactions to a PDF, permanently removing content under black rectangles.

        Copy the input PDF to the output path first, then call this on the copy.

        Args:
            path: Path to the PDF file to redact (modified in place).
            redactions: List of redaction specs. Each dict must have:
                - page (int): 0-indexed page number
                And at least one of:
                - text (str): text to search for and redact
                - rect (list[float]): [x0, y0, x1, y1] bounding box to redact

        Returns:
            Summary of redactions applied.
        """
        from collections import defaultdict

        import pymupdf

        doc = pymupdf.open(path)
        applied = 0
        by_page: dict[int, list[dict]] = defaultdict(list)
        for r in redactions:
            by_page[r["page"]].append(r)

        for page_num in sorted(by_page):
            page = doc[page_num]
            for r in by_page[page_num]:
                if "text" in r:
                    hits = page.search_for(r["text"])
                    if not hits:
                        for word in r["text"].split():
                            if len(word) > 2:
                                hits.extend(page.search_for(word))
                    for rect in hits:
                        page.add_redact_annot(rect, fill=(0, 0, 0))
                        applied += 1
                if "rect" in r:
                    rect = pymupdf.Rect(r["rect"])
                    page.add_redact_annot(rect, fill=(0, 0, 0))
                    applied += 1
            page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE)

        import shutil
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        doc.save(tmp.name)
        doc.close()
        shutil.move(tmp.name, path)
        return f"Applied {applied} redactions across {len(by_page)} page(s)"

    _redaction_skill = Skill(
        name="redaction",
        instructions="""How to redact content from PDFs using the provided tools.

    ## Workflow

    1. Use render_pdf_page() to visually inspect each page
    2. Use predict() to identify all PII on the rendered page images
    3. Copy the input PDF to the output directory using shutil.copy()
    4. Call apply_pdf_redactions() with all identified text strings and
       bounding boxes — it permanently removes content under black rectangles
    5. Verify by re-rendering the redacted pages with render_pdf_page()

    ## Text redaction

    For text content, pass the exact text string in the redaction spec:

        result = await apply_pdf_redactions(
            path="/sandbox/output/redacted_documents/file.pdf",
            redactions=[
                {"page": 0, "text": "John Smith"},
                {"page": 0, "text": "555-1234"},
                {"page": 1, "text": "123 Main St"},
            ]
        )

    If a full text match fails, the tool automatically falls back to
    matching individual words. Prefer shorter, exact substrings.

    ## Area redaction (signatures, logos, images)

    For non-text elements, use predict() to estimate bounding box
    coordinates from the rendered page image, then pass them as a rect:

        result = await apply_pdf_redactions(
            path="/sandbox/output/redacted_documents/file.pdf",
            redactions=[
                {"page": 2, "rect": [100, 400, 300, 500]},
            ]
        )

    ## Important

    - Always copy the input to the output directory BEFORE calling
      apply_pdf_redactions — it modifies the file in place
    - Save to: /sandbox/output/redacted_documents/filename.pdf""",
        tools={"apply_pdf_redactions": _apply_pdf_redactions},
    )

    _signature = RedactDocuments.with_instructions(
        RedactDocuments.instructions
        + "\n\n# Redaction Criteria\n\n"
        + redaction_criteria
    )

    _lm = dspy.LM(**redact_lm_selector.value, cache=cache_toggle.value)
    _sub_lm = dspy.LM(redact_sub_lm_selector.value, cache=cache_toggle.value)

    _rlm = PredictRLM(
        _signature,
        sub_lm=_sub_lm,
        skills=[pdf_skill, _redaction_skill],
        max_iterations=30,
        verbose=True,
    )

    _start = time.perf_counter()
    with dspy.context(lm=_lm):
        redact_prediction = await _rlm.acall(
            documents=[File(path=str(_pdf_path.resolve()))],
        )
    redact_duration = time.perf_counter() - _start
    redact_lm_history = list(_lm.history)
    redact_sub_lm_history = list(_sub_lm.history)
    return (
        redact_duration,
        redact_lm_history,
        redact_prediction,
        redact_sub_lm_history,
    )


@app.cell(hide_code=True)
def _(redact_prediction):
    import shutil

    _result = redact_prediction.result
    _redacted_files = redact_prediction.redacted_documents

    _sections = [
        mo.md(
            f"### Redaction Complete: {_result.total_redactions} redactions applied"
        ),
    ]

    if _redacted_files and _redacted_files[0].path:
        _src = Path(_redacted_files[0].path)
        _output_dir = REPO_ROOT / "output"
        _output_dir.mkdir(exist_ok=True)
        _dest = _output_dir / _src.name
        shutil.copy2(_src, _dest)
        _sections.append(mo.md(f"**Saved to:** `{_dest.relative_to(REPO_ROOT)}`"))

        _doc = pymupdf.open(str(_dest))
        _page_images = []
        for _i in range(len(_doc)):
            _pix = _doc[_i].get_pixmap(dpi=150)
            _uri = f"data:image/png;base64,{base64.b64encode(_pix.tobytes('png')).decode()}"
            _page_images.append(
                mo.vstack([
                    mo.md(f"**Page {_i + 1}**"),
                    mo.image(src=_uri, width=500),
                ])
            )
        _doc.close()
        _sections.extend(_page_images)

    if _result.page_summaries:
        _rows = "\n".join(
            f"| {s.page} | {s.redaction_count} | {', '.join(s.categories)} |"
            for s in _result.page_summaries
        )
        _sections.append(
            mo.md(
                f"""
    ### Per-Page Summary

    | Page | Redactions | Categories |
    |---:|---:|---|
    {_rows}
                """
            )
        )

    mo.vstack(_sections)
    return


@app.cell(hide_code=True)
def _(
    redact_duration,
    redact_lm_history,
    redact_lm_selector,
    redact_sub_lm_history,
    redact_sub_lm_selector,
):
    _lm_cost = sum(e.get("cost", 0) or 0 for e in redact_lm_history)
    _lm_input = sum(
        e.get("usage", {}).get("prompt_tokens", 0) or 0
        for e in redact_lm_history
    )
    _lm_output = sum(
        e.get("usage", {}).get("completion_tokens", 0) or 0
        for e in redact_lm_history
    )

    _sub_cost = sum(e.get("cost", 0) or 0 for e in redact_sub_lm_history)
    _sub_input = sum(
        e.get("usage", {}).get("prompt_tokens", 0) or 0
        for e in redact_sub_lm_history
    )
    _sub_output = sum(
        e.get("usage", {}).get("completion_tokens", 0) or 0
        for e in redact_sub_lm_history
    )

    _total = _lm_cost + _sub_cost
    _mins, _secs = divmod(int(redact_duration), 60)

    mo.md(
        f"""
    ### Run Stats

    | | Main LM (`{redact_lm_selector.value['model']}`) | Sub-LM (`{redact_sub_lm_selector.value}`) |
    |---|---|---|
    | Calls | {len(redact_lm_history)} | {len(redact_sub_lm_history)} |
    | Input tokens | {_lm_input:,} | {_sub_input:,} |
    | Output tokens | {_lm_output:,} | {_sub_output:,} |
    | Cost | ${_lm_cost:.2f} | ${_sub_cost:.2f} |

    **6 pages redacted in {_mins}m {_secs}s for ${_total:.2f} total.**
        """
    )
    return


@app.cell(hide_code=True)
def _():
    _svg = (REPO_ROOT / "docs/bitter_lesson_spectrum.svg").read_text()
    mo.vstack(
        [
            mo.md(
                """
    ## The bigger bet: a smaller harness, a smarter model

    This repo is a bet on Rich Sutton's **bitter lesson**: methods that
    leverage computation ultimately dominate methods that leverage human
    knowledge.
    If the base model gets better at code, the runtime gets better too.
    **No rewiring. No rethinking that harness.** That's the bitter lesson
    playing out in your favour.
                """
            ),
            mo.Html(f'<div style="display:flex;justify-content:center">{_svg}</div>'),
            mo.md(
                """
    ## What matters isn't that you *can't* do this with a model in a harness.
    ## What matters is that a model *without* a harness can do exactly what your hand-engineered harness does. 
    Every time the model updates, your RLM gets better for free. Your harness doesn't.
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    _qr_repo_b64 = base64.b64encode(
        (REPO_ROOT / "docs/qr_repo.svg").read_bytes()
    ).decode()
    _qr_x_b64 = base64.b64encode(
        (REPO_ROOT / "docs/qr_x.svg").read_bytes()
    ).decode()

    _c = "background:#e8e8e8;padding:0.2em 0.5em;border-radius:4px;font-size:0.95em"
    _qr = "flex-shrink:0;width:160px;height:160px"

    mo.Html(f"""
    <hr style="margin-bottom:1.5em"/>
    <div style="max-width:720px;margin:0 auto">
      <h2 style="margin-bottom:0.6em">Your first RLM is one prompt away</h2>

      <p style="margin:0 0 1.2em 0">
        <code style="{_c}">uv add predict-rlm</code> or
        <code style="{_c}">pip install predict-rlm</code>
      </p>

      <p style="margin:0 0 0.2em 0">
        With your coding agent:
        <code style="{_c}">npx skills add Trampoline-AI/predict-rlm</code>
        &mdash; then ask:
      </p>
      <p style="margin:0 0 1.2em 0">
        <code style="{_c}">/rlm build an RLM that extracts line items from PDF invoices into a spreadsheet</code>
      </p>

      <p style="margin:0 0 1.5em 0">
        It's early &mdash; feedback, bug reports, and PRs are very welcome.
      </p>

      <div style="display:flex;align-items:center;gap:1.5em;margin-bottom:0.4em">
        <img src="data:image/svg+xml;base64,{_qr_repo_b64}" style="{_qr}"/>
        <div>
          <div style="font-size:1.4em;font-weight:bold">Trampoline-AI/predict-rlm</div>
          <div style="color:#666">Give us a start on GitHub! (already 145+!!)</div>
          <div style="color:#333">MIT LICENSED</div>
        </div>
      </div>

      <div style="display:flex;align-items:center;gap:1.5em">
        <img src="data:image/svg+xml;base64,{_qr_x_b64}" style="{_qr}"/>
        <div>
          <div style="font-size:1.4em;font-weight:bold">@GabLesperance</div>
          <div style="color:#666">Find me on X</div>
        </div>
      </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Appendix & othert shennanigans
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ---

    ## What it replaces, and what it does not

    predict-rlm is not a claim that every agent framework is obsolete.
    It's a claim that some tasks deserve a different control model.

    | Pattern | Strength | Limitation |
    |---|---|---|
    | ReAct-style loop | Simple and familiar | The loop is still fixed by the human designer |
    | Claude Code-style | Great for code-centric tasks | Results re-enter context as text blobs |
    | LangChain / CrewAI | Broad integration surface | Control flow remains externally managed |
    | **predict-rlm** | **Recursive, typed, sandboxed execution** | **Best for long, structured, stateful tasks** |

    The real difference is not features — it's *where intelligence is
    allowed to live*. The old model says humans author orchestration.
    predict-rlm says the model should author more of it, inside a
    safer runtime.
    """)
    return


if __name__ == "__main__":
    app.run()
