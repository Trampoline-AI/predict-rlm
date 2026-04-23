"""Skills for the document redaction example."""

from __future__ import annotations

from collections import defaultdict
from typing import Annotated

import pymupdf

from predict_rlm import Skill
from predict_rlm.files import SyncedFile
from predict_rlm.skills import pdf as pdf_skill


def apply_pdf_redactions(
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


redaction_skill = Skill(
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
    tools={"apply_pdf_redactions": apply_pdf_redactions},
)

__all__ = ["pdf_skill", "redaction_skill"]
