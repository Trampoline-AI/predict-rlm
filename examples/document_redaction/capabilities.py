"""Skills for the document redaction example."""

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

If search_for() returns no hits, try a shorter substring or different
casing. Text in PDFs may be split across lines or have extra whitespace.

## Area redaction (signatures, logos, images)

For non-text elements, redact by bounding box coordinates. Coordinates
are in PDF points (72 pt/inch), origin at top-left:

    import pymupdf
    rect = pymupdf.Rect(x0, y0, x1, y1)
    page.add_redact_annot(rect, fill=(0, 0, 0))
    page.apply_redactions()

To estimate coordinates, render the page as an image and use predict()
to identify the bounding box of the element to redact.

## Verification

After applying redactions, re-render the page and verify visually:

    pix = page.get_pixmap(dpi=200)
    uri = f"data:image/png;base64,{base64.b64encode(pix.tobytes('png')).decode()}"
    check = await predict(
        "page: dspy.Image -> remaining_pii: list[str]",
        instructions="List any PII still visible on this page.",
        page=uri,
    )

## Important

- Call apply_redactions() after adding all annotations for a page —
  it permanently removes the underlying content.
- Save the redacted PDF to the output directory when done:
    doc.save("/sandbox/output/redacted_documents/filename.pdf")
- Always close the document: doc.close()""",
)

__all__ = ["pdf_skill", "redaction_skill"]
