"""PDF skill — pypdf for reading in the sandbox, host-side rendering via pymupdf."""

from __future__ import annotations

from typing import Annotated

from predict_rlm import Skill
from predict_rlm.files import SyncedFile


def render_pdf_page(
    path: Annotated[str, SyncedFile(writeback=False)],
    page_num: int,
    dpi: int = 200,
) -> str:
    """Render a PDF page as a base64-encoded PNG data URI for use with predict().

    Args:
        path: Path to the PDF file in the sandbox filesystem.
        page_num: 0-indexed page number.
        dpi: Image resolution in dots per inch (default 200).

    Returns:
        A data URI string "data:image/png;base64,..." for use as a dspy.Image value.
    """
    import base64

    import pymupdf

    doc = pymupdf.open(path)
    pix = doc[page_num].get_pixmap(dpi=dpi)
    uri = f"data:image/png;base64,{base64.b64encode(pix.tobytes('png')).decode()}"
    doc.close()
    return uri


pdf_skill = Skill(
    name="pdf",
    instructions="""Use pypdf to read PDFs and the render_pdf_page tool to view pages visually.

## Opening and inspecting

    from pypdf import PdfReader
    reader = PdfReader(path)
    print(f"Pages: {len(reader.pages)}")
    metadata = reader.metadata  # title, author, subject, etc.

## Reading pages — prefer visual rendering over raw text

Always render pages as images for analysis with predict(). Raw text
extraction loses layout, tables, headers, and formatting that are
critical for understanding documents. Use extract_text() only for
keyword searches or when you need to find specific strings.

Render a page as an image using the render_pdf_page tool:

    uri = await render_pdf_page(path=pdf_path, page_num=0, dpi=200)
    result = await predict("page: dspy.Image -> ...", page=uri)

Render and analyze multiple pages in parallel:

    import asyncio
    render_tasks = [render_pdf_page(path=pdf_path, page_num=i) for i in range(num_pages)]
    uris = await asyncio.gather(*render_tasks)

    analysis_tasks = [predict("page: dspy.Image -> ...", page=uri) for uri in uris]
    results = await asyncio.gather(*analysis_tasks)

## Text extraction (for searching, not analysis)

    text = reader.pages[page_num].extract_text()

## Writing PDFs

    from pypdf import PdfWriter
    writer = PdfWriter()
    writer.append(reader)
    with open("/sandbox/output/field_name/output.pdf", "wb") as f:
        writer.write(f)""",
    packages=["pypdf"],
    tools={"render_pdf_page": render_pdf_page},
)
