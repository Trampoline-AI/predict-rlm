"""PDF skill — pymupdf for reading, rendering, and writing PDFs in the sandbox."""

from predict_rlm import Skill

pdf_skill = Skill(
    name="pdf",
    instructions="""Use pymupdf to work with PDF files mounted in the sandbox.

## Opening and inspecting

    import pymupdf
    doc = pymupdf.open(path)
    print(f"Pages: {len(doc)}")
    toc = doc.get_toc()  # [[level, title, page_num], ...]
    metadata = doc.metadata  # dict with title, author, subject, etc.

## Reading pages — prefer visual rendering over raw text

Always render pages as images for analysis with predict(). Raw text
extraction loses layout, tables, headers, and formatting that are
critical for understanding documents. Use get_text() only for
keyword searches or when you need to find specific strings.

Render a page as an image:

    import base64
    pix = doc[page_num].get_pixmap(dpi=200)
    uri = f"data:image/png;base64,{base64.b64encode(pix.tobytes('png')).decode()}"
    result = await predict("page: dspy.Image -> ...", page=uri)

Render multiple pages in parallel:

    import asyncio, base64

    def render_page(doc, i):
        pix = doc[i].get_pixmap(dpi=200)
        return f"data:image/png;base64,{base64.b64encode(pix.tobytes('png')).decode()}"

    images = [render_page(doc, i) for i in range(len(doc))]
    tasks = [predict("page: dspy.Image -> ...", page=img) for img in images]
    results = await asyncio.gather(*tasks)

## Text extraction (for searching, not analysis)

    text = doc[page_num].get_text()           # plain text
    blocks = doc[page_num].get_text_blocks()   # [(x0,y0,x1,y1,text,block_no,type), ...]
    words = doc[page_num].get_text_words()     # [(x0,y0,x1,y1,word,block,line,word_no), ...]

## Searching

    results = doc[page_num].search_for("keyword")  # list of Rect objects

## Table extraction

    tables = doc[page_num].find_tables()
    for table in tables:
        data = table.extract()  # list of lists (rows x cols)

## Images

    images = doc[page_num].get_images()       # [(xref, smask, w, h, bpc, cs, ...), ...]
    img_bytes = doc.extract_image(xref)        # dict with "image" (bytes), "ext", etc.

## Annotations and links

    links = doc[page_num].get_links()          # [{"kind": ..., "uri": ..., ...}, ...]
    annots = list(doc[page_num].annots())      # annotation objects

## Writing and modifying

    doc[page_num].insert_text((x, y), "text")
    doc[page_num].add_redact_annot(rect)
    doc[page_num].apply_redactions()
    doc.save("/sandbox/output/field_name/modified.pdf")
    doc.close()

Always close the document when done.""",
    packages=["pymupdf"],
)
