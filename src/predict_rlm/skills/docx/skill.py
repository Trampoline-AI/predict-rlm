"""Docx skill — python-docx for reading, writing, and modifying Word documents in the sandbox."""

from pathlib import Path

from predict_rlm import Skill

_MODULES_DIR = Path(__file__).parent / "modules"

docx_skill = Skill(
    name="docx",
    instructions="""Use python-docx to work with Word documents (.docx) in the sandbox.

# Word Document Operations Guide

All operations use the `python-docx` library — pure Python, no external applications, no subprocess calls. Everything runs safely inside a sandboxed environment.

---

## Choosing Your Approach

| Goal | Method |
|------|--------|
| Generate a new document | Build with `Document()`, add content, save |
| Read or extract text | Open with `Document('file.docx')`, iterate paragraphs and tables |
| Modify an existing file | Load it, change what you need, save to a new path |
| Adjust layout or styling | Manipulate section properties, paragraph formats, and run fonts |

---

## Generating a New Document

```python
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document()

# Set base font for the Normal style
style = doc.styles["Normal"]
style.font.name = "Arial"
style.font.size = Pt(11)

# Title
title = doc.add_heading("Annual Review", level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

# Body text with mixed formatting
p = doc.add_paragraph()
p.add_run("The fiscal year ended with ").font.size = Pt(11)
highlight = p.add_run("record-setting margins")
highlight.bold = True
highlight.font.color.rgb = RGBColor(0, 102, 51)
p.add_run(".")

doc.save("annual_review.docx")
```

### Headings

`add_heading(text, level)` accepts levels 0 through 9. Level 0 renders as the document title style; levels 1-3 are most common for section structure.

```python
doc.add_heading("Executive Summary", level=1)
doc.add_heading("Revenue Breakdown", level=2)
doc.add_heading("By Region", level=3)
```

### Bullet and Numbered Lists

Use the built-in list styles rather than manually inserting bullet characters:

```python
doc.add_paragraph("First finding", style="List Bullet")
doc.add_paragraph("Second finding", style="List Bullet")

doc.add_paragraph("Step one", style="List Number")
doc.add_paragraph("Step two", style="List Number")
```

For nested lists, use `List Bullet 2` / `List Number 2` (indented one level), `List Bullet 3` / `List Number 3` for deeper nesting.

### Page Breaks

```python
doc.add_page_break()
```

---

## Page Layout

Every document has at least one section. Control paper size, margins, and orientation through the section object.

```python
from docx.shared import Inches
from docx.enum.section import WD_ORIENT

section = doc.sections[0]

# US Letter
section.page_width = Inches(8.5)
section.page_height = Inches(11)

# Margins
section.top_margin = Inches(1)
section.bottom_margin = Inches(1)
section.left_margin = Inches(1.25)
section.right_margin = Inches(1.25)
```

### Landscape Pages

To switch orientation mid-document, add a new section:

```python
landscape = doc.add_section(start_type=2)  # starts on a new page
landscape.orientation = WD_ORIENT.LANDSCAPE
landscape.page_width = Inches(11)
landscape.page_height = Inches(8.5)
```

Note that when switching to landscape you must also swap `page_width` and `page_height` manually — python-docx does not do this automatically.

---

## Headers and Footers

```python
section = doc.sections[0]

# Header
hdr = section.header
hdr.paragraphs[0].text = "CONFIDENTIAL"
hdr.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT

# Footer
ftr = section.footer
ftr.paragraphs[0].text = "Acme Corp — Internal Use Only"
ftr.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
```

### Dynamic Page Numbers in Footers

Page number fields require inserting a field code via the underlying XML:

```python
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

fp = section.footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER

# Begin field
run1 = fp.add_run()
begin = OxmlElement("w:fldChar")
begin.set(qn("w:fldCharType"), "begin")
run1._r.append(begin)

# Field instruction
run2 = fp.add_run()
instr = OxmlElement("w:instrText")
instr.set(qn("xml:space"), "preserve")
instr.text = " PAGE "
run2._r.append(instr)

# End field
run3 = fp.add_run()
end = OxmlElement("w:fldChar")
end.set(qn("w:fldCharType"), "end")
run3._r.append(end)
```

---

## Tables

### Building a Table

```python
from docx.shared import Inches, Pt, RGBColor
from docx.oxml.ns import nsdecls
from docx.oxml import parse_xml

table = doc.add_table(rows=4, cols=3)
table.style = "Table Grid"

# Populate the header row
for i, label in enumerate(["Region", "Q1 Revenue", "Q2 Revenue"]):
    cell = table.cell(0, i)
    cell.text = label

    # Shade the header
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="2F5496" w:val="clear"/>')
    cell._tc.get_or_add_tcPr().append(shading)

    # White bold text
    for run in cell.paragraphs[0].runs:
        run.font.color.rgb = RGBColor(255, 255, 255)
        run.font.bold = True

# Fill data rows
rows_data = [
    ("North", "$142K", "$158K"),
    ("South", "$98K",  "$107K"),
    ("West",  "$175K", "$189K"),
]
for row_idx, (region, q1, q2) in enumerate(rows_data, start=1):
    table.cell(row_idx, 0).text = region
    table.cell(row_idx, 1).text = q1
    table.cell(row_idx, 2).text = q2
```

### Setting Column Widths

```python
for cell in table.columns[0].cells:
    cell.width = Inches(1.5)
for cell in table.columns[1].cells:
    cell.width = Inches(2)
for cell in table.columns[2].cells:
    cell.width = Inches(2)
```

### Merging Cells

```python
# Merge the first row across all columns for a banner header
top_left = table.cell(0, 0)
top_right = table.cell(0, 2)
top_left.merge(top_right)
top_left.text = "Revenue Summary"
```

### Alternating Row Shading

```python
for idx, row in enumerate(table.rows):
    if idx == 0:
        continue  # skip header
    if idx % 2 == 0:
        for cell in row.cells:
            shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F2F2F2" w:val="clear"/>')
            cell._tc.get_or_add_tcPr().append(shading)
```

**Important**: always use `w:val="clear"` in shading elements — using `"solid"` produces black backgrounds in some renderers.

---

## Text Formatting

### Paragraph-Level Controls

```python
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING

pf = paragraph.paragraph_format
pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
pf.space_before = Pt(6)
pf.space_after = Pt(12)
pf.first_line_indent = Inches(0.5)
```

### Character-Level Controls (Runs)

A paragraph is made up of one or more "runs" — contiguous spans sharing the same formatting.

```python
run = paragraph.add_run("Important note")
run.font.name = "Arial"
run.font.size = Pt(12)
run.font.bold = True
run.font.italic = True
run.font.underline = True
run.font.color.rgb = RGBColor(200, 0, 0)
```

### Applying Styles

Prefer styles over direct formatting when you want consistency across the document:

```python
doc.add_paragraph("A quotation from the report.", style="Quote")
doc.add_paragraph("Key takeaway here.", style="Intense Quote")

# Customize the Normal style globally
normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(11)
normal.paragraph_format.space_after = Pt(8)
```

---

## Images

```python
# Full-width image
doc.add_picture("chart.png", width=Inches(6))

# Inline image within a paragraph
p = doc.add_paragraph()
r = p.add_run()
r.add_picture("icon.png", width=Inches(0.5))
p.add_run("  Caption text next to the icon.")
```

If you only know the width, python-docx preserves the aspect ratio automatically. Specify both `width` and `height` only when you need to force exact dimensions.

---

## Extracting Content

### All Body Text

```python
doc = Document("report.docx")
full_text = "\\n".join(p.text for p in doc.paragraphs)
```

### Text from Tables

```python
for table in doc.tables:
    for row in table.rows:
        cells = [cell.text for cell in row.cells]
        print(" | ".join(cells))
```

### Combined Extraction (Paragraphs + Tables in Document Order)

The methods above miss the interleaving order. To walk the document body in sequence:

```python
from docx.oxml.ns import qn

body = doc.element.body
for child in body.iterchildren():
    tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
    if tag == "p":
        text = child.text or ""
        for sub in child.iter():
            if sub.text:
                text += sub.text
        # This is a paragraph
    elif tag == "tbl":
        # This is a table — process rows/cells
        pass
```

---

## Modifying Existing Documents

### Open, Edit, Save

```python
doc = Document("contract.docx")

# Change a specific paragraph
doc.paragraphs[2].text = "Updated clause language."

# Always save to a new path to preserve the original
doc.save("contract_v2.docx")
```

**Warning**: setting `paragraph.text` directly replaces the entire paragraph content and **strips all run-level formatting**. To preserve formatting while changing words, operate on individual runs instead.

### Preserving Formatting During Edits

```python
for para in doc.paragraphs:
    for run in para.runs:
        if "PLACEHOLDER" in run.text:
            run.text = run.text.replace("PLACEHOLDER", "Acme Corporation")
            # Font, bold, italic, color all remain intact
```

### Inserting and Removing Structural Elements

```python
# python-docx doesn't have a direct "delete paragraph" method.
# Remove a paragraph by deleting its underlying XML element:
unwanted = doc.paragraphs[5]
unwanted._element.getparent().remove(unwanted._element)
```

---

## Table of Contents

python-docx can insert a TOC field code that Word / Google Docs will populate when the file is opened. The TOC won't have visible entries until the user updates fields in their application (usually prompted automatically).

```python
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

p = doc.add_paragraph()

# Begin field
run_begin = p.add_run()
fld_begin = OxmlElement("w:fldChar")
fld_begin.set(qn("w:fldCharType"), "begin")
run_begin._r.append(fld_begin)

# TOC instruction
run_instr = p.add_run()
instr = OxmlElement("w:instrText")
instr.set(qn("xml:space"), "preserve")
instr.text = ' TOC \\\\o "1-3" \\\\h \\\\z \\\\u '
run_instr._r.append(instr)

# End field
run_end = p.add_run()
fld_end = OxmlElement("w:fldChar")
fld_end.set(qn("w:fldCharType"), "end")
run_end._r.append(fld_end)
```

Place this before your first heading. The `\\\\o "1-3"` flag tells Word to include heading levels 1 through 3.

---

## Hyperlinks

python-docx doesn't have a built-in hyperlink API. Add them via the relationship and XML layers:

```python
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

def insert_hyperlink(paragraph, url, display_text):
    part = paragraph.part
    rel_id = part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)

    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), rel_id)

    run = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    rStyle = OxmlElement("w:rStyle")
    rStyle.set(qn("w:val"), "Hyperlink")
    rPr.append(rStyle)
    run.append(rPr)

    text = OxmlElement("w:t")
    text.text = display_text
    run.append(text)

    hyperlink.append(run)
    paragraph._p.append(hyperlink)

# Usage
p = doc.add_paragraph("Visit our website: ")
insert_hyperlink(p, "https://example.com", "example.com")
```

---

## Markdown to Docx

The `md2docx` module converts markdown-formatted text into properly styled Word elements. Use it when you want to write content as markdown and have it rendered with appropriate Word formatting:

```python
from md2docx import add_markdown, add_styled_markdown, setup_document

doc = setup_document()  # pre-configured with 1-inch margins, Arial 11pt

add_markdown(doc, '''
# Section Title

Our company has **extensive experience** in providing services
across *multiple sectors* including:

- Government facilities
- Commercial properties

| Certification | Status |
|---------------|--------|
| CPR/First Aid | Current |

> The contractor shall provide 24/7 coverage.
''')

# Confidence-based coloring for proposals
add_styled_markdown(doc, "We fully meet this requirement.", confidence="high")
add_styled_markdown(doc, "We likely can meet this.", confidence="medium")
add_styled_markdown(doc, "[INSERT: specific value]", confidence="placeholder")

doc.save("output.docx")
```

---

## Gotchas and Tips

### Things That Silently Break

- **Setting `.text` on a paragraph** wipes all runs and their formatting. Edit individual runs instead.
- **Opening with the wrong constructor**: `Document("file.docx")` opens an existing file; `Document()` with no arguments creates a blank one. Passing a nonexistent path raises an error.
- **Saving over the input file** while it's still being read can corrupt it. Always save to a separate path, then rename if needed.
- **Table cell shading**: use `w:val="clear"` in shading XML, never `"solid"` — the latter produces black fills in some viewers.

### Limitations of python-docx

- **No formula evaluation**: field codes (TOC, page numbers, cross-references) are written as instructions; the actual values only appear when the file is opened in Word or Google Docs.
- **Tracked changes**: read-only access through the underlying XML (`lxml`). python-docx can't natively add or accept tracked changes.
- **Comments**: similarly require XML-level manipulation to add or read.
- **Macros / VBA**: not accessible. `.docm` files can be opened but macros are ignored.
- **Complex layout**: floating text boxes, SmartArt, and embedded charts are preserved if already present, but can't be created from scratch.

### Sandbox Compatibility

This guide uses only `python-docx` and its dependency `lxml`. No subprocess calls, no LibreOffice, no Node.js, no external binaries. Everything runs in a pure-Python sandbox. The only filesystem operations are reading input `.docx` files and writing output ones.

---

## Code Style

Keep Python code tight — short variable names, no commentary that restates the obvious, no unnecessary print statements. The document is the deliverable, not the script.

Inside the document itself, use styles consistently, keep fonts uniform, and label sections with clear headings so the reader can navigate without guessing.""",
    packages=["python-docx"],
    modules={"md2docx": str(_MODULES_DIR / "md2docx.py")},
)
