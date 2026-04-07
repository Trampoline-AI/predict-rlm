"""Convert markdown-formatted text into python-docx elements.

Mount this module into the RLM sandbox so the RLM can write content
as markdown and have it rendered with proper Word formatting.

Usage inside the sandbox::

    from md2docx import add_markdown, add_styled_markdown
    from docx import Document

    doc = Document()
    add_markdown(doc, '''
    # Executive Summary

    Our company has **extensive experience** in providing security
    services across *multiple sectors* including:

    - Government facilities
    - Commercial properties
    - Educational institutions

    ## Key Qualifications

    | Certification | Status |
    |---------------|--------|
    | CPR/First Aid | Current |
    | Armed Guard   | Current |

    > The contractor shall provide 24/7 security coverage.

    We fully comply with this requirement. Our team operates
    **round-the-clock shifts** with a minimum of 3 guards per shift.
    ''')

    # With confidence-based coloring:
    add_styled_markdown(doc, "We have done this exact work.", confidence="high")
    add_styled_markdown(doc, "We likely can meet this.", confidence="medium")
    add_styled_markdown(doc, "[INSERT: specific contract value]", confidence="placeholder")
"""

import re

from docx import Document as _Document
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
from docx.shared import Inches, Pt, RGBColor

# ---------------------------------------------------------------------------
# Color constants
# ---------------------------------------------------------------------------

COLOR_BLACK = RGBColor(0x00, 0x00, 0x00)
COLOR_ORANGE = RGBColor(0xE6, 0x7E, 0x22)
COLOR_RED = RGBColor(0xE7, 0x4C, 0x3C)
COLOR_GRAY = RGBColor(0x99, 0x99, 0x99)

CONFIDENCE_COLORS = {
    "high": COLOR_BLACK,
    "medium": COLOR_ORANGE,
    "low": COLOR_ORANGE,
    "placeholder": COLOR_RED,
    "source": COLOR_GRAY,
}


# ---------------------------------------------------------------------------
# Inline formatting parser
# ---------------------------------------------------------------------------

_INLINE_RE = re.compile(
    r"(\*\*\*(.+?)\*\*\*"   # ***bold italic***
    r"|\*\*(.+?)\*\*"        # **bold**
    r"|\*(.+?)\*"            # *italic*
    r"|`(.+?)`"              # `code`
    r")"
)


def _add_inline_runs(paragraph, text, base_color=None, base_italic=False):
    """Parse inline markdown and add runs with appropriate formatting."""
    pos = 0
    for m in _INLINE_RE.finditer(text):
        # Add text before this match
        before = text[pos:m.start()]
        if before:
            run = paragraph.add_run(before)
            if base_color:
                run.font.color.rgb = base_color
            if base_italic:
                run.italic = True

        # Determine which group matched
        if m.group(2):  # ***bold italic***
            run = paragraph.add_run(m.group(2))
            run.bold = True
            run.italic = True
        elif m.group(3):  # **bold**
            run = paragraph.add_run(m.group(3))
            run.bold = True
        elif m.group(4):  # *italic*
            run = paragraph.add_run(m.group(4))
            run.italic = True
        elif m.group(5):  # `code`
            run = paragraph.add_run(m.group(5))
            run.font.name = "Courier New"
            run.font.size = Pt(10)

        if base_color:
            run.font.color.rgb = base_color

        pos = m.end()

    # Add remaining text
    remaining = text[pos:]
    if remaining:
        run = paragraph.add_run(remaining)
        if base_color:
            run.font.color.rgb = base_color
        if base_italic:
            run.italic = True


# ---------------------------------------------------------------------------
# Table parser
# ---------------------------------------------------------------------------

def _parse_table_lines(lines):
    """Parse markdown table lines into header and rows."""
    rows = []
    for line in lines:
        line = line.strip()
        if line.startswith("|"):
            line = line[1:]
        if line.endswith("|"):
            line = line[:-1]
        cells = [c.strip() for c in line.split("|")]
        rows.append(cells)

    if len(rows) < 2:
        return None, []

    header = rows[0]
    # Skip separator row (contains dashes)
    data = []
    for row in rows[1:]:
        if all(set(c.strip()) <= {"-", ":", " "} for c in row):
            continue
        data.append(row)

    return header, data


def _add_table(doc, header, data_rows):
    """Add a formatted table to the document."""
    num_cols = len(header)
    table = doc.add_table(rows=1 + len(data_rows), cols=num_cols)
    table.style = "Table Grid"

    # Header row
    for i, text in enumerate(header):
        cell = table.cell(0, i)
        cell.text = text
        shading = parse_xml(
            f'<w:shd {nsdecls("w")} w:fill="2F5496" w:val="clear"/>'
        )
        cell._tc.get_or_add_tcPr().append(shading)
        for run in cell.paragraphs[0].runs:
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            run.font.bold = True

    # Data rows
    for row_idx, row_data in enumerate(data_rows):
        for col_idx, text in enumerate(row_data):
            if col_idx < num_cols:
                table.cell(row_idx + 1, col_idx).text = text
        # Alternating shading
        if row_idx % 2 == 0:
            for col_idx in range(num_cols):
                cell = table.cell(row_idx + 1, col_idx)
                shading = parse_xml(
                    f'<w:shd {nsdecls("w")} w:fill="F2F2F2" w:val="clear"/>'
                )
                cell._tc.get_or_add_tcPr().append(shading)

    doc.add_paragraph()  # spacing after table


# ---------------------------------------------------------------------------
# Block-level markdown parser
# ---------------------------------------------------------------------------

def add_markdown(doc, text, default_color=None):
    """Parse markdown text and add it to a python-docx Document.

    Supports: headings (#), bold, italic, code spans, bullet lists,
    numbered lists, blockquotes, tables, horizontal rules, and
    paragraphs with inline formatting.

    Parameters
    ----------
    doc : Document
        The python-docx Document to add content to.
    text : str
        Markdown-formatted text.
    default_color : RGBColor, optional
        Default text color for all content.
    """
    lines = text.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            i += 1
            continue

        # Horizontal rule
        if re.match(r"^(-{3,}|\*{3,}|_{3,})$", stripped):
            # Add a thin line via a paragraph with bottom border
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(6)
            i += 1
            continue

        # Headings
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            # Remove any trailing # marks
            heading_text = re.sub(r"\s*#+\s*$", "", heading_text)
            doc.add_heading(heading_text, level=min(level, 4))
            i += 1
            continue

        # Table (starts with |)
        if stripped.startswith("|") and i + 1 < len(lines):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            if len(table_lines) >= 2:
                header, data = _parse_table_lines(table_lines)
                if header:
                    _add_table(doc, header, data)
            continue

        # Blockquote
        if stripped.startswith(">"):
            quote_text = stripped[1:].strip()
            # Gather continuation lines
            i += 1
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote_text += " " + lines[i].strip()[1:].strip()
                i += 1
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.5)
            _add_inline_runs(p, quote_text, base_color=COLOR_GRAY, base_italic=True)
            continue

        # Bullet list
        bullet_match = re.match(r"^[-*+]\s+(.+)$", stripped)
        if bullet_match:
            p = doc.add_paragraph(style="List Bullet")
            _add_inline_runs(p, bullet_match.group(1), base_color=default_color)
            i += 1
            continue

        # Numbered list
        num_match = re.match(r"^\d+[.)]\s+(.+)$", stripped)
        if num_match:
            p = doc.add_paragraph(style="List Number")
            _add_inline_runs(p, num_match.group(1), base_color=default_color)
            i += 1
            continue

        # Regular paragraph — gather consecutive non-empty, non-special lines
        para_lines = [stripped]
        i += 1
        while i < len(lines):
            next_stripped = lines[i].strip()
            if not next_stripped:
                break
            if re.match(r"^(#{1,6})\s+", next_stripped):
                break
            if next_stripped.startswith("|"):
                break
            if next_stripped.startswith(">"):
                break
            if re.match(r"^[-*+]\s+", next_stripped):
                break
            if re.match(r"^\d+[.)]\s+", next_stripped):
                break
            if re.match(r"^(-{3,}|\*{3,}|_{3,})$", next_stripped):
                break
            para_lines.append(next_stripped)
            i += 1

        para_text = " ".join(para_lines)
        p = doc.add_paragraph()
        _add_inline_runs(p, para_text, base_color=default_color)


def add_styled_markdown(doc, text, confidence="high"):
    """Add markdown content with confidence-based color styling.

    Parameters
    ----------
    doc : Document
        The python-docx Document.
    text : str
        Markdown-formatted text.
    confidence : str
        One of "high", "medium", "low", "placeholder", "source".
        Controls the text color:
        - high: black (normal)
        - medium/low: orange (needs review)
        - placeholder: red italic (user must fill in)
        - source: gray italic 9pt (source annotation)
    """
    color = CONFIDENCE_COLORS.get(confidence, COLOR_BLACK)
    add_markdown(doc, text, default_color=color if confidence != "high" else None)


def setup_document(doc=None):
    """Create or configure a Document with standard proposal formatting.

    Returns a Document with 1-inch margins, Arial 11pt body text,
    and configured heading styles.
    """
    if doc is None:
        doc = _Document()

    # Page setup
    section = doc.sections[0]
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)

    # Default font
    style = doc.styles["Normal"]
    style.font.name = "Arial"
    style.font.size = Pt(11)
    style.paragraph_format.space_before = Pt(3)
    style.paragraph_format.space_after = Pt(6)

    return doc
