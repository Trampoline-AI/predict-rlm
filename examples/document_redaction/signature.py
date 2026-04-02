import dspy

from predict_rlm import File

from .schema import RedactionResult


class RedactDocuments(dspy.Signature):
    """Redact sensitive information from documents based on criteria.

    1. **Read the redaction criteria** (appended below) to understand what
       types of information must be redacted.

    2. **Survey the documents** — file names, page counts, document types.

    3. **Inspect each page** visually and identify all text matching the
       redaction criteria.

    4. **Apply redactions** for text and non-text elements like signatures
       or logos. If a text match fails, try a shorter or different substring.

    5. **Verify the result** by re-rendering redacted pages and confirming
       sensitive content is gone.

    6. **Save the redacted PDFs** to the output directory and **produce the
       result** with counts, per-page summaries, and targets.
    """

    documents: list[File] = dspy.InputField(
        desc="PDF documents to redact"
    )
    redacted_documents: list[File] = dspy.OutputField(
        desc="Redacted PDF files"
    )
    result: RedactionResult = dspy.OutputField(
        desc="Redaction result with counts and per-page summaries"
    )
