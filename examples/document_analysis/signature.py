import dspy

from predict_rlm import File

from .schema import DocumentAnalysis


class AnalyzeDocuments(dspy.Signature):
    """Analyze documents and produce a structured report.

    1. **Read the report criteria** (appended below) to understand what
       information to extract and in what format. The criteria define the
       structure and sections of the report.

    2. **Survey the documents** to understand what you're working with:
       file names, page counts, document types (main document,
       attachments, amendments, etc.).

    3. **Gather information** systematically by rendering pages as images
       and using predict() to extract relevant content. Work through the
       criteria's requested sections, locating the relevant information
       in the documents.

    4. **Produce the report** following the format specified in the criteria.
       Use tables for structured data, prose for analysis and context.
    """

    documents: list[File] = dspy.InputField(desc="PDF documents to analyze")
    analysis: DocumentAnalysis = dspy.OutputField(
        desc="Structured analysis with markdown report, key dates, and key entities"
    )
    docx_report: File = dspy.OutputField(desc="The report in Docx format")
