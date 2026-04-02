import dspy

from predict_rlm import File

from .schema import ComparisonResult


class CompareContracts(dspy.Signature):
    """Compare PDF contracts and produce a structured report of differences.

    1. **Survey both documents** — file names, page counts, structure,
       and section headings. Print a summary of each document's layout.

    2. **Read each document** by rendering pages as images and using
       predict() to extract section content. Process pages in parallel
       with asyncio.gather().

    3. **Identify corresponding sections** between the documents and
       compare them systematically. Note sections that are new, removed,
       or modified.

    4. **Analyze differences** — for each section, classify the change
       as major, minor, or identical. Identify key differences in terms,
       dates, obligations, pricing, liability, and other critical areas.

    5. **Produce the result** with a full markdown comparison report,
       per-section diffs, key differences with impact analysis, and an
       executive summary.
    """

    contracts: list[File] = dspy.InputField(
        desc="PDF contract files to compare"
    )
    result: ComparisonResult = dspy.OutputField(
        desc="Structured comparison with per-section diffs and key differences"
    )
