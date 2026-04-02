"""Run the document analysis example.

Drop PDF files into the `sample/input/` directory next to this script, then run:

    uv run examples/document_analysis/run.py
    uv run examples/document_analysis/run.py --debug

Requires:
    pip install 'predict-rlm[examples]'   # for PDF rendering via pymupdf

Environment:
    Set OPENAI_API_KEY (or whatever LLM provider you configure below).
"""

import argparse
import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path

import dspy

from predict_rlm import File

# Add examples/ to path so we can import the document_analysis package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from document_analysis import DocumentAnalyzer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SOURCE_DIR = Path(__file__).parent / "sample" / "input"
LLM_MODEL = "openai/gpt-5.4"
SUB_LM_MODEL = "openai/gpt-5.1"
CRITERIA = """
Analyze the document(s) and produce a comprehensive briefing report
structured as follows.

    ---

    **Formatting guidelines:**

The report should be professional, easy to read, and visually elegant.
Mix prose, tables, bullets, and numbered items to draw the topology of
the information and help the reader quickly parse the report. Do not
include page references — present information directly. Use bold
sparingly. Favor prose over bullets; use bullets very sparingly.

    ---

    **Report sections:**

    1.  **Executive Summary**
    What is this document about? Who are the key parties involved?
    What are the most important facts, decisions, or actions described?

2.  **Key Dates and Timeline**
    All significant dates mentioned in the document: deadlines,
    effective dates, milestones, meetings, expiration dates. Present
    in chronological order. Flag any unusually tight timelines.

3.  **Key Entities and Stakeholders**
    People, organizations, and roles mentioned in the document.
    For each, note their role and relevance. Include contact
    information where available.

4.  **Financial Information**
    Any monetary amounts, fees, budgets, pricing structures,
    payment terms, or financial obligations mentioned. Summarize
    in a table if multiple items exist.
""".strip()


def get_model_config(model: str):
    if model == "openai/gpt-5.4":
        return dict(
            model=model,
            num_retries=5,
            reasoning_effort="low",
        )
    else:
        return dict(
            model=model,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze documents with an RLM")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print REPL code, output, errors, and tool calls to stderr",
    )
    parser.add_argument(
        "--model",
        default=LLM_MODEL,
        help=f"LLM model to use (default: {LLM_MODEL})",
    )
    parser.add_argument(
        "--sub-lm-model",
        default=SUB_LM_MODEL,
        help=f"Sub-LM model to use (default: {SUB_LM_MODEL})",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=30,
        help="Maximum REPL iterations (default: 30)",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="PDF files or directories to analyze (default: sample/input/)",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    # Discover files — accept file paths or directories
    if args.files:
        pdfs = []
        for f in args.files:
            p = Path(f)
            if not p.exists():
                print(f"File not found: {p}")
                return
            if p.is_dir():
                pdfs.extend(sorted(p.glob("*.pdf")))
            else:
                pdfs.append(p)
    else:
        pdfs = sorted(SOURCE_DIR.glob("*.pdf"))

    if not pdfs:
        print(f"No PDF files found in {SOURCE_DIR.resolve()}")
        print("Drop some PDFs there, or pass file paths as arguments.")
        return

    print(f"Found {len(pdfs)} file(s):")
    for p in pdfs:
        print(f"  - {p.name}")
    print()

    model_config = get_model_config(args.model)

    # Set up the LLMs
    lm = dspy.LM(**model_config, cache=False)
    sub_lm = dspy.LM(args.sub_lm_model, cache=False)

    # Build File references and count pages for stats
    import pymupdf

    documents = [File(path=str(p.resolve())) for p in pdfs]
    total_pages = 0
    for doc in documents:
        with pymupdf.open(doc.path) as pdf:
            pages = len(pdf)
        total_pages += pages
        print(f"  {Path(doc.path).name}  ({pages} pages)")
    print()

    # Run the analyzer
    print("Analyzing documents...")
    print("-" * 60)

    analyzer = DocumentAnalyzer(
        sub_lm=sub_lm,
        max_iterations=args.max_iterations,
        verbose=True,
        debug=args.debug,
    )
    start_time = time.perf_counter()
    with dspy.context(lm=lm):
        result = await analyzer.aforward(documents=documents, criteria=CRITERIA)
    run_duration = time.perf_counter() - start_time

    # Save report to output dir
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(__file__).parent / "output" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.md"
    report_path.write_text(result.report)
    print(f"Report saved to: {report_path}")

    # Print results
    print()
    print("=" * 60)
    print("REPORT")
    print("=" * 60)
    print(result.report)

    if result.key_dates:
        print()
        print("KEY DATES")
        print("-" * 40)
        for d in result.key_dates:
            time_str = f" at {d.time}" if d.time else ""
            tz_str = f" {d.timezone}" if d.timezone else ""
            print(f"  {d.date}{time_str}{tz_str} — {d.name}")

    if result.key_entities:
        print()
        print("KEY ENTITIES")
        print("-" * 40)
        for e in result.key_entities:
            role = f" ({e.role})" if e.role else ""
            contact = f" — {e.contact}" if e.contact else ""
            print(f"  {e.name}{role}{contact}")

    # Run stats
    lm_history = list(lm.history)
    sub_lm_history = list(sub_lm.history)

    lm_cost = sum(e.get("cost", 0) or 0 for e in lm_history)
    lm_input = sum(e.get("usage", {}).get("prompt_tokens", 0) or 0 for e in lm_history)
    lm_output = sum(e.get("usage", {}).get("completion_tokens", 0) or 0 for e in lm_history)

    sub_lm_cost = sum(e.get("cost", 0) or 0 for e in sub_lm_history)
    sub_lm_input = sum(e.get("usage", {}).get("prompt_tokens", 0) or 0 for e in sub_lm_history)
    sub_lm_output = sum(
        e.get("usage", {}).get("completion_tokens", 0) or 0 for e in sub_lm_history
    )

    total_cost = lm_cost + sub_lm_cost
    mins, secs = divmod(int(run_duration), 60)

    print()
    print("=" * 60)
    print("RUN STATS")
    print("=" * 60)
    print(f"Main LM:   {args.model}")
    print(f"Sub-LM:    {args.sub_lm_model}")
    print(f"Documents: {len(pdfs)} ({total_pages} pages)")
    print(f"Duration:  {mins}m {secs}s")
    print()
    print(f"Main LM ({len(lm_history)} calls):")
    print(f"  Input:  {lm_input:,} tokens")
    print(f"  Output: {lm_output:,} tokens")
    print(f"  Cost:   ${lm_cost:.4f}")
    print()
    print(f"Sub-LM ({len(sub_lm_history)} calls):")
    print(f"  Input:  {sub_lm_input:,} tokens")
    print(f"  Output: {sub_lm_output:,} tokens")
    print(f"  Cost:   ${sub_lm_cost:.4f}")
    print()
    print(f"Total cost: ${total_cost:.4f} (${total_cost / max(total_pages, 1):.4f}/page)")


if __name__ == "__main__":
    asyncio.run(main())
