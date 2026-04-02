"""Run the document redaction example.

    uv run examples/document_redaction/run.py
    uv run examples/document_redaction/run.py --debug
    uv run examples/document_redaction/run.py /path/to/docs/

Requires:
    pip install 'predict-rlm[examples]'   # for PDF rendering via pymupdf

Environment:
    Set OPENAI_API_KEY (or whatever LLM provider you configure below).
"""

import argparse
import asyncio
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import dspy

from predict_rlm import File

# Add examples/ to path so we can import the document_redaction package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from document_redaction import DocumentRedactor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SOURCE_DIR = Path(__file__).parent / "sample" / "input"
LLM_MODEL = "openai/gpt-5.4"
SUB_LM_MODEL = "openai/gpt-5.1"
CRITERIA = """
Redact all personally identifiable information (PII), including:

1. **Names** — Full names of individuals (not company or organization names)
2. **Contact info** — Phone numbers, email addresses, fax numbers
3. **Addresses** — Street addresses, P.O. boxes (not city/state/country)
4. **Government IDs** — Social security numbers, tax IDs, passport numbers
5. **Financial info** — Bank account numbers, credit card numbers, routing numbers
6. **Signatures** — Handwritten signatures (redact the bounding area)

Added to this, redact any dates found in the document, in any format.
""".strip()


def get_model_config(model: str):
    if model == "openai/gpt-5.4":
        return dict(
            model=model,
            num_retries=5,
            reasoning_effort="none",
        )
    else:
        return dict(
            model=model,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Redact documents with an RLM")
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
        "--criteria",
        default=CRITERIA,
        help="Redaction criteria (default: PII redaction)",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="PDF files or directories to redact (default: sample/input/)",
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

    # Run the redactor
    print("Redacting documents...")
    print("-" * 60)

    redactor = DocumentRedactor(
        sub_lm=sub_lm,
        max_iterations=args.max_iterations,
        verbose=True,
        debug=args.debug,
    )
    start_time = time.perf_counter()
    with dspy.context(lm=lm):
        prediction = await redactor.aforward(documents=documents, criteria=args.criteria)
    run_duration = time.perf_counter() - start_time

    result = prediction.result

    # Copy redacted PDFs to output dir
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(__file__).parent / "output" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    for f in prediction.redacted_documents:
        if f.path:
            src = Path(f.path)
            dest = output_dir / src.name
            shutil.copy2(src, dest)
            print(f"Output file: {dest}")

    # Print results
    print()
    print("=" * 60)
    print("REDACTION RESULTS")
    print("=" * 60)
    print(f"Total redactions applied: {result.total_redactions}")

    if result.page_summaries:
        print()
        print("PER-PAGE SUMMARY")
        print("-" * 40)
        for ps in result.page_summaries:
            cats = ", ".join(ps.categories) if ps.categories else "none"
            print(f"  Page {ps.page}: {ps.redaction_count} redaction(s) [{cats}]")

    if result.targets:
        print()
        print("REDACTION TARGETS")
        print("-" * 40)
        for t in result.targets:
            print(f'  Page {t.page} [{t.category}]: "{t.text}" — {t.reason}')

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
    print(f"Sub-LM:       {args.sub_lm_model}")
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
