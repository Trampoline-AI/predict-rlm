"""Run the contract comparison example.

Drop two or more PDF contracts into the `sample/input/` directory, then run:

    uv run examples/contract_comparison/run.py
    uv run examples/contract_comparison/run.py --debug

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

# Add examples/ to path so we can import the contract_comparison package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contract_comparison import ContractComparator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SOURCE_DIR = Path(__file__).parent / "sample" / "input"
LLM_MODEL = "openai/gpt-5.4"
SUB_LM_MODEL = "openai/gpt-5.1"


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
    parser = argparse.ArgumentParser(description="Compare PDF contracts with an RLM")
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
        help="PDF files or directories to compare (default: sample/input/)",
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

    if len(pdfs) < 2:
        print(f"Need at least 2 PDF files to compare (found {len(pdfs)})")
        print(f"Drop PDFs into {SOURCE_DIR.resolve()}, or pass file paths as arguments.")
        return

    print(f"Comparing {len(pdfs)} contract(s):")
    for p in pdfs:
        print(f"  - {p.name}")
    print()

    model_config = get_model_config(args.model)

    # Set up the LLMs
    lm = dspy.LM(**model_config, cache=False)
    sub_lm = dspy.LM(args.sub_lm_model, cache=False)

    # Build File references and count pages for stats
    import pymupdf

    contracts = [File(path=str(p.resolve())) for p in pdfs]
    total_pages = 0
    for c in contracts:
        with pymupdf.open(c.path) as pdf:
            pages = len(pdf)
        total_pages += pages
        print(f"  {Path(c.path).name}  ({pages} pages)")
    print()

    # Run the comparator
    print("Comparing contracts...")
    print("-" * 60)

    comparator = ContractComparator(
        sub_lm=sub_lm,
        max_iterations=args.max_iterations,
        verbose=True,
        debug=args.debug,
    )
    start_time = time.perf_counter()
    with dspy.context(lm=lm):
        result = await comparator.aforward(contracts=contracts)
    run_duration = time.perf_counter() - start_time

    # Print results
    print()
    print("=" * 60)
    print("COMPARISON REPORT")
    print("=" * 60)
    print(result.report)
    print()

    print("KEY DIFFERENCES")
    print("-" * 40)
    for diff in result.key_differences:
        print(f"  [{diff.area}] {diff.description}")
        print(f"    Impact: {diff.impact}")
        print()

    print("SECTION DIFFS")
    print("-" * 40)
    for sd in result.section_diffs:
        print(f"  {sd.section_name} [{sd.significance}]")
        if sd.significance != "identical":
            print(f"    {sd.difference_summary}")
        print()

    print("EXECUTIVE SUMMARY")
    print("-" * 40)
    print(result.summary)
    print()

    # Save report to output
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(__file__).parent / "output" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "comparison-report.md"
    report_path.write_text(result.report)
    print(f"Report saved to: {report_path}")
    print()

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

    print("=" * 60)
    print("RUN STATS")
    print("=" * 60)
    print(f"Main LM:    {args.model}")
    print(f"Sub-LM:     {args.sub_lm_model}")
    print(f"Contracts:  {len(pdfs)} ({total_pages} pages)")
    print(f"Duration:   {mins}m {secs}s")
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
