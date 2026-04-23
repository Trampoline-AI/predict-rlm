"""Run the event crawler example.

Analyze an event page to produce a markdown executive summary
of speakers, attendees, schedule, and more.

    uv run examples/event_crawler/run.py --url https://lu.ma/some-event
    uv run examples/event_crawler/run.py --url https://example.com/conf --debug

Environment:
    Set OPENAI_API_KEY (or whatever LLM provider you configure below).
    Set PERPLEXITY_API_KEY for speaker/attendee enrichment via Perplexity search.
"""

import argparse
import asyncio
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import dspy

# Add examples/ to path so we can import the event_crawler package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from event_crawler import EventCrawler, EventOntology

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LLM_MODEL = "openai/gpt-5.4"
SUB_LM_MODEL = "openai/gpt-5.1"


def get_model_config(model: str):
    return dict(model=model, num_retries=5, reasoning_effort="low")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crawl an event page and produce an executive summary"
    )
    parser.add_argument(
        "--url",
        required=True,
        help="URL of the event page to crawl",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Entity categories to extract (default: speakers attendees schedule sponsors organizers)",
    )
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
        default=40,
        help="Maximum REPL iterations (default: 40)",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    print(f"Event URL: {args.url}")

    ontology = EventOntology()
    if args.categories:
        ontology = EventOntology(categories=args.categories)
    print(f"Categories: {', '.join(ontology.categories)}")
    print()

    model_config = get_model_config(args.model)
    lm = dspy.LM(**model_config, cache=False)
    sub_lm = dspy.LM(args.sub_lm_model, cache=False, reasoning_effort="low")

    print("Crawling event...")
    print("-" * 60)

    crawler = EventCrawler(
        sub_lm=sub_lm,
        max_iterations=args.max_iterations,
        verbose=True,
        debug=args.debug,
    )
    start_time = time.perf_counter()
    with dspy.context(lm=lm):
        prediction = await crawler.aforward(
            url=args.url,
            ontology=ontology,
        )
    run_duration = time.perf_counter() - start_time

    # Save report to output dir
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(__file__).parent / "output" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    if prediction.report and prediction.report.path:
        src = Path(prediction.report.path)
        dest = output_dir / "summary.md"
        shutil.copy2(src, dest)
        print(f"\nReport saved to: {dest}")

        print()
        print("=" * 60)
        print("EXECUTIVE SUMMARY")
        print("=" * 60)
        print(dest.read_text())
    else:
        print("\nWarning: No report file was generated.")

    # Run stats
    lm_history = list(lm.history)
    sub_lm_history = list(sub_lm.history)

    lm_cost = sum(e.get("cost", 0) or 0 for e in lm_history)
    lm_input = sum(
        e.get("usage", {}).get("prompt_tokens", 0) or 0 for e in lm_history
    )
    lm_output = sum(
        e.get("usage", {}).get("completion_tokens", 0) or 0 for e in lm_history
    )

    sub_lm_cost = sum(e.get("cost", 0) or 0 for e in sub_lm_history)
    sub_lm_input = sum(
        e.get("usage", {}).get("prompt_tokens", 0) or 0 for e in sub_lm_history
    )
    sub_lm_output = sum(
        e.get("usage", {}).get("completion_tokens", 0) or 0
        for e in sub_lm_history
    )

    total_cost = lm_cost + sub_lm_cost
    mins, secs = divmod(int(run_duration), 60)

    print()
    print("=" * 60)
    print("RUN STATS")
    print("=" * 60)
    print(f"Main LM:    {args.model}")
    print(f"Sub-LM:     {args.sub_lm_model}")
    print(f"Event URL:  {args.url}")
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
    print(f"Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
