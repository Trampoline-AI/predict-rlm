"""Run the image analysis example.

Pass image files (PNG, JPG, WEBP) and a query:

    uv run examples/image_analysis/run.py --query "What do you see?" photo1.png photo2.jpg
    uv run examples/image_analysis/run.py --query "Compare these images" images/
    uv run examples/image_analysis/run.py --debug --query "Describe each image" *.png

Environment:
    Set OPENAI_API_KEY (or whatever LLM provider you configure below).
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

import dspy

from predict_rlm import File

# Add examples/ to path so we can import the image_analysis package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from image_analysis import ImageAnalyzer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
SOURCE_DIR = Path(__file__).parent / "sample" / "input"
LLM_MODEL = "openai/gpt-5.4"
SUB_LM_MODEL = "openai/gpt-5.1"
DEFAULT_QUERY = """

    What letters appear in each image, and how many times does each letter appear? Always include: logo text, header address/phone/fax, header email, header website URL, "Page N" footers, etc.

    For each image:
    1. Extract the visible text multiple times (at least 2-3 extractions per image)
    2. Compare the extractions - if they differ, extract again until you get consistent results
    3. Only after you have consistent text extraction, count the letters programmatically (case insensitive)

    Use prompts like "Return ONLY the exact text visible, nothing else."
    Do all counting and comparison in Python, not via predict().

    Treat uppercase and lowercase as the same letter (case-insensitive).
    Output the letter statistics in alphabetical order (A-Z)."""


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
    parser = argparse.ArgumentParser(description="Analyze images with a natural language query")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print REPL code, output, errors, and tool calls to stderr",
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help=f"Question about the images (default: {DEFAULT_QUERY!r})",
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
        help="Image files or directories to analyze (default: sample/input/)",
    )
    return parser.parse_args()


def discover_images(paths: list[str] | None) -> list[Path]:
    """Find image files from paths or the default sample directory."""
    if paths:
        images = []
        for f in paths:
            p = Path(f)
            if not p.exists():
                print(f"File not found: {p}")
                sys.exit(1)
            if p.is_dir():
                images.extend(
                    sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
                )
            else:
                images.append(p)
        return images
    return sorted(f for f in SOURCE_DIR.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)


async def main():
    args = parse_args()

    images = discover_images(args.files)
    if not images:
        print(f"No image files found in {SOURCE_DIR.resolve()}")
        print("Pass image file paths as arguments:")
        print("  uv run examples/image_analysis/run.py --query 'What is this?' photo.png")
        return

    print(f"Found {len(images)} image(s):")
    for p in images:
        print(f"  - {p.name}")
    print(f"\nQuery: {args.query}\n")

    model_config = get_model_config(args.model)

    lm = dspy.LM(**model_config, cache=False)
    sub_lm = dspy.LM(args.sub_lm_model, cache=False)

    file_refs = [File(path=str(p.resolve())) for p in images]

    print("Analyzing images...")
    print("-" * 60)

    analyzer = ImageAnalyzer(
        sub_lm=sub_lm,
        max_iterations=args.max_iterations,
        verbose=True,
        debug=args.debug,
    )
    start_time = time.perf_counter()
    with dspy.context(lm=lm):
        prediction = await analyzer.aforward(images=file_refs, query=args.query)
    run_duration = time.perf_counter() - start_time

    # Print results
    print()
    print("=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(prediction.answer)
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
    print(f"Main LM:   {args.model}")
    print(f"Sub-LM:    {args.sub_lm_model}")
    print(f"Images:    {len(images)}")
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
    print(f"Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
