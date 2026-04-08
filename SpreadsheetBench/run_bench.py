import argparse
import asyncio
import json
import logging
import os
import shutil
import time
from datetime import datetime

from tqdm import tqdm

import dspy

from predict_rlm import File
from spreadsheet_rlm import SpreadsheetRLM

LM = "openai/mercury-2"
SUB_LM = "openai/gpt-5.1"


RLM_LOGGER = "dspy.predict.rlm"


def setup_run_logging(log_dir="logs"):
    """Create a run folder and suppress noisy loggers. Returns the run dir path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(log_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Only let the RLM logger through; silence everything else
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger(RLM_LOGGER).setLevel(logging.DEBUG)

    return run_dir


def get_model_config(model: str):
    if model == "openai/gpt-5.4":
        return dict(
            model=model,
            num_retries=5,
            reasoning_effort="none",
        )
    elif model == "openai/mercury-2":
        return dict(
            model="openai/mercury-2",
            api_base="https://api.inceptionlabs.ai/v1",
            api_key=os.environ["INCEPTION_API_KEY"],
            extra_body={"reasoning_effort": "instant"},
        )
    else:
        return dict(
            model=model,
        )


async def run_benchmark(opt):
    run_dir = setup_run_logging()
    print(f"Logs: {run_dir}/")

    dataset_path = os.path.abspath(f"data/{opt.dataset}")
    with open(f"{dataset_path}/dataset.json") as fp:
        dataset = json.load(fp)

    output_dir = f"{dataset_path}/outputs/single_rlm"
    os.makedirs(output_dir, exist_ok=True)

    config = get_model_config(LM)

    lm = dspy.LM(**config, cache=False)
    sub_lm = dspy.LM(SUB_LM, reasoning_effort="none", cache=False)

    rlm = SpreadsheetRLM(
        lm=lm,
        sub_lm=sub_lm,
        max_iterations=opt.max_iterations,
        verbose=True,
        debug=True,
    )

    ran = 0
    succeeded = 0
    skipped = 0
    start_time = time.time()

    for data in tqdm(
        dataset[: opt.limit] if opt.limit else dataset, desc="Running RLM"
    ):
        task_id = str(data["id"])
        instruction = data["instruction"]
        spreadsheet_dir = f"{dataset_path}/{data['spreadsheet_path']}"

        for idx in range(1, 4):
            input_file = f"{spreadsheet_dir}/{idx}_{task_id}_input.xlsx"
            output_file = f"{output_dir}/{idx}_{task_id}_output.xlsx"

            if os.path.exists(output_file) and not opt.overwrite:
                skipped += 1
                continue

            if not os.path.exists(input_file):
                print(f"Missing input: {input_file}, skipping")
                skipped += 1
                continue

            # Set up per-test-case logging
            task_dir = os.path.join(run_dir, str(task_id))
            os.makedirs(task_dir, exist_ok=True)
            log_file = os.path.join(task_dir, f"run_{idx}.log")
            logger = logging.getLogger(RLM_LOGGER)
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
            logger.addHandler(fh)

            ran += 1
            t0 = time.time()
            ok = False
            try:
                result = await rlm.aforward(
                    spreadsheet=File(path=input_file),
                    instruction=instruction,
                )
                if result and result.path and os.path.exists(result.path):
                    shutil.copy2(result.path, output_file)
                    ok = True
                    succeeded += 1
                else:
                    print(f"No output for {task_id} test case {idx}")
            except Exception as e:
                print(f"Error on {task_id} test case {idx}: {e}")
            finally:
                elapsed = time.time() - t0
                mins, secs = divmod(int(elapsed), 60)
                with open(log_file, "a") as f:
                    f.write(f"\n{'=' * 40}\n")
                    f.write(f"STATS: task {task_id} case {idx}\n")
                    f.write(f"{'=' * 40}\n")
                    f.write(f"Duration: {mins}m {secs}s\n")
                    f.write(f"Result: {'OK' if ok else 'FAIL'}\n")
                logger.removeHandler(fh)
                fh.close()

    run_duration = time.time() - start_time

    # Count total outputs on disk
    total_outputs = len(
        [f for f in os.listdir(output_dir) if f.endswith("_output.xlsx")]
    )

    # Collect LM stats
    lm_history = list(lm.history)
    sub_lm_history = list(sub_lm.history)

    lm_cost = sum(e.get("cost", 0) or 0 for e in lm_history)
    lm_input = sum(e.get("usage", {}).get("prompt_tokens", 0) or 0 for e in lm_history)
    lm_output = sum(
        e.get("usage", {}).get("completion_tokens", 0) or 0 for e in lm_history
    )

    sub_lm_cost = sum(e.get("cost", 0) or 0 for e in sub_lm_history)
    sub_lm_input = sum(
        e.get("usage", {}).get("prompt_tokens", 0) or 0 for e in sub_lm_history
    )
    sub_lm_output = sum(
        e.get("usage", {}).get("completion_tokens", 0) or 0 for e in sub_lm_history
    )

    total_cost = lm_cost + sub_lm_cost
    mins, secs = divmod(int(run_duration), 60)
    tasks_ran = succeeded + (ran - succeeded)

    # Print stats to both terminal and log
    stats = f"""
{"=" * 60}
RUN STATS
{"=" * 60}
Main LM:    {LM}
Sub-LM:     {SUB_LM}
Tasks:      {tasks_ran} ({succeeded}/{ran} test cases succeeded, {skipped} skipped)
Duration:   {mins}m {secs}s
Outputs:    {total_outputs} on disk

Main LM ({len(lm_history)} calls):
  Input:  {lm_input:,} tokens
  Output: {lm_output:,} tokens
  Cost:   ${lm_cost:.4f}

Sub-LM ({len(sub_lm_history)} calls):
  Input:  {sub_lm_input:,} tokens
  Output: {sub_lm_output:,} tokens
  Cost:   ${sub_lm_cost:.4f}

Total cost: ${total_cost:.4f}
{"=" * 60}

Outputs at: {output_dir}

To evaluate, run:
  uv run python evaluate_rlm.py --setting single --model rlm --dataset {opt.dataset}"""

    print(stats)
    with open(os.path.join(run_dir, "summary.log"), "w") as f:
        f.write(stats)

    # Write a pointer so evaluate_rlm.py can find the latest run
    with open(os.path.join("logs", "latest"), "w") as f:
        f.write(run_dir)


def main():
    parser = argparse.ArgumentParser(description="Run SpreadsheetBench with RLM")
    parser.add_argument(
        "--dataset", type=str, default="sample_data_200", help="Dataset name"
    )
    parser.add_argument(
        "--sub_lm",
        type=str,
        default=None,
        help="Sub-LM model string (e.g. 'openai/gpt-4o')",
    )
    parser.add_argument(
        "--max_iterations", type=int, default=30, help="Max RLM iterations per task"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of tasks (for testing)"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output files"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose RLM output")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    opt = parser.parse_args()
    asyncio.run(run_benchmark(opt))


if __name__ == "__main__":
    main()
