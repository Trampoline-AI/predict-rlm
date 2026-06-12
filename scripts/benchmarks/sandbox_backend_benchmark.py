#!/usr/bin/env python3
"""Benchmark PredictRLM sandbox backend overhead without an LM."""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import math
import queue
import shutil
import statistics
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

BACKENDS = ("jspi", "sbx", "sbx-pool")
SCENARIOS = (
    "startup_execute_shutdown",
    "warm_tiny",
    "warm_fib_recursive",
    "warm_fib_iterative",
    "warm_file_io",
    "pool_tiny",
    "pool_replenish_tiny",
    "pool_replenish_fib_iterative",
)
DEFAULT_SCENARIOS = (
    "warm_tiny",
    "warm_fib_recursive",
    "warm_fib_iterative",
    "pool_tiny",
)
DEFAULT_TASKS_BY_SCENARIO = {
    "startup_execute_shutdown": 5,
    "warm_tiny": 100,
    "warm_fib_recursive": 25,
    "warm_fib_iterative": 100,
    "warm_file_io": 25,
    "pool_tiny": 100,
    "pool_replenish_tiny": 100,
    "pool_replenish_fib_iterative": 25,
}
TINY_CODE = "x = 1 + 1\nprint(x)"
FIB_RECURSIVE_CODE = """
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

result = fib(28)
if result != 317811:
    raise RuntimeError(f"unexpected fib result: {result}")
print(result)
""".strip()
FIB_ITERATIVE_CODE = """
modulus = 1_000_000_007
a = 0
b = 1
checksum = 0
for i in range(200_000):
    a, b = b, (a + b + i) % modulus
    checksum = (checksum + b * (i + 1)) % modulus
if checksum != 23992021:
    raise RuntimeError(f"unexpected fib checksum: {checksum}")
print(checksum)
""".strip()
FILE_IO_CODE = """
from pathlib import Path

text = Path('/sandbox/input/bench/input.txt').read_text(encoding='utf-8')
out = Path('/sandbox/output/bench/output.txt')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(text.upper() + '\\n' + str(len(text)), encoding='utf-8')
""".strip()


@dataclass(frozen=True)
class AggregateStats:
    mean_seconds: float | None
    p50_seconds: float | None
    p95_seconds: float | None


@dataclass
class BenchmarkResult:
    backend: str
    scenario: str
    tasks: int
    ok: int
    failed: int
    wall_seconds: float
    tasks_per_second: float
    mean_seconds: float | None
    p50_seconds: float | None
    p95_seconds: float | None
    startup_seconds: float | None = None
    error: str | None = None
    durations: list[float] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


def aggregate_stats(durations: list[float]) -> AggregateStats:
    if not durations:
        return AggregateStats(None, None, None)
    sorted_durations = sorted(durations)
    p95_index = max(0, min(len(sorted_durations) - 1, math.ceil(0.95 * len(sorted_durations)) - 1))
    return AggregateStats(
        mean_seconds=statistics.fmean(sorted_durations),
        p50_seconds=statistics.median(sorted_durations),
        p95_seconds=sorted_durations[p95_index],
    )


def applicable_scenarios(backend: str, requested: list[str] | None = None) -> list[str]:
    selected = list(requested) if requested else list(DEFAULT_SCENARIOS)
    if backend == "sbx-pool":
        allowed = {"pool_tiny", "pool_replenish_tiny", "pool_replenish_fib_iterative"}
    else:
        allowed = {
            "startup_execute_shutdown",
            "warm_tiny",
            "warm_fib_recursive",
            "warm_fib_iterative",
            "warm_file_io",
        }
    return [scenario for scenario in selected if scenario in allowed]


def task_count_for_scenario(scenario: str, requested_tasks: int | None) -> int:
    if requested_tasks is not None:
        return requested_tasks
    return DEFAULT_TASKS_BY_SCENARIO[scenario]


def selected_backends(backend: str) -> list[str]:
    return list(BACKENDS) if backend == "all" else [backend]


def sbx_buffer_size_for_pool(pool_size: int, requested_buffer_size: int | None) -> int:
    return pool_size if requested_buffer_size is None else requested_buffer_size


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark PredictRLM JSPI/Pyodide and Docker sbx sandbox backends."
    )
    parser.add_argument("--backend", choices=("jspi", "sbx", "sbx-pool", "all"), default="all")
    parser.add_argument("--tasks", type=int)
    parser.add_argument("--scenario", action="append", choices=SCENARIOS)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--sbx-pool-size", type=int)
    parser.add_argument("--sbx-buffer-size", type=int)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.tasks is not None and args.tasks < 1:
        raise SystemExit("--tasks must be at least 1")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1")
    if args.sbx_pool_size is not None and args.sbx_pool_size < 1:
        raise SystemExit("--sbx-pool-size must be at least 1")
    if args.sbx_buffer_size is not None and args.sbx_buffer_size < 0:
        raise SystemExit("--sbx-buffer-size must be at least 0")


def check_backend_available(backend: str) -> str | None:
    if backend == "jspi" and shutil.which("deno") is None:
        return "Deno executable not found on PATH; install Deno to run JSPI benchmarks."
    if backend in {"sbx", "sbx-pool"} and shutil.which("sbx") is None:
        return "sbx executable not found on PATH; install Docker Sandboxes CLI to run sbx benchmarks."
    return None


def make_interpreter(backend: str):
    if backend == "jspi":
        from predict_rlm.backends import JspiBackend

        return JspiBackend(tools={}, preinstall_packages=False)
    if backend == "sbx":
        from predict_rlm.backends import SbxBackend, SbxConfig

        config = SbxConfig(name=f"predict-rlm-bench-{uuid.uuid4().hex[:12]}")
        return SbxBackend(config=config, tools={}, preinstall_packages=False)
    raise ValueError(f"Unsupported backend: {backend}")


def make_pool(size: int):
    from predict_rlm.backends import SbxConfig, SbxPool

    config = SbxConfig(name=f"predict-rlm-bench-pool-{uuid.uuid4().hex[:12]}")
    return SbxPool(size=size, config=config, tools={}, preinstall_packages=False)


def make_sbx_pool_interpreter(prefix: str, index: int):
    from predict_rlm.backends import SbxBackend, SbxConfig

    config = SbxConfig(name=f"{prefix}-{index}")
    return SbxBackend(config=config, tools={}, preinstall_packages=False)


def execute_file_io(interpreter: Any, source: Path, synced_output: Path) -> None:
    interpreter.mount_file_at(str(source), "/sandbox/input/bench/input.txt")
    interpreter.mkdir_p("/sandbox/output/bench")
    interpreter.execute(FILE_IO_CODE)
    files = interpreter.list_dir("/sandbox/output/bench")
    if "/sandbox/output/bench/output.txt" not in files:
        raise RuntimeError(f"expected output file missing; saw {files}")
    interpreter.sync_file_to("/sandbox/output/bench/output.txt", str(synced_output))
    if not synced_output.exists():
        raise RuntimeError("sync_file_to did not produce host output")


def execute_code(interpreter: Any, code: str) -> None:
    interpreter.execute(code)


def run_timed_task(fn: Callable[[], None]) -> tuple[bool, float, str | None]:
    start = time.perf_counter()
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - benchmark should record failures and continue.
        return False, time.perf_counter() - start, repr(exc)
    return True, time.perf_counter() - start, None


def build_result(
    *,
    backend: str,
    scenario: str,
    tasks: int,
    wall_seconds: float,
    durations: list[float],
    failures: list[str],
    startup_seconds: float | None = None,
    error: str | None = None,
) -> BenchmarkResult:
    stats = aggregate_stats(durations)
    ok = max(0, len(durations) - len(failures))
    return BenchmarkResult(
        backend=backend,
        scenario=scenario,
        tasks=tasks,
        ok=ok,
        failed=len(failures),
        wall_seconds=wall_seconds,
        tasks_per_second=(ok / wall_seconds) if wall_seconds > 0 else 0.0,
        mean_seconds=stats.mean_seconds,
        p50_seconds=stats.p50_seconds,
        p95_seconds=stats.p95_seconds,
        startup_seconds=startup_seconds,
        error=error,
        durations=durations,
        failures=failures,
    )


def skipped_result(backend: str, scenario: str, tasks: int, error: str) -> BenchmarkResult:
    return build_result(
        backend=backend,
        scenario=scenario,
        tasks=tasks,
        wall_seconds=0.0,
        durations=[],
        failures=[],
        error=error,
    )


def run_startup_execute_shutdown(backend: str, tasks: int, fail_fast: bool) -> BenchmarkResult:
    durations: list[float] = []
    failures: list[str] = []
    wall_start = time.perf_counter()
    for _ in range(tasks):
        interpreter = None
        task_start = time.perf_counter()
        error = None
        try:
            interpreter = make_interpreter(backend)
            interpreter.execute(TINY_CODE)
        except Exception as exc:  # noqa: BLE001 - benchmark should record failures and continue.
            error = repr(exc)
        finally:
            if interpreter is not None:
                interpreter.shutdown()
            durations.append(time.perf_counter() - task_start)
            if error is not None:
                failures.append(error)
                if fail_fast:
                    break
    return build_result(
        backend=backend,
        scenario="startup_execute_shutdown",
        tasks=tasks,
        wall_seconds=time.perf_counter() - wall_start,
        durations=durations,
        failures=failures,
    )


def run_warm_interpreter(backend: str, scenario: str, tasks: int, fail_fast: bool) -> BenchmarkResult:
    durations: list[float] = []
    failures: list[str] = []
    interpreter = None
    temp_dir_obj: tempfile.TemporaryDirectory[str] | None = None
    startup_start = time.perf_counter()
    wall_start = startup_start
    try:
        interpreter = make_interpreter(backend)
        interpreter.execute("pass")
        startup_seconds = time.perf_counter() - startup_start
        wall_start = time.perf_counter()

        if scenario == "warm_file_io":
            temp_dir_obj = tempfile.TemporaryDirectory(prefix="predict-rlm-sandbox-bench-")
            temp_dir = Path(temp_dir_obj.name)
            source = temp_dir / "input.txt"
            source.write_text("predict rlm sandbox benchmark\n" * 4, encoding="utf-8")

        for index in range(tasks):
            if scenario == "warm_tiny":
                fn = partial(execute_code, interpreter, TINY_CODE)
            elif scenario == "warm_fib_recursive":
                fn = partial(execute_code, interpreter, FIB_RECURSIVE_CODE)
            elif scenario == "warm_fib_iterative":
                fn = partial(execute_code, interpreter, FIB_ITERATIVE_CODE)
            elif scenario == "warm_file_io":
                synced_output = Path(temp_dir_obj.name) / f"output-{index}.txt"
                fn = partial(execute_file_io, interpreter, source, synced_output)
            else:
                raise ValueError(f"Unsupported warm scenario: {scenario}")

            ok, duration, error = run_timed_task(fn)
            durations.append(duration)
            if not ok and error is not None:
                failures.append(error)
                if fail_fast:
                    break
    except Exception as exc:  # noqa: BLE001 - converted into a benchmark row.
        startup_seconds = time.perf_counter() - startup_start
        failures.append(repr(exc))
    finally:
        if interpreter is not None:
            interpreter.shutdown()
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()
    return build_result(
        backend=backend,
        scenario=scenario,
        tasks=tasks,
        wall_seconds=time.perf_counter() - wall_start,
        durations=durations,
        failures=failures,
        startup_seconds=startup_seconds,
    )


def run_pool_tiny(tasks: int, concurrency: int, pool_size: int, fail_fast: bool) -> BenchmarkResult:
    durations: list[float] = []
    failures: list[str] = []
    pool = None
    startup_start = time.perf_counter()
    wall_start = startup_start
    try:
        pool = make_pool(pool_size)
        pool.start()
        startup_seconds = time.perf_counter() - startup_start
        wall_start = time.perf_counter()

        def task() -> tuple[bool, float, str | None]:
            def leased_execute() -> None:
                with pool.lease() as interpreter:
                    interpreter.execute(TINY_CODE)

            return run_timed_task(leased_execute)

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(task) for _ in range(tasks)]
            for future in concurrent.futures.as_completed(futures):
                ok, duration, error = future.result()
                durations.append(duration)
                if not ok and error is not None:
                    failures.append(error)
                    if fail_fast:
                        for pending in futures:
                            pending.cancel()
                        break
    except Exception as exc:  # noqa: BLE001 - converted into a benchmark row.
        startup_seconds = time.perf_counter() - startup_start
        failures.append(repr(exc))
    finally:
        if pool is not None:
            pool.shutdown()
    return build_result(
        backend="sbx-pool",
        scenario="pool_tiny",
        tasks=tasks,
        wall_seconds=time.perf_counter() - wall_start,
        durations=durations,
        failures=failures,
        startup_seconds=startup_seconds,
    )


def run_pool_replenish(
    scenario: str,
    tasks: int,
    concurrency: int,
    pool_size: int,
    buffer_size: int,
    fail_fast: bool,
) -> BenchmarkResult:
    durations: list[float] = []
    failures: list[str] = []
    ready: queue.Queue[Any] = queue.Queue()
    live: set[Any] = set()
    live_lock = threading.Lock()
    replacement_lock = threading.Lock()
    failure_lock = threading.Lock()
    prefix = f"predict-rlm-bench-replenish-{uuid.uuid4().hex[:12]}"
    next_index = itertools.count()
    startup_seconds = 0.0
    replenish_wall_seconds: float | None = None
    code = TINY_CODE if scenario == "pool_replenish_tiny" else FIB_ITERATIVE_CODE

    def create_interpreter() -> Any:
        interpreter = make_sbx_pool_interpreter(prefix, next(next_index))
        interpreter.prewarm()
        with live_lock:
            live.add(interpreter)
        return interpreter

    def shutdown_interpreter(interpreter: Any, *, suppress_errors: bool = False) -> None:
        try:
            interpreter.shutdown()
        except Exception:
            if not suppress_errors:
                raise
        finally:
            with live_lock:
                live.discard(interpreter)

    startup_start = time.perf_counter()
    wall_start = startup_start
    replacement_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, buffer_size))
    replacement_futures: set[concurrent.futures.Future[Any]] = set()
    retirement_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, pool_size))
    try:
        warm_count = pool_size + buffer_size
        with concurrent.futures.ThreadPoolExecutor(max_workers=warm_count) as executor:
            futures = [executor.submit(create_interpreter) for _ in range(warm_count)]
            for future in concurrent.futures.as_completed(futures):
                ready.put(future.result())
        startup_seconds = time.perf_counter() - startup_start
        wall_start = time.perf_counter()

        def queue_replacement(future: concurrent.futures.Future[Any]) -> None:
            with replacement_lock:
                replacement_futures.discard(future)
            try:
                ready.put(future.result())
            except Exception as exc:  # noqa: BLE001 - benchmark should record failures and continue.
                with failure_lock:
                    failures.append(repr(exc))

        def submit_replacement() -> None:
            future = replacement_executor.submit(create_interpreter)
            with replacement_lock:
                replacement_futures.add(future)
            future.add_done_callback(queue_replacement)

        def submit_retirement(interpreter: Any) -> None:
            retirement_executor.submit(shutdown_interpreter, interpreter, suppress_errors=True)

        def task() -> tuple[bool, float, str | None]:
            interpreter = ready.get()
            submit_replacement()
            try:
                return run_timed_task(partial(execute_code, interpreter, code))
            finally:
                submit_retirement(interpreter)

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(concurrency, pool_size)) as executor:
            futures = [executor.submit(task) for _ in range(tasks)]
            for future in concurrent.futures.as_completed(futures):
                ok, duration, error = future.result()
                durations.append(duration)
                if not ok and error is not None:
                    failures.append(error)
                    if fail_fast:
                        for pending in futures:
                            pending.cancel()
                        break
        replenish_wall_seconds = time.perf_counter() - wall_start
    except Exception as exc:  # noqa: BLE001 - converted into a benchmark row.
        if startup_seconds == 0.0:
            startup_seconds = time.perf_counter() - startup_start
        failures.append(repr(exc))
    finally:
        replacement_executor.shutdown(wait=True)
        retirement_executor.shutdown(wait=True)
        while True:
            try:
                ready.get_nowait()
            except queue.Empty:
                break
        with live_lock:
            interpreters = list(live)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(interpreters))) as executor:
            list(executor.map(partial(shutdown_interpreter, suppress_errors=True), interpreters))
    return build_result(
        backend="sbx-pool",
        scenario=scenario,
        tasks=tasks,
        wall_seconds=replenish_wall_seconds or (time.perf_counter() - wall_start),
        durations=durations,
        failures=failures,
        startup_seconds=startup_seconds,
    )


def run_benchmark(
    backend: str,
    scenario: str,
    *,
    tasks: int,
    concurrency: int,
    sbx_pool_size: int,
    sbx_buffer_size: int,
    fail_fast: bool,
) -> BenchmarkResult:
    unavailable = check_backend_available(backend)
    if unavailable is not None:
        return skipped_result(backend, scenario, tasks, unavailable)
    if scenario == "startup_execute_shutdown":
        return run_startup_execute_shutdown(backend, tasks, fail_fast)
    if scenario in {
        "warm_tiny",
        "warm_fib_recursive",
        "warm_fib_iterative",
        "warm_file_io",
    }:
        return run_warm_interpreter(backend, scenario, tasks, fail_fast)
    if scenario == "pool_tiny" and backend == "sbx-pool":
        return run_pool_tiny(tasks, concurrency, sbx_pool_size, fail_fast)
    if scenario in {"pool_replenish_tiny", "pool_replenish_fib_iterative"} and backend == "sbx-pool":
        return run_pool_replenish(
            scenario, tasks, concurrency, sbx_pool_size, sbx_buffer_size, fail_fast
        )
    return skipped_result(backend, scenario, tasks, f"Scenario {scenario!r} is not applicable")


def fmt_seconds(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def print_table(results: list[BenchmarkResult]) -> None:
    headers = [
        "backend",
        "scenario",
        "tasks",
        "ok",
        "failed",
        "wall_s",
        "tasks/s",
        "mean_s",
        "p50_s",
        "p95_s",
        "startup_s",
    ]
    rows = [
        [
            result.backend,
            result.scenario,
            str(result.tasks),
            str(result.ok),
            str(result.failed),
            fmt_seconds(result.wall_seconds),
            f"{result.tasks_per_second:.2f}",
            fmt_seconds(result.mean_seconds),
            fmt_seconds(result.p50_seconds),
            fmt_seconds(result.p95_seconds),
            fmt_seconds(result.startup_seconds),
        ]
        for result in results
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows)) if rows else len(header)
        for index, header in enumerate(headers)
    ]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))

    errors = [result for result in results if result.error or result.failures]
    if errors:
        print("\nErrors/skips:")
        for result in errors:
            message = result.error or result.failures[0]
            print(f"- {result.backend}/{result.scenario}: {message}")


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    validate_args(args)

    pool_size = args.sbx_pool_size or args.concurrency
    sbx_buffer_size = sbx_buffer_size_for_pool(pool_size, args.sbx_buffer_size)
    results: list[BenchmarkResult] = []
    for backend in selected_backends(args.backend):
        scenarios = applicable_scenarios(backend, args.scenario)
        for scenario in scenarios:
            tasks = task_count_for_scenario(scenario, args.tasks)
            print(f"Running {backend}/{scenario} ({tasks} tasks)...", flush=True)
            result = run_benchmark(
                backend,
                scenario,
                tasks=tasks,
                concurrency=args.concurrency,
                sbx_pool_size=pool_size,
                sbx_buffer_size=sbx_buffer_size,
                fail_fast=args.fail_fast,
            )
            results.append(result)
            print_table([result])
            print(flush=True)
            if args.fail_fast and result.failed:
                print_table(results)
                if args.json_out:
                    args.json_out.write_text(json.dumps([asdict(item) for item in results], indent=2))
                return 1

    print_table(results)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps([asdict(item) for item in results], indent=2))
    return 1 if any(result.failed for result in results) else 0


if __name__ == "__main__":
    sys.exit(main())
