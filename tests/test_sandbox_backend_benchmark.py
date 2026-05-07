from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "benchmarks"
    / "sandbox_backend_benchmark.py"
)


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("sandbox_backend_benchmark", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_aggregate_stats_handles_empty_and_percentiles():
    bench = load_benchmark_module()

    empty = bench.aggregate_stats([])
    assert empty.mean_seconds is None
    assert empty.p50_seconds is None
    assert empty.p95_seconds is None

    stats = bench.aggregate_stats([0.5, 0.1, 0.3, 0.2, 0.4])
    assert stats.mean_seconds == pytest.approx(0.3)
    assert stats.p50_seconds == pytest.approx(0.3)
    assert stats.p95_seconds == pytest.approx(0.5)


def test_scenario_selection_filters_by_backend():
    bench = load_benchmark_module()

    assert bench.selected_backends("all") == ["jspi", "sbx", "sbx-pool"]
    assert bench.applicable_scenarios("jspi", None) == [
        "warm_tiny",
        "warm_fib_recursive",
        "warm_fib_iterative",
    ]
    assert bench.applicable_scenarios("jspi", ["startup_execute_shutdown"]) == [
        "startup_execute_shutdown"
    ]
    assert bench.applicable_scenarios("sbx", ["pool_tiny", "warm_tiny"]) == ["warm_tiny"]
    assert bench.applicable_scenarios("sbx-pool", None) == ["pool_tiny"]
    assert bench.applicable_scenarios("sbx-pool", ["warm_tiny", "pool_tiny"]) == ["pool_tiny"]
    assert bench.applicable_scenarios(
        "sbx-pool", ["pool_replenish_tiny", "pool_replenish_fib_iterative"]
    ) == ["pool_replenish_tiny", "pool_replenish_fib_iterative"]


def test_parser_defaults_and_sbx_pool_size_default():
    bench = load_benchmark_module()

    args = bench.make_parser().parse_args([])
    bench.validate_args(args)

    assert args.backend == "all"
    assert args.tasks is None
    assert args.scenario is None
    assert args.concurrency == 1
    assert args.sbx_pool_size is None
    assert args.sbx_buffer_size is None

    assert bench.task_count_for_scenario("startup_execute_shutdown", args.tasks) == 5
    assert bench.task_count_for_scenario("warm_tiny", args.tasks) == 100
    assert bench.task_count_for_scenario("warm_fib_recursive", args.tasks) == 25
    assert bench.task_count_for_scenario("pool_tiny", args.tasks) == 100
    assert bench.task_count_for_scenario("pool_replenish_tiny", args.tasks) == 100
    assert bench.task_count_for_scenario("pool_replenish_fib_iterative", args.tasks) == 25


def test_validate_args_rejects_negative_buffer_size():
    bench = load_benchmark_module()

    args = bench.make_parser().parse_args(["--sbx-buffer-size", "-1"])

    with pytest.raises(SystemExit, match="--sbx-buffer-size must be at least 0"):
        bench.validate_args(args)


def test_sbx_buffer_size_defaults_to_pool_size():
    bench = load_benchmark_module()

    assert bench.sbx_buffer_size_for_pool(pool_size=5, requested_buffer_size=None) == 5
    assert bench.sbx_buffer_size_for_pool(pool_size=5, requested_buffer_size=2) == 2


def test_parser_rejects_old_surge_flag():
    bench = load_benchmark_module()

    with pytest.raises(SystemExit):
        bench.make_parser().parse_args(["--sbx-surge", "1"])


def test_tasks_override_applies_to_all_scenarios():
    bench = load_benchmark_module()

    args = bench.make_parser().parse_args(["--tasks", "7"])
    bench.validate_args(args)

    for scenario in bench.SCENARIOS:
        assert bench.task_count_for_scenario(scenario, args.tasks) == 7


def test_build_result_counts_failures_without_negative_ok():
    bench = load_benchmark_module()

    result = bench.build_result(
        backend="sbx",
        scenario="warm_tiny",
        tasks=100,
        wall_seconds=0.2,
        durations=[],
        failures=["startup failed"],
    )

    assert result.ok == 0
    assert result.failed == 1
    assert result.tasks_per_second == 0.0


def test_pool_replenishment_starts_when_sandbox_is_acquired(monkeypatch):
    bench = load_benchmark_module()
    replacement_started = bench.threading.Event()
    execute_entered = bench.threading.Event()
    created: list["FakeInterpreter"] = []

    class FakeInterpreter:
        def __init__(self, index: int) -> None:
            self.index = index

        def prewarm(self) -> None:
            created.append(self)
            if self.index == 2:
                replacement_started.set()

        def shutdown(self) -> None:
            pass

    def make_fake_interpreter(prefix: str, index: int) -> FakeInterpreter:
        return FakeInterpreter(index)

    def execute_with_replacement_overlap(interpreter: FakeInterpreter, code: str) -> None:
        assert interpreter.index == 0
        assert code == bench.TINY_CODE
        execute_entered.set()
        assert replacement_started.wait(timeout=1.0)

    monkeypatch.setattr(bench, "make_sbx_pool_interpreter", make_fake_interpreter)
    monkeypatch.setattr(bench, "execute_code", execute_with_replacement_overlap)

    result = bench.run_pool_replenish(
        "pool_replenish_tiny",
        tasks=1,
        concurrency=1,
        pool_size=1,
        buffer_size=1,
        fail_fast=True,
    )

    assert result.failed == 0
    assert [interpreter.index for interpreter in created] == [0, 1, 2]


def test_pool_replenishment_wall_time_excludes_slow_retirement(monkeypatch):
    bench = load_benchmark_module()
    slow_shutdown_seconds = 0.2

    class FakeInterpreter:
        def __init__(self, index: int) -> None:
            self.index = index

        def prewarm(self) -> None:
            pass

        def shutdown(self) -> None:
            time.sleep(slow_shutdown_seconds)

    def make_fake_interpreter(prefix: str, index: int) -> FakeInterpreter:
        return FakeInterpreter(index)

    def execute_fast(interpreter: FakeInterpreter, code: str) -> None:
        assert code == bench.TINY_CODE

    monkeypatch.setattr(bench, "make_sbx_pool_interpreter", make_fake_interpreter)
    monkeypatch.setattr(bench, "execute_code", execute_fast)

    result = bench.run_pool_replenish(
        "pool_replenish_tiny",
        tasks=1,
        concurrency=1,
        pool_size=1,
        buffer_size=1,
        fail_fast=True,
    )

    assert result.failed == 0
    assert result.wall_seconds < slow_shutdown_seconds / 2
