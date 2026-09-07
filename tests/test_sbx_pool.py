"""Focused tests for the SBX pool lifecycle."""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("websockets")

from predict_rlm.backends.sbx import SbxConfig, SbxPool  # noqa: E402
from predict_rlm.backends.sbx.execution import SbxPoolExecutionBackend  # noqa: E402
from predict_rlm.runtime import (  # noqa: E402
    ExecutionSpec,
    SessionRequirements,
    UnsupportedOperationError,
)


class AsyncFakeInterpreter:
    def __init__(self, index: int, events: list[tuple[Any, ...]]) -> None:
        self.index = index
        self.events = events
        self.fail_configure = False
        self.fail_reset = False
        self.fail_prewarm = False
        self.prewarm_started: asyncio.Event | None = None
        self.prewarm_release: asyncio.Event | None = None
        self.live_host_work = False
        self.retirement_started: asyncio.Event | None = None
        self.retirement_release: asyncio.Event | None = None

    async def aprewarm(self) -> None:
        await asyncio.sleep(0)
        self.events.append(("prewarm", self.index))
        if self.fail_prewarm:
            raise RuntimeError("prewarm failed")
        if self.prewarm_started is not None:
            self.prewarm_started.set()
        if self.prewarm_release is not None:
            await self.prewarm_release.wait()

    async def aconfigure_runtime(self, **kwargs: Any) -> None:
        await asyncio.sleep(0)
        self.events.append(("configure", self.index, kwargs))
        if self.fail_configure:
            self.fail_configure = False
            raise RuntimeError("configure failed")

    async def areset(self) -> None:
        await asyncio.sleep(0)
        if self.live_host_work:
            raise RuntimeError("host work still active")
        self.events.append(("reset", self.index))
        if self.fail_reset:
            raise RuntimeError("reset failed")

    async def ashutdown(self) -> None:
        await asyncio.sleep(0)
        self.events.append(("shutdown", self.index))

    def retire_when_host_work_finishes(self) -> bool:
        if not self.live_host_work:
            return False
        self.events.append(("retire", self.index))
        return True

    async def aretire_when_host_work_finishes(self) -> bool:
        if not self.live_host_work:
            return False
        self.events.append(("aretire", self.index))
        if self.retirement_started is not None:
            self.retirement_started.set()
        if self.retirement_release is not None:
            await self.retirement_release.wait()
        self.live_host_work = False
        await self.ashutdown()
        return True

    def _shutdown_async_transport_after_loop_closed(self) -> None:
        self.events.append(("loop-transport-shutdown", self.index))


class SyncFakeInterpreter:
    def __init__(self, index: int, events: list[tuple[Any, ...]]) -> None:
        self.index = index
        self.events = events
        self.fail_reset = False
        self.fail_prewarm = False

    def prewarm(self) -> None:
        self.events.append(("prewarm", self.index))
        if self.fail_prewarm:
            raise RuntimeError("prewarm failed")

    def configure_runtime(self, **kwargs: Any) -> None:
        self.events.append(("configure", self.index, kwargs))

    def reset(self) -> None:
        self.events.append(("reset", self.index))
        if self.fail_reset:
            raise RuntimeError("reset failed")

    def shutdown(self) -> None:
        self.events.append(("shutdown", self.index))


def make_pool(tmp_path: Path, monkeypatch, *, size: int = 1):
    pool = SbxPool(
        size=size,
        preinstall_packages=False,
        _staging_root=tmp_path / "pool",
    )
    events: list[tuple[Any, ...]] = []
    created: list[AsyncFakeInterpreter] = []

    def create_interpreter(index: int) -> AsyncFakeInterpreter:
        interpreter = AsyncFakeInterpreter(index, events)
        created.append(interpreter)
        return interpreter

    monkeypatch.setattr(pool, "_create_interpreter", create_interpreter)
    return pool, events, created


def make_sync_pool(tmp_path: Path, monkeypatch):
    pool = SbxPool(
        size=1,
        preinstall_packages=False,
        _staging_root=tmp_path / "sync-pool",
    )
    events: list[tuple[Any, ...]] = []
    created: list[SyncFakeInterpreter] = []

    def create_interpreter(index: int) -> SyncFakeInterpreter:
        interpreter = SyncFakeInterpreter(index, events)
        created.append(interpreter)
        return interpreter

    monkeypatch.setattr(pool, "_create_interpreter", create_interpreter)
    return pool, events, created


@pytest.mark.asyncio
async def test_alease_replaces_interpreter_after_reset_failure(tmp_path: Path, monkeypatch):
    pool, events, created = make_pool(tmp_path, monkeypatch)

    async with pool.alease() as interpreter:
        interpreter.fail_reset = True

    assert len(created) == 2
    assert ("shutdown", 0) in events
    assert events.count(("prewarm", 0)) == 2

    async with pool.alease() as interpreter:
        assert interpreter is created[1]

    await pool.ashutdown()


@pytest.mark.asyncio
async def test_cancelled_alease_still_finishes_busy_interpreter_retirement(
    tmp_path: Path,
    monkeypatch,
):
    pool, events, created = make_pool(tmp_path, monkeypatch)
    lease_entered = asyncio.Event()
    retirement_started = asyncio.Event()
    retirement_release = asyncio.Event()

    async def use_busy_interpreter() -> None:
        async with pool.alease() as interpreter:
            interpreter.live_host_work = True
            interpreter.retirement_started = retirement_started
            interpreter.retirement_release = retirement_release
            lease_entered.set()

    lease = asyncio.create_task(use_busy_interpreter())
    await lease_entered.wait()
    await retirement_started.wait()
    lease.cancel()
    await asyncio.sleep(0)

    assert not lease.done()
    assert pool._available.empty()

    retirement_release.set()
    with pytest.raises(asyncio.CancelledError):
        await lease

    assert ("shutdown", 0) in events
    assert list(pool._available.queue) == [created[1]]
    await pool.ashutdown()


@pytest.mark.asyncio
async def test_failed_reset_replacement_never_strands_pool_capacity(
    tmp_path: Path,
    monkeypatch,
):
    pool, _, created = make_pool(tmp_path, monkeypatch)
    await pool.astart()
    created[0].fail_reset = True

    original_create = pool._create_interpreter

    def create_failed_replacement(index):
        replacement = original_create(index)
        replacement.fail_prewarm = True
        return replacement

    monkeypatch.setattr(pool, "_create_interpreter", create_failed_replacement)

    with pytest.raises(RuntimeError, match="prewarm failed"):
        async with pool.alease():
            pass

    assert created[0] not in list(pool._available.queue)
    with pytest.raises(RuntimeError, match="shut down|replacement"):
        await asyncio.wait_for(pool._acquire_interpreter_async(), timeout=0.1)


@pytest.mark.asyncio
async def test_failed_loop_migration_never_requeues_retired_interpreter(
    tmp_path: Path,
    monkeypatch,
):
    pool, _, created = make_pool(tmp_path, monkeypatch)
    await pool.astart()
    retired = created[0]
    old_loop = asyncio.new_event_loop()
    old_loop.close()
    retired._async_loop = old_loop

    original_create = pool._create_interpreter

    def create_failed_replacement(index):
        replacement = original_create(index)
        replacement.fail_prewarm = True
        return replacement

    monkeypatch.setattr(pool, "_create_interpreter", create_failed_replacement)

    with pytest.raises(RuntimeError, match="prewarm failed"):
        async with pool.alease():
            pass

    assert retired not in list(pool._available.queue)
    with pytest.raises(RuntimeError, match="shut down|replacement"):
        await asyncio.wait_for(pool._acquire_interpreter_async(), timeout=0.1)


@pytest.mark.asyncio
async def test_cancelled_loop_migration_publishes_only_warm_replacement(
    tmp_path: Path,
    monkeypatch,
):
    pool, _, created = make_pool(tmp_path, monkeypatch)
    await pool.astart()
    retired = created[0]
    old_loop = asyncio.new_event_loop()
    old_loop.close()
    retired._async_loop = old_loop
    prewarm_started = asyncio.Event()
    prewarm_release = asyncio.Event()

    original_create = pool._create_interpreter

    def create_blocked_replacement(index):
        replacement = original_create(index)
        replacement.prewarm_started = prewarm_started
        replacement.prewarm_release = prewarm_release
        return replacement

    monkeypatch.setattr(pool, "_create_interpreter", create_blocked_replacement)

    async def lease_once() -> None:
        async with pool.alease():
            pytest.fail("cancelled migration must not enter the lease body")

    migration = asyncio.create_task(lease_once())
    await prewarm_started.wait()
    migration.cancel()
    await asyncio.sleep(0)
    assert not migration.done()
    assert retired not in list(pool._available.queue)

    prewarm_release.set()
    with pytest.raises(asyncio.CancelledError):
        await migration

    replacement = created[1]
    assert list(pool._available.queue) == [replacement]
    async with pool.alease() as interpreter:
        assert interpreter is replacement

    await pool.ashutdown()


def test_sync_failed_reset_replacement_never_requeues_retired_interpreter(
    tmp_path: Path,
    monkeypatch,
):
    pool, _, created = make_sync_pool(tmp_path, monkeypatch)
    pool.start()
    retired = created[0]
    retired.fail_reset = True

    original_create = pool._create_interpreter

    def create_failed_replacement(index):
        replacement = original_create(index)
        replacement.fail_prewarm = True
        return replacement

    monkeypatch.setattr(pool, "_create_interpreter", create_failed_replacement)

    with pytest.raises(RuntimeError, match="prewarm failed"):
        with pool.lease():
            pass

    assert retired not in list(pool._available.queue)
    with pytest.raises(RuntimeError, match="replacement"):
        pool._acquire_interpreter()


@pytest.mark.asyncio
async def test_alease_releases_interpreter_when_configuration_fails(
    tmp_path: Path, monkeypatch
):
    pool, events, created = make_pool(tmp_path, monkeypatch)
    await pool.astart()
    created[0].fail_configure = True

    with pytest.raises(RuntimeError, match="configure failed"):
        async with pool.alease():
            pass

    async with pool.alease() as interpreter:
        assert interpreter is created[0]

    await pool.ashutdown()
    assert [event[0] for event in events].count("reset") == 2


@pytest.mark.asyncio
async def test_ashutdown_unblocks_waiter_and_does_not_reset_leased_interpreter(
    tmp_path: Path, monkeypatch
):
    pool, events, _ = make_pool(tmp_path, monkeypatch)

    async def waiting_lease() -> str:
        try:
            async with pool.alease():
                return "acquired"
        except RuntimeError as exc:
            return str(exc)

    async with pool.alease():
        waiter = asyncio.create_task(waiting_lease())
        await asyncio.sleep(0)
        await pool.ashutdown()
        assert await waiter == "SbxPool is shut down"

    assert ("shutdown", 0) in events
    assert ("reset", 0) not in events
    assert pool._available.qsize() == 0


@pytest.mark.asyncio
async def test_cancelling_waiting_alease_does_not_consume_interpreter(
    tmp_path: Path, monkeypatch
):
    pool, _, created = make_pool(tmp_path, monkeypatch)

    async def waiting_lease() -> None:
        async with pool.alease():
            pytest.fail("cancelled waiter acquired an interpreter")

    async with pool.alease():
        waiter = asyncio.create_task(waiting_lease())
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

    async with pool.alease() as interpreter:
        assert interpreter is created[0]

    await pool.ashutdown()


@pytest.mark.asyncio
async def test_alease_does_not_suppress_body_cancellation_during_shutdown(
    tmp_path: Path, monkeypatch
):
    pool, _, _ = make_pool(tmp_path, monkeypatch)
    lease_entered = asyncio.Event()

    async def leased_work() -> None:
        async with pool.alease():
            lease_entered.set()
            await asyncio.Future()

    task = asyncio.create_task(leased_work())
    await lease_entered.wait()
    await pool.ashutdown()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_alease_rejects_direct_workspace_before_start(tmp_path: Path, monkeypatch):
    pool, _, created = make_pool(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="prewarmed SbxPool"):
        async with pool.alease(direct_workspace_mounts=[object()]):
            pass

    assert created == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "spec",
    [
        ExecutionSpec(
            allowed_domains=("other.internal",),
            extra_read_paths=("/host/input",),
            extra_write_paths=("/host/output",),
        ),
        ExecutionSpec(
            allowed_domains=("service.internal",),
            extra_read_paths=("/host/other-input",),
            extra_write_paths=("/host/output",),
        ),
        ExecutionSpec(
            allowed_domains=("service.internal",),
            extra_read_paths=("/host/input",),
            extra_write_paths=("/host/other-output",),
        ),
    ],
)
async def test_pool_execution_rejects_policy_changes_before_lease(
    spec: ExecutionSpec,
):
    class FixedPolicyPool:
        session_requirements = SessionRequirements(
            allowed_domains=("service.internal",),
            extra_read_paths=("/host/input",),
            extra_write_paths=("/host/output",),
        )

        def __init__(self) -> None:
            self.acquisitions = 0

        @asynccontextmanager
        async def alease(self, **kwargs):
            self.acquisitions += 1
            raise AssertionError("mismatched policy reached pool lease")
            yield

    pool = FixedPolicyPool()
    backend = SbxPoolExecutionBackend(pool)

    with pytest.raises(UnsupportedOperationError, match="fixed policy"):
        async with backend.start(
            spec,
            SimpleNamespace(session=None, ownership=None),
        ):
            pass

    assert pool.acquisitions == 0


@pytest.mark.asyncio
async def test_pool_execution_accepts_semantically_reordered_fixed_policy():
    class FixedPolicyPool:
        session_requirements = SessionRequirements(
            allowed_domains=("first.internal", "second.internal"),
            extra_read_paths=("/host/first", "/host/second"),
            extra_write_paths=("/host/output",),
        )

        def __init__(self) -> None:
            self.acquisitions = 0

        @asynccontextmanager
        async def alease(self, **kwargs):
            self.acquisitions += 1
            yield object()

    pool = FixedPolicyPool()
    backend = SbxPoolExecutionBackend(pool)
    spec = ExecutionSpec(
        allowed_domains=("second.internal", "first.internal"),
        extra_read_paths=("/host/second", "/host/first"),
        extra_write_paths=("/host/output",),
    )

    async with backend.start(
        spec,
        SimpleNamespace(session=None, ownership=None),
    ):
        pass

    assert pool.acquisitions == 1


PAYLOAD_PATH = Path(__file__).parents[1] / "src/predict_rlm/backends/supervisor/_payload.py"


@pytest.mark.sbx
class TestSbxPool:
    def test_start_failure_shuts_down_created_interpreters_and_leaves_pool_stopped(
        self, tmp_path: Path, monkeypatch
    ):
        pool = SbxPool(
            size=3,
            config=SbxConfig(name="pool-test"),
            preinstall_packages=False,
            _staging_root=tmp_path / "pool",
        )
        created = []

        class FakeInterpreter:
            def __init__(self, index: int) -> None:
                self.index = index
                self.shutdown_called = False

            def prewarm(self) -> None:
                if self.index == 1:
                    raise RuntimeError("prewarm failed")

            def shutdown(self) -> None:
                self.shutdown_called = True

        def create_interpreter(index: int) -> FakeInterpreter:
            interpreter = FakeInterpreter(index)
            created.append(interpreter)
            return interpreter

        monkeypatch.setattr(pool, "_create_interpreter", create_interpreter)

        with pytest.raises(RuntimeError, match="prewarm failed"):
            pool.start()

        assert created
        assert all(interpreter.shutdown_called for interpreter in created)
        assert not pool._started
        assert pool._all_interpreters == []
        assert pool._available.qsize() == 0

    def test_shutdown_runs_concurrently_and_attempts_all_interpreters(
        self, tmp_path: Path, monkeypatch
    ):
        pool = SbxPool(
            size=3,
            config=SbxConfig(name="pool-test"),
            preinstall_packages=False,
            _staging_root=tmp_path / "pool",
        )
        barrier = threading.Barrier(3)
        active = 0
        max_active = 0
        active_lock = threading.Lock()
        shutdown_indexes: list[int] = []

        class FakeInterpreter:
            def __init__(self, index: int) -> None:
                self.index = index

            def prewarm(self) -> None:
                return None

            def shutdown(self) -> None:
                nonlocal active, max_active
                with active_lock:
                    active += 1
                    max_active = max(max_active, active)
                try:
                    barrier.wait(timeout=1)
                    shutdown_indexes.append(self.index)
                    if self.index == 1:
                        raise RuntimeError("shutdown failed")
                finally:
                    with active_lock:
                        active -= 1

        monkeypatch.setattr(pool, "_create_interpreter", lambda index: FakeInterpreter(index))
        pool.start()

        with pytest.raises(RuntimeError, match="shutdown failed"):
            pool.shutdown()

        assert max_active == 3
        assert sorted(shutdown_indexes) == [0, 1, 2]
        assert not pool._started
        assert pool._shutdown
        assert pool._all_interpreters == []
        assert pool._available.qsize() == 0

    def test_lease_is_exclusive_and_release_resets(self, tmp_path: Path):
        pool = SbxPool(
            size=1,
            config=SbxConfig(name="pool-test"),
            preinstall_packages=False,
            _supervisor_command=[sys.executable, "-u", str(PAYLOAD_PATH)],
            _staging_root=tmp_path / "pool",
        )
        acquired = threading.Event()
        released = threading.Event()

        def second_lease() -> None:
            with pool.lease() as interpreter:
                acquired.set()
                assert interpreter.execute("print('x' in globals())").strip() == "False"

        try:
            pool.start()
            with pool.lease() as interpreter:
                interpreter.execute("x = 7")
                staged = pool._all_interpreters[0]._host_path_for_virtual_path(
                    "/sandbox/output/value.txt"
                )
                staged.parent.mkdir(parents=True, exist_ok=True)
                staged.write_text("leaked", encoding="utf-8")
                thread = threading.Thread(target=second_lease)
                thread.start()
                assert not acquired.wait(0.2)

            released.set()
            thread.join(timeout=5)

            assert released.is_set()
            assert acquired.is_set()
            with pool.lease() as interpreter:
                assert interpreter.list_dir("/sandbox") == []
        finally:
            pool.shutdown()

    def test_shutdown_requested_during_start_prevents_waiting_lease_acquire(
        self, tmp_path: Path, monkeypatch
    ):
        pool = SbxPool(
            size=1,
            config=SbxConfig(name="pool-test"),
            preinstall_packages=False,
            _staging_root=tmp_path / "pool",
        )
        prewarm_started = threading.Event()
        allow_prewarm = threading.Event()

        class FakeInterpreter:
            def __init__(self) -> None:
                self.shutdown_called = False

            def prewarm(self) -> None:
                prewarm_started.set()
                assert allow_prewarm.wait(timeout=2)

            def configure_runtime(self, **kwargs) -> None:
                return None

            def reset(self) -> None:
                return None

            def shutdown(self) -> None:
                self.shutdown_called = True

        interpreter = FakeInterpreter()
        monkeypatch.setattr(pool, "_create_interpreter", lambda index: interpreter)

        lease_results: list[str] = []

        def lease_during_start() -> None:
            try:
                with pool.lease():
                    lease_results.append("acquired")
            except RuntimeError as exc:
                lease_results.append(str(exc))

        lease_thread = threading.Thread(target=lease_during_start)
        lease_thread.start()
        assert prewarm_started.wait(timeout=2)

        shutdown_thread = threading.Thread(target=pool.shutdown)
        shutdown_thread.start()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with pool._state_changed:
                if pool._shutdown_requested:
                    break
            time.sleep(0.01)
        else:
            pytest.fail("shutdown did not request pool stop while startup was active")
        allow_prewarm.set()

        lease_thread.join(timeout=2)
        shutdown_thread.join(timeout=2)

        assert not lease_thread.is_alive()
        assert not shutdown_thread.is_alive()
        assert lease_results == ["SbxPool is shut down"]
        assert interpreter.shutdown_called
        assert pool._available.qsize() == 0

    def test_lease_after_shutdown_raises_until_explicit_restart(
        self, tmp_path: Path, monkeypatch
    ):
        pool = SbxPool(
            size=1,
            config=SbxConfig(name="pool-test"),
            preinstall_packages=False,
            _staging_root=tmp_path / "pool",
        )

        class FakeInterpreter:
            def __init__(self, index: int) -> None:
                self.index = index

            def prewarm(self) -> None:
                return None

            def configure_runtime(self, **kwargs) -> None:
                return None

            def reset(self) -> None:
                return None

            def shutdown(self) -> None:
                return None

        created: list[FakeInterpreter] = []

        def create_interpreter(index: int) -> FakeInterpreter:
            interpreter = FakeInterpreter(index)
            created.append(interpreter)
            return interpreter

        monkeypatch.setattr(pool, "_create_interpreter", create_interpreter)

        pool.start()
        pool.shutdown()

        with pytest.raises(RuntimeError, match="SbxPool is shut down"):
            with pool.lease():
                pass

        pool.start()
        try:
            with pool.lease() as interpreter:
                assert interpreter is created[-1]
        finally:
            pool.shutdown()
