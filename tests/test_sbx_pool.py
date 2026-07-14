"""Focused tests for the SBX pool lifecycle."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("websockets")

from predict_rlm.backends.sbx import SbxPool  # noqa: E402


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
async def test_alease_is_exclusive_and_awaits_async_lifecycle(tmp_path: Path, monkeypatch):
    pool, events, created = make_pool(tmp_path, monkeypatch)
    second_acquired = asyncio.Event()

    def forbidden_to_thread(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("async SBX pool lifecycle delegated to a thread")

    monkeypatch.setattr(asyncio, "to_thread", forbidden_to_thread)

    async def second_lease() -> None:
        async with pool.alease() as interpreter:
            assert interpreter is created[0]
            second_acquired.set()

    async with pool.alease(tools={"tool": lambda: None}) as interpreter:
        assert interpreter is created[0]
        waiter = asyncio.create_task(second_lease())
        await asyncio.sleep(0)
        assert not second_acquired.is_set()

    await waiter
    await pool.ashutdown()

    assert [event[0] for event in events] == [
        "prewarm",
        "configure",
        "reset",
        "configure",
        "reset",
        "shutdown",
    ]
    assert list(events[1][2]["tools"]) == ["tool"]


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
async def test_alease_retires_busy_interpreter_without_reset_or_immediate_shutdown(
    tmp_path: Path,
    monkeypatch,
):
    pool, events, created = make_pool(tmp_path, monkeypatch)

    async with pool.alease() as interpreter:
        interpreter.live_host_work = True
        interpreter.fail_reset = True

    assert len(created) == 2
    assert ("reset", 0) not in events
    assert ("aretire", 0) in events
    assert ("shutdown", 0) in events
    assert list(pool._available.queue) == [created[1]]

    created[0].live_host_work = False
    await pool.ashutdown()


@pytest.mark.asyncio
async def test_alease_awaits_busy_interpreter_retirement_before_replacement(
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
    await asyncio.sleep(0.02)

    try:
        assert retirement_started.is_set()
        assert not lease.done()
        assert pool._available.empty()
    finally:
        retirement_release.set()
        await lease

    assert ("shutdown", 0) in events
    assert list(pool._available.queue) == [created[1]]
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
async def test_alease_releases_interpreter_when_configuration_fails(tmp_path: Path, monkeypatch):
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

    assert pool.supports_direct_workspaces is False
    assert created == []
