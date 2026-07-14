from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import dspy
import pytest

from predict_rlm import PredictRLM, Workspace


class WorkspaceCancellationSignature(dspy.Signature):
    workspace: Workspace = dspy.InputField()
    answer: str = dspy.OutputField()


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("deno") is None, reason="JSPI lifecycle test requires Deno")
@pytest.mark.asyncio
async def test_jspi_cancellation_flushes_mirror_before_sandbox_shutdown(tmp_path: Path):
    mutation_completed = asyncio.Event()

    async def signal_mutation() -> str:
        """Tell the host that the sandbox mutation completed."""
        asyncio.get_running_loop().call_later(0.1, mutation_completed.set)
        return "ok"

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    source = workspace_root / "source.txt"
    source.write_text("before", encoding="utf-8")

    rlm = PredictRLM(
        WorkspaceCancellationSignature,
        lm=MagicMock(history=[]),
        tools={"signal_mutation": signal_mutation},
        max_iterations=1,
        verbose=False,
    )
    rlm.generate_action.acall = AsyncMock(
        return_value=dspy.Prediction(
            reasoning="mutate the workspace and remain active",
            code=(
                "from pathlib import Path\n"
                "Path('/sandbox/workspace/source.txt').write_text('after-cancel')\n"
                "await signal_mutation()\n"
                "while True:\n"
                "    pass"
            ),
        )
    )
    rlm._configure_run_predictors = MagicMock()

    invocation = asyncio.create_task(
        rlm.aforward(workspace=Workspace(path=str(workspace_root)))
    )
    mutation_wait = asyncio.create_task(mutation_completed.wait())
    done, _ = await asyncio.wait(
        {invocation, mutation_wait},
        timeout=30,
        return_when=asyncio.FIRST_COMPLETED,
    )
    if invocation in done:
        await invocation
    if mutation_wait not in done:
        invocation.cancel()
        mutation_wait.cancel()
        await asyncio.gather(invocation, mutation_wait, return_exceptions=True)
        pytest.fail("sandbox mutation did not complete before the test timeout")
    invocation.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(invocation, timeout=30)

    assert source.read_text(encoding="utf-8") == "after-cancel"
