"""Tests for JspiBackend with concurrent async tool execution."""

import asyncio
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import dspy
import pytest
from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput

from predict_rlm import PredictRLM, Workspace
from predict_rlm.backends import JspiBackend
from predict_rlm.backends.base import SandboxFatalError

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("deno") is None, reason="JSPI tests require Deno"),
]


class TestSubmitDefaults:
    def test_bare_submit_uses_single_output_default(self):
        interpreter = JspiBackend(
            preinstall_packages=False,
            output_fields=[
                {
                    "name": "answer",
                    "type": "str",
                    "has_default": True,
                    "default": None,
                }
            ],
        )
        try:
            result = interpreter.execute("SUBMIT()")
        finally:
            interpreter.shutdown()

        assert isinstance(result, FinalOutput)
        assert result.output == {"answer": None}

    def test_bare_submit_without_default_still_errors(self):
        interpreter = JspiBackend(
            preinstall_packages=False,
            output_fields=[{"name": "answer", "type": "str"}],
        )
        try:
            with pytest.raises(CodeInterpreterError, match="missing.*answer"):
                interpreter.execute("SUBMIT()")
        finally:
            interpreter.shutdown()


class TestCodeFenceStripping:
    def test_repl_fence_handles_inline_backticks(self):
        """Inline ``` (not on own line) inside code is preserved."""
        interpreter = JspiBackend(preinstall_packages=False)
        try:
            # Inline ``` in strings work fine - they're not on their own line
            result = interpreter.execute("""```repl
s = "has ```backticks``` inline"
print(len(s))
```""")
            assert "26" in str(result)
        finally:
            interpreter.shutdown()

    def test_multiple_repl_blocks(self):
        """Multiple ```repl blocks are all extracted and executed in order."""
        interpreter = JspiBackend(preinstall_packages=False)
        try:
            result = interpreter.execute("""Here's step 1:
```repl
x = 10
print(f"x = {x}")
```

And step 2:
```repl
y = 20
print(f"y = {y}")
```

Finally step 3:
```repl
z = x + y
print(f"z = {z}")
```""")
            # All three blocks should execute, and state persists between them
            assert "x = 10" in str(result)
            assert "y = 20" in str(result)
            assert "z = 30" in str(result)
        finally:
            interpreter.shutdown()


class TestPydanticSerialization:
    def test_pydantic_model_tool_result_is_mapping(self):
        from pydantic import BaseModel

        class Source(BaseModel):
            title: str

        class Retrieval(BaseModel):
            model_name: str
            sources: list[Source]

        def retrieve() -> Retrieval:
            return Retrieval(
                model_name="test-index",
                sources=[Source(title="PredictRLM documentation")],
            )

        interpreter = JspiBackend(tools={"retrieve": retrieve})
        try:
            output = interpreter.execute("""
result = await retrieve()
print(result["model_name"])
print(result["sources"][0]["title"])
""")
            assert "test-index" in str(output)
            assert "PredictRLM documentation" in str(output)
        finally:
            interpreter.shutdown()

    def test_nested_pydantic_models(self):
        """Nested Pydantic models are serialized correctly."""
        received_data = []

        def process_nested(data: dict) -> str:
            received_data.append(data)
            return f"Got {len(data.get('items', []))} items"

        interpreter = JspiBackend(tools={"process_nested": process_nested})
        try:
            output = interpreter.execute("""
from pydantic import BaseModel
from typing import List, Optional

class Item(BaseModel):
    id: int
    name: str
    price: float

class Order(BaseModel):
    order_id: str
    customer: str
    items: List[Item]
    notes: Optional[str] = None

order = Order(
    order_id="ORD-123",
    customer="Bob",
    items=[
        Item(id=1, name="Widget", price=9.99),
        Item(id=2, name="Gadget", price=19.99),
    ],
    notes="Rush delivery"
)

result = await process_nested(order)
print(result)
""")
            assert "Got 2 items" in str(output)
            assert len(received_data) == 1
            assert received_data[0]["order_id"] == "ORD-123"
            assert len(received_data[0]["items"]) == 2
            assert received_data[0]["items"][0]["name"] == "Widget"
        finally:
            interpreter.shutdown()

    def test_non_serializable_falls_back_to_string(self):
        """Non-serializable objects are converted to strings gracefully."""
        received_args = []

        def receive_anything(x) -> str:
            received_args.append(x)
            return f"Got type: {type(x).__name__}"

        interpreter = JspiBackend(tools={"receive_anything": receive_anything})
        try:
            # Pass a coroutine (non-standard object) - should be converted to string
            output = interpreter.execute("""
async def dummy():
    return 42

coro = dummy()
result = await receive_anything(coro)
print(result)
# Clean up
coro.close()
""")
            # Should succeed with string representation
            assert "Got type: str" in str(output)
            assert len(received_args) == 1
            # The coroutine should have been converted to its string repr
            assert "coroutine" in str(received_args[0]).lower()

            # Tool should still work normally
            output2 = interpreter.execute("""
result = await receive_anything({"normal": "dict"})
print(result)
""")
            assert "Got type: dict" in str(output2)
        finally:
            interpreter.shutdown()


class TestNoneValueSerialization:
    def test_pydantic_model_with_none_fields_injected_as_variable(self):
        """Pydantic models with None fields are accessible in the sandbox."""
        from pydantic import BaseModel, Field

        class ExtractedItem(BaseModel):
            title: str
            priority: str | None = Field(default=None)
            due_date: str | None = Field(default=None)
            active: bool = True

        items = [
            ExtractedItem(title="Task A", priority=None, due_date=None, active=True),
            ExtractedItem(title="Task B", priority="high", due_date="2025-01-01", active=False),
        ]

        interpreter = JspiBackend(preinstall_packages=False)
        try:
            result = interpreter.execute(
                """
print(f"count={len(items)}")
print(f"a_priority={items[0]['priority']}")
print(f"a_due={items[0]['due_date']}")
print(f"a_active={items[0]['active']}")
print(f"b_priority={items[1]['priority']}")
print(f"b_active={items[1]['active']}")
""",
                variables={"items": items},
            )
            output = str(result)
            assert "count=2" in output
            assert "a_priority=None" in output
            assert "a_due=None" in output
            assert "a_active=True" in output
            assert "b_priority=high" in output
            assert "b_active=False" in output
        finally:
            interpreter.shutdown()


class TestCustomPydanticTypesInSignatures:
    def test_nested_pydantic_types_in_signature(self):
        """Nested Pydantic models extract schemas with $defs for nested types."""
        received_schemas = []

        def mock_predict(signature: str, pydantic_schemas=None, **kwargs):
            received_schemas.append(pydantic_schemas)
            return {
                "person": {"name": "Alice", "address": {"street": "123 Main", "city": "NYC"}}
            }

        interpreter = JspiBackend(tools={"predict": mock_predict})
        try:
            result = interpreter.execute("""
from pydantic import BaseModel

class Address(BaseModel):
    street: str
    city: str

class Person(BaseModel):
    name: str
    address: Address  # Nested model

result = await predict("text: str -> person: Person", text="test")
print(f"Got person: {result['person'].name} at {result['person'].address.city}")
""")
            assert "Got person: Alice at NYC" in str(result)
            # Verify schema was extracted
            assert len(received_schemas) == 1
            schemas = received_schemas[0]
            assert schemas is not None
            assert "Person" in schemas
            # Verify nested type is in $defs
            assert "$defs" in schemas["Person"]
            assert "Address" in schemas["Person"]["$defs"]
        finally:
            interpreter.shutdown()


class TestSerializationFailureRecovery:
    def test_type_error_during_serialization(self):
        """Tools survive TypeError during argument serialization."""
        received = []

        def accept_dict(d: dict) -> str:
            received.append(d)
            return f"Got dict with {len(d)} keys"

        interpreter = JspiBackend(preinstall_packages=False, tools={"accept_dict": accept_dict})
        try:
            # Success
            output1 = interpreter.execute("""
result = await accept_dict({"key": "value"})
print(result)
""")
            assert "Got dict with 1 keys" in str(output1)

            # Try to pass something that looks like it has model_dump but fails
            output2 = interpreter.execute("""
class FakeModel:
    def model_dump(self):
        raise TypeError("Cannot serialize")

try:
    obj = FakeModel()
    result = await accept_dict(obj)
    print("Should not reach")
except Exception as e:
    print(f"Caught: {type(e).__name__}")
""")
            # Should catch the error (either TypeError or the wrapped error)
            assert "Caught:" in str(output2)

            # Tool should still work
            output3 = interpreter.execute("""
result = await accept_dict({"recovery": True})
print(result)
""")
            assert "Got dict with 1 keys" in str(output3)
        finally:
            interpreter.shutdown()


class TestCancellationAndLateResponses:
    def test_late_responses_are_ignored(self):
        """Late tool responses (after cancellation) are gracefully ignored."""
        import asyncio
        import time

        completion_times = []

        async def timed_tool(msg, delay=0.1):
            start = time.time()
            await asyncio.sleep(delay)
            end = time.time()
            completion_times.append((msg, end - start))
            if msg == "fail":
                raise ValueError("Intentional failure")
            return f"result: {msg}"

        interpreter = JspiBackend(
            preinstall_packages=False,
            tools={"timed_tool": timed_tool},
        )
        try:
            # First: Execute a gather where one fails quickly, others take longer
            output1 = interpreter.execute("""
import asyncio

async def run():
    try:
        # fail happens at 0.01s, others at 0.3s
        results = await asyncio.gather(
            timed_tool("slow1", 0.3),
            timed_tool("fail", 0.01),  # Fails quickly
            timed_tool("slow2", 0.3),
        )
        return results
    except RuntimeError as e:
        return f"Caught: {e}"

result = await run()
print(result)
""")
            assert "Caught:" in str(output1)
            assert "Intentional failure" in str(output1)

            # Second: Execute a simple call - should work fine
            # (any late responses from slow1/slow2 should be ignored)
            output2 = interpreter.execute("""
result = await timed_tool("after", 0.05)
print(result)
""")
            assert "result: after" in str(output2)

        finally:
            interpreter.shutdown()


class TestPydanticReconstruction:
    def test_can_add_fields_after_reconstruction(self):
        """LM can add metadata fields to reconstructed models (extra='allow')."""

        def mock_predict(signature: str, pydantic_schemas=None, **kwargs):
            return {
                "tasks": [
                    {"category": "Cert", "title": "Get ISO cert", "extra_field": "bonus"},
                ]
            }

        interpreter = JspiBackend(tools={"predict": mock_predict})
        try:
            result = interpreter.execute("""
from pydantic import BaseModel

class TaskItem(BaseModel):
    category: str
    title: str

result = await predict("doc: str -> tasks: list[TaskItem]", doc="test")
task = result["tasks"][0]
print(f"title: {task.title}, extra: {task.extra_field}")
""")
            assert "title: Get ISO cert, extra: bonus" in str(result)
        finally:
            interpreter.shutdown()

    def test_predict_result_items_no_collision(self):
        """result.items returns stored list, not dict.items() method."""

        def mock_predict(signature: str, pydantic_schemas=None, **kwargs):
            return {"items": ["apple", "banana", "cherry"]}

        interpreter = JspiBackend(tools={"predict": mock_predict})
        try:
            result = interpreter.execute("""
result = await predict("text: str -> items: list[str]", text="fruits")
# This previously collided with dict.items() method
for item in result.items:
    print(f"fruit: {item}")
""")
            output = str(result)
            assert "fruit: apple" in output
            assert "fruit: banana" in output
            assert "fruit: cherry" in output
        finally:
            interpreter.shutdown()


class TestSandboxFatalErrors:
    def test_exec_timeout_raises_sandbox_fatal_error(self):
        """exec_timeout firing raises SandboxFatalError, not CodeInterpreterError."""
        interpreter = JspiBackend(preinstall_packages=False, exec_timeout=2.0)
        try:
            with pytest.raises(SandboxFatalError, match="timed out"):
                interpreter.execute("while True:\n    pass\n")
        finally:
            interpreter.shutdown()


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
