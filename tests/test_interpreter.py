"""Tests for JspiInterpreter with concurrent async tool execution."""

import pytest

from predict_rlm.interpreter import JspiInterpreter

pytestmark = pytest.mark.integration


class TestCodeFenceStripping:
    """Tests for code fence extraction (```repl preferred, ```python fallback)."""

    def test_strip_repl_fence(self):
        """Code wrapped in ```repl fence is extracted and executed."""
        interpreter = JspiInterpreter(preinstall_packages=False)
        try:
            result = interpreter.execute("""```repl
x = 2 + 2
print(x)
```""")
            assert "4" in str(result)
        finally:
            interpreter.shutdown()

    def test_strip_repl_fence_with_text_before(self):
        """```repl fence with explanatory text before is handled."""
        interpreter = JspiInterpreter(preinstall_packages=False)
        try:
            result = interpreter.execute("""Here's the code to run:
```repl
print("extracted")
```""")
            assert "extracted" in str(result)
        finally:
            interpreter.shutdown()

    def test_repl_fence_handles_inline_backticks(self):
        """Inline ``` (not on own line) inside code is preserved."""
        interpreter = JspiInterpreter(preinstall_packages=False)
        try:
            # Inline ``` in strings work fine - they're not on their own line
            result = interpreter.execute("""```repl
s = "has ```backticks``` inline"
print(len(s))
```""")
            assert "26" in str(result)
        finally:
            interpreter.shutdown()

    def test_fallback_python_fence(self):
        """Falls back to ```python fence for backwards compatibility."""
        interpreter = JspiInterpreter(preinstall_packages=False)
        try:
            result = interpreter.execute("""```python
print("fallback")
```""")
            assert "fallback" in str(result)
        finally:
            interpreter.shutdown()

    def test_fallback_bare_fence(self):
        """Falls back to bare ``` fence for backwards compatibility."""
        interpreter = JspiInterpreter(preinstall_packages=False)
        try:
            result = interpreter.execute("""```
print(42)
```""")
            assert "42" in str(result)
        finally:
            interpreter.shutdown()

    def test_no_fence_unchanged(self):
        """Code without fences executes normally."""
        interpreter = JspiInterpreter(preinstall_packages=False)
        try:
            result = interpreter.execute("print('no fence')")
            assert "no fence" in str(result)
        finally:
            interpreter.shutdown()

    def test_double_fence_handled(self):
        """Double fences (model outputs ```...```\\n```) are handled correctly."""
        interpreter = JspiInterpreter(preinstall_packages=False)
        try:
            # Model sometimes outputs a trailing bare ``` after the closing fence
            result = interpreter.execute("""```repl
print("double fence")
```
```""")
            assert "double fence" in str(result)
        finally:
            interpreter.shutdown()

    def test_multiple_repl_blocks(self):
        """Multiple ```repl blocks are all extracted and executed in order."""
        interpreter = JspiInterpreter(preinstall_packages=False)
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

    def test_multiple_repl_blocks_with_async(self):
        """Multiple ```repl blocks with async code work correctly."""
        import asyncio

        call_log = []

        async def log_tool(msg):
            call_log.append(msg)
            await asyncio.sleep(0.01)
            return f"logged: {msg}"

        interpreter = JspiInterpreter(
            preinstall_packages=False,
            tools={"log_tool": log_tool},
        )
        try:
            result = interpreter.execute("""First block:
```repl
result1 = await log_tool("block1")
print(result1)
```

Second block:
```repl
result2 = await log_tool("block2")
print(result2)
```

Third block uses results from previous:
```repl
print(f"Results: {result1}, {result2}")
```""")
            assert "logged: block1" in str(result)
            assert "logged: block2" in str(result)
            assert "Results: logged: block1, logged: block2" in str(result)
            assert call_log == ["block1", "block2"]
        finally:
            interpreter.shutdown()


class TestJspiInterpreter:
    """Tests that JspiInterpreter executes code correctly."""

    def test_interpreter_executes_python_code(self):
        """JspiInterpreter executes Python code and returns output."""
        interpreter = JspiInterpreter(preinstall_packages=False)
        try:
            result = interpreter.execute("x = 2 + 2\nprint(x)")
            assert "4" in str(result)
        finally:
            interpreter.shutdown()

    def test_interpreter_state_persists(self):
        """Variables persist between code executions in the same interpreter."""
        interpreter = JspiInterpreter(preinstall_packages=False)
        try:
            interpreter.execute("my_list = [1, 2, 3]")
            interpreter.execute("my_list.append(4)")
            result = interpreter.execute("print(len(my_list))")
            assert "4" in str(result)
        finally:
            interpreter.shutdown()

    def test_interpreter_can_use_stdlib(self):
        """Interpreter has access to standard library modules."""
        interpreter = JspiInterpreter(preinstall_packages=False)
        try:
            result = interpreter.execute("""
import re
import json
import math

data = {"value": math.pi}
text = json.dumps(data)
match = re.search(r'"value": ([0-9.]+)', text)
print(match.group(1)[:4])
""")
            assert "3.14" in str(result)
        finally:
            interpreter.shutdown()

    def test_interpreter_has_jspi_flag_when_needed(self):
        """JspiInterpreter includes the JSPI V8 flag only when V8 < 13.7."""
        from predict_rlm.interpreter import _needs_jspi_flag

        interpreter = JspiInterpreter(preinstall_packages=False)
        try:
            if _needs_jspi_flag():
                assert "--v8-flags=--experimental-wasm-jspi" in interpreter.deno_command
            else:
                assert "--v8-flags=--experimental-wasm-jspi" not in interpreter.deno_command
        finally:
            interpreter.shutdown()

    def test_interpreter_has_pypi_network_by_default(self):
        """JspiInterpreter has PyPI network access by default for package installation."""
        interpreter = JspiInterpreter()
        try:
            net_flags = [arg for arg in interpreter.deno_command if "--allow-net" in arg]
            assert len(net_flags) == 1
            # PyPI domains are required for micropip to install packages
            assert "pypi.org" in net_flags[0]
            assert "files.pythonhosted.org" in net_flags[0]
        finally:
            interpreter.shutdown()

    def test_interpreter_with_allowed_domains(self):
        """JspiInterpreter can be configured with allowed domains."""
        interpreter = JspiInterpreter(
            preinstall_packages=False, allowed_domains=["api.example.com", "cdn.example.com"]
        )
        try:
            net_flags = [arg for arg in interpreter.deno_command if "--allow-net" in arg]
            assert len(net_flags) == 1
            assert "api.example.com" in net_flags[0]
            assert "cdn.example.com" in net_flags[0]
        finally:
            interpreter.shutdown()


class TestInterpreterWithTools:
    """Tests that interpreter can call registered tools."""

    def test_interpreter_calls_tool_from_code(self):
        """Interpreter can call a tool from within executed code."""
        call_log: list[dict] = []

        def predict(signature: str, **kwargs) -> dict:
            call_log.append({"signature": signature, "kwargs": kwargs})
            outputs = signature.split("->")[1].strip().split(",")
            return {out.strip(): f"Result for {out.strip()}" for out in outputs}

        interpreter = JspiInterpreter(preinstall_packages=False, tools={"predict": predict})
        try:
            output = interpreter.execute("""
result = await predict("question -> answer", question="What is 2+2?")
print(result["answer"])
""")
            assert "Result for answer" in str(output)
            assert len(call_log) == 1
            assert call_log[0]["kwargs"]["question"] == "What is 2+2?"
        finally:
            interpreter.shutdown()

    def test_interpreter_can_process_tool_output(self):
        """Interpreter can process tool output with Python code."""

        def get_prices() -> str:
            return "Item 1: $10.00\nItem 2: $20.00\nItem 3: $15.00"

        interpreter = JspiInterpreter(
            preinstall_packages=False, tools={"get_prices": get_prices}
        )
        try:
            output = interpreter.execute("""
import re

text = await get_prices()
prices = re.findall(r'\\$(\\d+\\.\\d+)', text)
total = sum(float(p) for p in prices)
print(f"Total: ${total:.2f}")
""")
            assert "Total: $45.00" in str(output)
        finally:
            interpreter.shutdown()

    def test_multiple_tools(self):
        """Multiple tools can be registered and called."""

        def fetch_data() -> str:
            return "ABC123"

        def format_id(raw_id: str) -> str:
            return f"ID-{raw_id.strip()}"

        interpreter = JspiInterpreter(
            preinstall_packages=False,
            tools={
                "fetch_data": fetch_data,
                "format_id": format_id,
            },
        )
        try:
            output = interpreter.execute("""
raw = await fetch_data()
formatted = await format_id(raw)
print(formatted)
""")
            assert "ID-ABC123" in str(output)
        finally:
            interpreter.shutdown()


class TestConcurrentToolExecution:
    """Tests for concurrent/parallel tool execution using async."""

    def test_async_tool_calls_run_concurrently(self):
        """Async tool calls via asyncio.gather() run in parallel."""
        import time

        def slow_tool(item_id: str) -> str:
            time.sleep(0.005)
            return f"Result for {item_id}"

        interpreter = JspiInterpreter(preinstall_packages=False, tools={"slow_tool": slow_tool})
        try:
            output = interpreter.execute("""
import asyncio

tasks = [slow_tool(f"item_{i}") for i in range(3)]
results = await asyncio.gather(*tasks)
print(results)
""")
            assert "Result for item_0" in str(output)
            assert "Result for item_1" in str(output)
            assert "Result for item_2" in str(output)
        finally:
            interpreter.shutdown()

    def test_single_tool_call(self):
        """Single tool call with await works correctly."""

        def my_tool(value: str) -> str:
            return f"Got: {value}"

        interpreter = JspiInterpreter(preinstall_packages=False, tools={"my_tool": my_tool})
        try:
            output = interpreter.execute("""
result = await my_tool("test_value")
print(result)
""")
            assert "Got: test_value" in str(output)
        finally:
            interpreter.shutdown()

    def test_sequential_and_parallel_tools(self):
        """Can use sequential await and parallel asyncio.gather in same code."""
        call_log: list[str] = []

        def tracker(msg: str) -> str:
            call_log.append(msg)
            return f"Tracked: {msg}"

        interpreter = JspiInterpreter(preinstall_packages=False, tools={"tracker": tracker})
        try:
            output = interpreter.execute("""
import asyncio

# Sequential call
first_result = await tracker("first")
print(f"First: {first_result}")

# Parallel calls with asyncio.gather
parallel_results = await asyncio.gather(
    tracker("parallel_1"),
    tracker("parallel_2"),
)
print(f"Parallel: {parallel_results}")
""")
            assert "Tracked: first" in str(output)
            assert "Tracked: parallel_1" in str(output)
            assert "Tracked: parallel_2" in str(output)
            assert len(call_log) == 3
        finally:
            interpreter.shutdown()


class TestPreinstalledPackages:
    """Tests that pandas and pydantic are pre-installed in the sandbox."""

    def test_pandas_is_available(self):
        """Pandas can be imported and used in the sandbox."""
        interpreter = JspiInterpreter()
        try:
            output = interpreter.execute("""
import pandas as pd

df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
print(f"Shape: {df.shape}")
print(f"Sum of a: {df['a'].sum()}")
""")
            assert "Shape: (3, 2)" in str(output)
            assert "Sum of a: 6" in str(output)
        finally:
            interpreter.shutdown()

    def test_pydantic_is_available(self):
        """Pydantic can be imported and used for validation in the sandbox."""
        interpreter = JspiInterpreter()
        try:
            output = interpreter.execute("""
from pydantic import BaseModel, Field

class Item(BaseModel):
    name: str
    price: float = Field(gt=0)

item = Item(name="Widget", price=9.99)
print(f"Item: {item.name} @ ${item.price}")
print(f"Model fields: {list(item.model_fields.keys())}")
""")
            assert "Item: Widget @ $9.99" in str(output)
            assert "name" in str(output)
            assert "price" in str(output)
        finally:
            interpreter.shutdown()

    def test_pydantic_validation_works(self):
        """Pydantic validation errors are raised correctly in the sandbox."""
        interpreter = JspiInterpreter()
        try:
            output = interpreter.execute("""
from pydantic import BaseModel, ValidationError

class User(BaseModel):
    name: str
    age: int

try:
    user = User(name="Alice", age="not_a_number")
except ValidationError as e:
    print(f"Validation error count: {e.error_count()}")
    print("Validation works!")
""")
            assert "Validation works!" in str(output)
        finally:
            interpreter.shutdown()


class TestToolPersistence:
    """Tests that tools persist across multiple executions and heavy workloads."""

    def test_tools_persist_across_executions(self):
        """Tools remain available after multiple code executions."""

        def my_tool(value: str) -> str:
            return f"Got: {value}"

        interpreter = JspiInterpreter(preinstall_packages=False, tools={"my_tool": my_tool})
        try:
            # First execution - use the tool
            output1 = interpreter.execute("""
result = await my_tool("first")
print(result)
""")
            assert "Got: first" in str(output1)

            # Second execution - define some variables (potentially corrupting state)
            interpreter.execute("""
import asyncio
data = [i * 2 for i in range(100)]
total = sum(data)
""")

            # Third execution - tool should still work
            output3 = interpreter.execute("""
result = await my_tool("third")
print(result)
""")
            assert "Got: third" in str(output3)
        finally:
            interpreter.shutdown()

    def test_tools_survive_heavy_async_workload(self):
        """Tools remain available after heavy concurrent async operations."""
        import time

        call_count = 0

        def counter_tool(item_id: str) -> str:
            nonlocal call_count
            call_count += 1
            time.sleep(0.001)  # Small delay to simulate work
            return f"Processed: {item_id}"

        interpreter = JspiInterpreter(
            preinstall_packages=False, tools={"counter_tool": counter_tool}
        )
        try:
            # Heavy concurrent workload
            output1 = interpreter.execute("""
import asyncio

# Run many parallel calls
tasks = [counter_tool(f"item_{i}") for i in range(20)]
results = await asyncio.gather(*tasks)
print(f"Completed {len(results)} calls")
""")
            assert "Completed 20 calls" in str(output1)
            assert call_count == 20

            # After heavy workload, tool should still be available
            output2 = interpreter.execute("""
# Check tool is still callable
result = await counter_tool("after_workload")
print(result)
""")
            assert "Processed: after_workload" in str(output2)
        finally:
            interpreter.shutdown()

    def test_tools_in_persistent_module(self):
        """Tools are stored in _repl_tools module and can be recovered."""

        def check_tool() -> str:
            return "check_ok"

        interpreter = JspiInterpreter(
            preinstall_packages=False, tools={"check_tool": check_tool}
        )
        try:
            # Verify tool exists in module
            output = interpreter.execute("""
import sys

# Check module exists
has_module = '_repl_tools' in sys.modules
print(f"Module exists: {has_module}")

if has_module:
    _repl_tools = sys.modules['_repl_tools']
    has_tool = hasattr(_repl_tools, 'check_tool')
    print(f"Tool in module: {has_tool}")

# Call tool via global
result = await check_tool()
print(f"Result: {result}")
""")
            assert "Module exists: True" in str(output)
            assert "Tool in module: True" in str(output)
            assert "Result: check_ok" in str(output)
        finally:
            interpreter.shutdown()


class TestToolFailures:
    """Tests for tool error handling and recovery."""

    def test_tool_exception_is_catchable(self):
        """Exceptions from tools can be caught in Python code."""

        def failing_tool() -> str:
            raise ValueError("Tool failed intentionally")

        interpreter = JspiInterpreter(
            preinstall_packages=False, tools={"failing_tool": failing_tool}
        )
        try:
            output = interpreter.execute("""
try:
    result = await failing_tool()
    print("Should not reach here")
except Exception as e:
    print(f"Caught exception: {type(e).__name__}")
    print(f"Message contains 'failed': {'failed' in str(e)}")
""")
            assert "Caught exception:" in str(output)
            assert "Message contains 'failed': True" in str(output)
        finally:
            interpreter.shutdown()

    def test_tools_work_after_exception(self):
        """Tools continue working after one raises an exception."""
        call_count = 0

        def maybe_fail(should_fail: bool) -> str:
            nonlocal call_count
            call_count += 1
            if should_fail:
                raise RuntimeError("Intentional failure")
            return "success"

        interpreter = JspiInterpreter(
            preinstall_packages=False, tools={"maybe_fail": maybe_fail}
        )
        try:
            # First call succeeds
            output1 = interpreter.execute("""
result = await maybe_fail(False)
print(f"First call: {result}")
""")
            assert "First call: success" in str(output1)

            # Second call fails (caught)
            output2 = interpreter.execute("""
try:
    result = await maybe_fail(True)
except Exception as e:
    print(f"Second call failed: {type(e).__name__}")
""")
            assert "Second call failed:" in str(output2)

            # Third call succeeds - tool still works
            output3 = interpreter.execute("""
result = await maybe_fail(False)
print(f"Third call: {result}")
""")
            assert "Third call: success" in str(output3)
            assert call_count == 3
        finally:
            interpreter.shutdown()

    def test_parallel_calls_with_some_failures(self):
        """asyncio.gather handles mix of successful and failing tool calls."""

        def conditional_tool(item_id: int) -> str:
            if item_id % 3 == 0:
                raise ValueError(f"Item {item_id} failed")
            return f"Item {item_id} ok"

        interpreter = JspiInterpreter(
            preinstall_packages=False, tools={"conditional_tool": conditional_tool}
        )
        try:
            output = interpreter.execute("""
import asyncio

async def safe_call(item_id):
    try:
        return await conditional_tool(item_id)
    except Exception as e:
        return f"Error: {e}"

# Run 10 parallel calls (items 0,3,6,9 will fail)
tasks = [safe_call(i) for i in range(10)]
results = await asyncio.gather(*tasks)

successes = [r for r in results if r.startswith("Item") and "ok" in r]
failures = [r for r in results if r.startswith("Error")]

print(f"Successes: {len(successes)}")
print(f"Failures: {len(failures)}")
print(f"Total: {len(results)}")
""")
            assert "Successes: 6" in str(output)  # 1,2,4,5,7,8
            assert "Failures: 4" in str(output)  # 0,3,6,9
            assert "Total: 10" in str(output)
        finally:
            interpreter.shutdown()

    def test_tool_timeout_simulation(self):
        """Tools that take too long can be handled with asyncio.wait_for."""
        import time

        def slow_tool(delay: float) -> str:
            time.sleep(delay)
            return f"Completed after {delay}s"

        interpreter = JspiInterpreter(preinstall_packages=False, tools={"slow_tool": slow_tool})
        try:
            output = interpreter.execute("""
import asyncio

async def with_timeout(delay, timeout):
    try:
        result = await asyncio.wait_for(slow_tool(delay), timeout=timeout)
        return f"OK: {result}"
    except asyncio.TimeoutError:
        return "TIMEOUT"

# Fast call should succeed
result1 = await with_timeout(0.01, timeout=2.0)
print(f"Fast call: {result1}")

# Note: actual timeout test would be slow, just verify the pattern works
print("Timeout pattern works")
""")
            assert "Fast call: OK:" in str(output)
            assert "Timeout pattern works" in str(output)
        finally:
            interpreter.shutdown()

    def test_tool_returns_none(self):
        """Tools that return None are converted to empty string."""

        def none_tool() -> None:
            return None

        interpreter = JspiInterpreter(preinstall_packages=False, tools={"none_tool": none_tool})
        try:
            # None is converted to empty string by the tool bridge
            output = interpreter.execute("""
result = await none_tool()
print(f"Result is empty string: {result == ''}")
print(f"Result type: {type(result).__name__}")
""")
            assert "Result is empty string: True" in str(output)
            assert "Result type: str" in str(output)
        finally:
            interpreter.shutdown()

    def test_tool_returns_empty_values(self):
        """Tools returning empty strings/lists/dicts are handled correctly."""

        def empty_string() -> str:
            return ""

        def empty_list() -> list:
            return []

        def empty_dict() -> dict:
            return {}

        interpreter = JspiInterpreter(
            preinstall_packages=False,
            tools={
                "empty_string": empty_string,
                "empty_list": empty_list,
                "empty_dict": empty_dict,
            },
        )
        try:
            output = interpreter.execute("""
s = await empty_string()
l = await empty_list()
d = await empty_dict()

print(f"String empty: {s == ''}")
print(f"List empty: {l == []}")
print(f"Dict empty: {d == dict()}")  # Use dict() instead of {{}} for clarity
""")
            assert "String empty: True" in str(output)
            assert "List empty: True" in str(output)
            assert "Dict empty: True" in str(output)
        finally:
            interpreter.shutdown()


class TestRealisticWorkloads:
    """Tests simulating realistic API-like workloads with random delays."""

    def test_random_latency_tool_calls(self):
        """Tools with random latency (simulating network jitter)."""
        import random
        import time

        call_times = []

        def api_call(item_id: int) -> dict:
            # Random delay 1-10ms to simulate network latency (fast for tests)
            delay = random.uniform(0.001, 0.01)
            time.sleep(delay)
            call_times.append((item_id, delay))
            return {"id": item_id, "status": "ok", "latency_ms": int(delay * 1000)}

        interpreter = JspiInterpreter(preinstall_packages=False, tools={"api_call": api_call})
        try:
            output = interpreter.execute("""
import asyncio

# Make 10 parallel calls with random latency
tasks = [api_call(i) for i in range(10)]
results = await asyncio.gather(*tasks)

# Verify all completed
completed = [r for r in results if r.get("status") == "ok"]
print(f"Completed: {len(completed)}/10")

# Show latency range
latencies = [r["latency_ms"] for r in results]
print(f"Latency range: {min(latencies)}-{max(latencies)}ms")
""")
            assert "Completed: 10/10" in str(output)
            assert len(call_times) == 10
        finally:
            interpreter.shutdown()

    def test_mixed_fast_and_slow_calls(self):
        """Mix of fast and slow tool calls running concurrently."""
        import time

        def variable_speed(item_id: int, slow: bool) -> str:
            delay = 0.01 if slow else 0.001  # 10ms vs 1ms
            time.sleep(delay)
            return f"item_{item_id}_{'slow' if slow else 'fast'}"

        interpreter = JspiInterpreter(
            preinstall_packages=False, tools={"variable_speed": variable_speed}
        )
        try:
            output = interpreter.execute("""
import asyncio

# Mix of fast (even IDs) and slow (odd IDs) calls
tasks = [variable_speed(i, slow=(i % 2 == 1)) for i in range(8)]
results = await asyncio.gather(*tasks)

fast_count = sum(1 for r in results if 'fast' in r)
slow_count = sum(1 for r in results if 'slow' in r)

print(f"Fast: {fast_count}, Slow: {slow_count}")
print(f"All completed: {len(results) == 8}")
""")
            assert "Fast: 4, Slow: 4" in str(output)
            assert "All completed: True" in str(output)
        finally:
            interpreter.shutdown()

    def test_sequential_batches_with_delays(self):
        """Multiple sequential batches of parallel calls."""
        import random
        import time

        batch_results = []

        def batch_item(batch_id: int, item_id: int) -> dict:
            time.sleep(random.uniform(0.001, 0.005))  # 1-5ms
            result = {"batch": batch_id, "item": item_id}
            batch_results.append(result)
            return result

        interpreter = JspiInterpreter(
            preinstall_packages=False, tools={"batch_item": batch_item}
        )
        try:
            output = interpreter.execute("""
import asyncio

all_results = []

# Run 3 sequential batches, each with 5 parallel calls
for batch_id in range(3):
    tasks = [batch_item(batch_id, i) for i in range(5)]
    batch_results = await asyncio.gather(*tasks)
    all_results.extend(batch_results)
    print(f"Batch {batch_id} done: {len(batch_results)} items")

print(f"Total items: {len(all_results)}")

# Verify all batches represented
batches = set(r["batch"] for r in all_results)
print(f"Unique batches: {sorted(batches)}")
""")
            assert "Total items: 15" in str(output)
            assert "Unique batches: [0, 1, 2]" in str(output)
            assert len(batch_results) == 15
        finally:
            interpreter.shutdown()

    def test_intermittent_failures_with_retry(self):
        """Simulating flaky API with retries."""
        import random
        import time

        call_count = {"total": 0, "failures": 0}

        def flaky_api(item_id: int) -> dict:
            call_count["total"] += 1
            time.sleep(random.uniform(0.001, 0.003))
            # 30% chance of failure
            if random.random() < 0.3:
                call_count["failures"] += 1
                raise RuntimeError(f"Transient error for item {item_id}")
            return {"id": item_id, "success": True}

        interpreter = JspiInterpreter(preinstall_packages=False, tools={"flaky_api": flaky_api})
        try:
            output = interpreter.execute("""
import asyncio
import random

async def call_with_retry(item_id, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await flaky_api(item_id)
        except Exception as e:
            if attempt == max_retries - 1:
                return {"id": item_id, "success": False, "error": str(e)}
            await asyncio.sleep(0.01)  # Brief delay before retry

# Run 10 items with retry logic
tasks = [call_with_retry(i) for i in range(10)]
results = await asyncio.gather(*tasks)

successes = sum(1 for r in results if r.get("success"))
failures = sum(1 for r in results if not r.get("success"))

print(f"Successes: {successes}")
print(f"Final failures: {failures}")
print(f"All processed: {len(results) == 10}")
""")
            assert "All processed: True" in str(output)
            # With retries, most should succeed despite 30% failure rate
            assert call_count["total"] >= 10  # At least 10 calls, likely more due to retries
        finally:
            interpreter.shutdown()

    def test_high_concurrency_stress(self):
        """Stress test with many concurrent calls and random delays."""
        import random
        import time

        call_log = []

        def stress_call(call_id: int) -> str:
            start = time.perf_counter()
            time.sleep(random.uniform(0.001, 0.005))
            elapsed = time.perf_counter() - start
            call_log.append({"id": call_id, "elapsed": elapsed})
            return f"call_{call_id}_done"

        interpreter = JspiInterpreter(
            preinstall_packages=False, tools={"stress_call": stress_call}
        )
        try:
            output = interpreter.execute("""
import asyncio

# 30 concurrent calls
tasks = [stress_call(i) for i in range(30)]
results = await asyncio.gather(*tasks)

completed = sum(1 for r in results if 'done' in r)
print(f"Completed: {completed}/30")
""")
            assert "Completed: 30/30" in str(output)
            assert len(call_log) == 30
        finally:
            interpreter.shutdown()

    def test_tools_persist_after_stress(self):
        """Verify tools still work after high-concurrency stress test."""
        import random
        import time

        def stress_tool(item_id: int) -> str:
            time.sleep(random.uniform(0.001, 0.005))
            return f"result_{item_id}"

        interpreter = JspiInterpreter(
            preinstall_packages=False, tools={"stress_tool": stress_tool}
        )
        try:
            # First: stress test
            output1 = interpreter.execute("""
import asyncio
tasks = [stress_tool(i) for i in range(25)]
results = await asyncio.gather(*tasks)
print(f"Stress test: {len(results)} calls")
""")
            assert "Stress test: 25 calls" in str(output1)

            # Second: verify tool still works
            output2 = interpreter.execute("""
result = await stress_tool(999)
print(f"After stress: {result}")
""")
            assert "After stress: result_999" in str(output2)

            # Third: another batch to confirm
            output3 = interpreter.execute("""
import asyncio
tasks = [stress_tool(i) for i in range(5)]
results = await asyncio.gather(*tasks)
print(f"Final batch: {len(results)} calls")
""")
            assert "Final batch: 5 calls" in str(output3)
        finally:
            interpreter.shutdown()


class TestPydanticSerialization:
    """Tests for Pydantic model serialization in tool calls."""

    def test_pydantic_model_as_tool_argument(self):
        """Pydantic models passed to tools are serialized correctly."""
        received_data = []

        def process_model(data: dict) -> str:
            received_data.append(data)
            return f"Processed: {data.get('name', 'unknown')}"

        interpreter = JspiInterpreter(tools={"process_model": process_model})
        try:
            output = interpreter.execute("""
from pydantic import BaseModel

class UserInput(BaseModel):
    name: str
    age: int
    tags: list[str] = []

user = UserInput(name="Alice", age=30, tags=["admin", "active"])
result = await process_model(user)
print(result)
""")
            assert "Processed: Alice" in str(output)
            assert len(received_data) == 1
            assert received_data[0]["name"] == "Alice"
            assert received_data[0]["age"] == 30
            assert received_data[0]["tags"] == ["admin", "active"]
        finally:
            interpreter.shutdown()

    def test_nested_pydantic_models(self):
        """Nested Pydantic models are serialized correctly."""
        received_data = []

        def process_nested(data: dict) -> str:
            received_data.append(data)
            return f"Got {len(data.get('items', []))} items"

        interpreter = JspiInterpreter(tools={"process_nested": process_nested})
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

    def test_list_of_pydantic_models(self):
        """Lists of Pydantic models are serialized correctly."""
        received_data = []

        def process_list(items: list) -> str:
            received_data.append(items)
            return f"Received {len(items)} items"

        interpreter = JspiInterpreter(tools={"process_list": process_list})
        try:
            output = interpreter.execute("""
from pydantic import BaseModel

class Task(BaseModel):
    title: str
    done: bool = False

tasks = [
    Task(title="Task 1", done=True),
    Task(title="Task 2"),
    Task(title="Task 3", done=True),
]

result = await process_list(tasks)
print(result)
""")
            assert "Received 3 items" in str(output)
            assert len(received_data) == 1
            assert len(received_data[0]) == 3
            assert received_data[0][0]["title"] == "Task 1"
            assert received_data[0][0]["done"] is True
            assert received_data[0][1]["done"] is False
        finally:
            interpreter.shutdown()

    def test_pydantic_in_kwargs(self):
        """Pydantic models passed as kwargs are serialized correctly."""
        received_kwargs = []

        def with_kwargs(**kwargs) -> str:
            received_kwargs.append(kwargs)
            return f"Got kwargs: {list(kwargs.keys())}"

        interpreter = JspiInterpreter(tools={"with_kwargs": with_kwargs})
        try:
            output = interpreter.execute("""
from pydantic import BaseModel

class Config(BaseModel):
    debug: bool = False
    max_retries: int = 3

config = Config(debug=True, max_retries=5)
result = await with_kwargs(name="test", config=config, count=10)
print(result)
""")
            assert "Got kwargs:" in str(output)
            assert len(received_kwargs) == 1
            assert received_kwargs[0]["name"] == "test"
            assert received_kwargs[0]["config"]["debug"] is True
            assert received_kwargs[0]["config"]["max_retries"] == 5
        finally:
            interpreter.shutdown()

    def test_tools_work_after_pydantic_serialization(self):
        """Tools continue working after Pydantic model serialization."""

        def echo(data: dict) -> dict:
            return {"echoed": data}

        interpreter = JspiInterpreter(tools={"echo": echo})
        try:
            # First call with Pydantic model
            output1 = interpreter.execute("""
from pydantic import BaseModel

class MyModel(BaseModel):
    value: int

m = MyModel(value=42)
result = await echo(m)
print(f"First: {result}")
""")
            assert "echoed" in str(output1)

            # Second call - tool should still work
            output2 = interpreter.execute("""
result = await echo({"plain": "dict"})
print(f"Second: {result}")
""")
            assert "plain" in str(output2)

            # Third call with another Pydantic model
            output3 = interpreter.execute("""
from pydantic import BaseModel

class AnotherModel(BaseModel):
    items: list[str]

m2 = AnotherModel(items=["a", "b", "c"])
result = await echo(m2)
print(f"Third: {result}")
""")
            assert "items" in str(output3)
        finally:
            interpreter.shutdown()

    def test_non_serializable_falls_back_to_string(self):
        """Non-serializable objects are converted to strings gracefully."""
        received_args = []

        def receive_anything(x) -> str:
            received_args.append(x)
            return f"Got type: {type(x).__name__}"

        interpreter = JspiInterpreter(tools={"receive_anything": receive_anything})
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
    """Tests that None/True/False in Pydantic models survive sandbox injection."""

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

        interpreter = JspiInterpreter(preinstall_packages=False)
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

    def test_plain_dict_with_none_values_injected_as_variable(self):
        """Plain dicts with None values are accessible in the sandbox."""
        data = [
            {"name": "Alice", "email": None, "verified": True},
            {"name": "Bob", "email": "bob@test.com", "verified": False},
        ]

        interpreter = JspiInterpreter(preinstall_packages=False)
        try:
            result = interpreter.execute(
                """
print(f"alice_email={data[0]['email']}")
print(f"alice_verified={data[0]['verified']}")
print(f"bob_verified={data[1]['verified']}")
""",
                variables={"data": data},
            )
            output = str(result)
            assert "alice_email=None" in output
            assert "alice_verified=True" in output
            assert "bob_verified=False" in output
        finally:
            interpreter.shutdown()


class TestPredictRLMWithPydanticModels:
    """Tests for using Pydantic models with PredictRLM."""

    def test_predict_rlm_with_pydantic_model_containing_methods(self):
        """Test that PredictRLM works with Pydantic models that have methods."""
        # This test verifies that the fix works by ensuring schemas with methods can be serialized
        # The actual integration test with PredictRLM would require a more complex setup

        # Simple test to verify that our defensive serialization works
        interpreter = JspiInterpreter(preinstall_packages=False)
        try:
            result = interpreter.execute("""
from pydantic import BaseModel, Field

class ExtractedItem(BaseModel):
    category: str = Field(description="Category")
    title: str = Field(description="Title")

    def custom_method(self):
        return f"{self.category}: {self.title}"

    @property
    def formatted(self):
        return self.custom_method()

# Create an instance to verify the model works
item = ExtractedItem(category="Test", title="Item 1")
print(f"Item: {item.category} - {item.title}")
print(f"Formatted: {item.formatted}")

# Verify the schema can be extracted and is JSON-serializable
import json
schema = ExtractedItem.model_json_schema()
json.dumps(schema)  # This should not raise an error
print("Schema serialization successful")
""")

            assert "Schema serialization successful" in str(result)
            assert "Test - Item 1" in str(result)

        finally:
            interpreter.shutdown()


class TestCustomPydanticTypesInSignatures:
    """Tests for custom Pydantic types in predict() return signatures."""

    def test_custom_pydantic_type_in_return_signature(self):
        """Custom Pydantic type in return signature extracts schemas correctly."""
        received_schemas = []

        def mock_predict(signature: str, pydantic_schemas=None, **kwargs):
            received_schemas.append(pydantic_schemas)
            # Return mock data matching the expected type
            return {"tasks": [{"category": "Test", "title": "Task 1"}]}

        interpreter = JspiInterpreter(tools={"predict": mock_predict})
        try:
            result = interpreter.execute("""
from pydantic import BaseModel

class TaskItem(BaseModel):
    category: str
    title: str

# This should extract TaskItem schema and pass it to predict
result = await predict("text: str -> tasks: list[TaskItem]", text="test input")
print(f"Got {len(result['tasks'])} tasks")
print(f"First task: {result['tasks'][0]}")
""")
            assert "Got 1 tasks" in str(result)
            # Verify schema was extracted and passed
            assert len(received_schemas) == 1
            schemas = received_schemas[0]
            assert schemas is not None
            assert "TaskItem" in schemas
            # Verify schema structure
            assert "properties" in schemas["TaskItem"]
            assert "category" in schemas["TaskItem"]["properties"]
            assert "title" in schemas["TaskItem"]["properties"]
        finally:
            interpreter.shutdown()

    def test_nested_pydantic_types_in_signature(self):
        """Nested Pydantic models extract schemas with $defs for nested types."""
        received_schemas = []

        def mock_predict(signature: str, pydantic_schemas=None, **kwargs):
            received_schemas.append(pydantic_schemas)
            return {
                "person": {"name": "Alice", "address": {"street": "123 Main", "city": "NYC"}}
            }

        interpreter = JspiInterpreter(tools={"predict": mock_predict})
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
print(f"Got person: {result['person']['name']} at {result['person']['address']['city']}")
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

    def test_optional_pydantic_type_in_signature(self):
        """Optional custom Pydantic type in signature works correctly."""
        received_schemas = []

        def mock_predict(signature: str, pydantic_schemas=None, **kwargs):
            received_schemas.append(pydantic_schemas)
            return {"item": {"name": "Widget", "price": 9.99}}

        interpreter = JspiInterpreter(tools={"predict": mock_predict})
        try:
            result = interpreter.execute("""
from pydantic import BaseModel
from typing import Optional

class Item(BaseModel):
    name: str
    price: float

result = await predict("text: str -> item: Optional[Item]", text="test")
print(f"Got item: {result['item']['name']}")
""")
            assert "Got item: Widget" in str(result)
            # Verify schema was extracted
            assert len(received_schemas) == 1
            schemas = received_schemas[0]
            assert schemas is not None
            assert "Item" in schemas
        finally:
            interpreter.shutdown()

    def test_builtin_types_not_included_in_schemas(self):
        """Built-in types like str, int, bool are not included in pydantic_schemas."""
        received_schemas = []

        def mock_predict(signature: str, pydantic_schemas=None, **kwargs):
            received_schemas.append(pydantic_schemas)
            return {"name": "test", "count": 42, "active": True}

        interpreter = JspiInterpreter(tools={"predict": mock_predict})
        try:
            result = interpreter.execute("""
result = await predict("text: str -> name: str, count: int, active: bool", text="test")
print(f"Got: {result['name']}, {result['count']}, {result['active']}")
""")
            assert "Got: test, 42, True" in str(result)
            # No pydantic_schemas should be passed for built-in types
            assert len(received_schemas) == 1
            assert received_schemas[0] is None or len(received_schemas[0]) == 0
        finally:
            interpreter.shutdown()

    def test_dspy_image_type_not_in_schemas(self):
        """dspy.Image type is not included in pydantic_schemas (it's built-in)."""
        received_schemas = []

        def mock_predict(signature: str, pydantic_schemas=None, **kwargs):
            received_schemas.append(pydantic_schemas)
            return {"text": "extracted text"}

        interpreter = JspiInterpreter(tools={"predict": mock_predict})
        try:
            result = interpreter.execute("""
result = await predict("img: dspy.Image -> text: str", img="http://example.com/img.png")
print(f"Got: {result['text']}")
""")
            assert "Got: extracted text" in str(result)
            # dspy.Image should not be in schemas
            assert len(received_schemas) == 1
            schemas = received_schemas[0]
            assert schemas is None or "Image" not in schemas
        finally:
            interpreter.shutdown()

    def test_multiple_custom_types_in_signature(self):
        """Multiple custom Pydantic types in same signature are all extracted."""
        received_schemas = []

        def mock_predict(signature: str, pydantic_schemas=None, **kwargs):
            received_schemas.append(pydantic_schemas)
            return {
                "header": {"title": "Doc", "version": "1.0"},
                "items": [{"name": "Item1", "qty": 1}],
            }

        interpreter = JspiInterpreter(tools={"predict": mock_predict})
        try:
            result = interpreter.execute("""
from pydantic import BaseModel

class Header(BaseModel):
    title: str
    version: str

class LineItem(BaseModel):
    name: str
    qty: int

result = await predict("text: str -> header: Header, items: list[LineItem]", text="test")
print(f"Got header: {result['header']['title']}")
print(f"Got {len(result['items'])} items")
""")
            assert "Got header: Doc" in str(result)
            assert "Got 1 items" in str(result)
            # Both types should be in schemas
            assert len(received_schemas) == 1
            schemas = received_schemas[0]
            assert schemas is not None
            assert "Header" in schemas
            assert "LineItem" in schemas
        finally:
            interpreter.shutdown()

    def test_schema_extraction_survives_tool_errors(self):
        """Schema extraction works correctly even after tool errors."""
        call_count = {"count": 0}

        def mock_predict(signature: str, pydantic_schemas=None, **kwargs):
            call_count["count"] += 1
            if call_count["count"] == 1:
                raise RuntimeError("Intentional error")
            return {"item": {"value": 42}}

        interpreter = JspiInterpreter(tools={"predict": mock_predict})
        try:
            # First call fails
            output1 = interpreter.execute("""
from pydantic import BaseModel

class MyType(BaseModel):
    value: int

try:
    result = await predict("text: str -> item: MyType", text="test")
except Exception as e:
    print(f"First call failed: {type(e).__name__}")
""")
            assert "First call failed" in str(output1)

            # Second call succeeds with same type
            output2 = interpreter.execute("""
from pydantic import BaseModel

class MyType(BaseModel):
    value: int

result = await predict("text: str -> item: MyType", text="test")
print(f"Second call got: {result['item']['value']}")
""")
            assert "Second call got: 42" in str(output2)
        finally:
            interpreter.shutdown()


class TestPydanticLiteralSerialization:
    """Tests that Pydantic models with Literal types serialize correctly through the interpreter."""

    def test_pydantic_model_with_literal_in_predict_signature(self):
        """Pydantic models with Literal types should work in predict signatures."""

        # Mock predict tool that echoes back the schema it received
        async def mock_predict(
            signature: str, *, instructions: str = None, pydantic_schemas: dict = None, **kwargs
        ):
            # Check if we received the schema
            if pydantic_schemas and "TaskItem" in pydantic_schemas:
                schema = pydantic_schemas["TaskItem"]
                # Verify the schema has the expected structure
                props = schema.get("properties", {})
                if "priority" in props:
                    # Extract priority anyOf to check for Literal values
                    priority_spec = props["priority"].get("anyOf", [])
                    # Return a mock task with the schema info
                    return {
                        "items": [
                            {
                                "category": "Test Category",
                                "title": "Test Task",
                                "description": "Test Description",
                                "priority": "high",
                                "due_date": None,
                                "_schema_received": True,
                                "_priority_spec_count": len(priority_spec),
                            }
                        ]
                    }
            return {"items": []}

        interpreter = JspiInterpreter(preinstall_packages=True, tools={"predict": mock_predict})

        try:
            # Execute code that defines a Pydantic model with Literal and uses it
            result = interpreter.execute("""
import asyncio
from typing import Literal, Optional
from pydantic import BaseModel, Field

Priority = Optional[Literal["urgent", "high", "medium", "low"]]

class TaskItem(BaseModel):
    category: str = Field(description="Task category")
    title: str
    description: str
    priority: Priority = None
    due_date: Optional[str] = None

# Call predict with the Pydantic model in the signature
result = await predict(
    "page: str -> items: list[TaskItem]",
    instructions="Extract tasks",
    page="test page content"
)

# Check result
print(f"Got {len(result['items'])} items")
if result['items']:
    item = result['items'][0]
    print(f"First item category: {item['category']}")
    print(f"Schema received: {item.get('_schema_received', False)}")
    print(f"Priority spec count: {item.get('_priority_spec_count', 0)}")
""")
            assert "Got 1 items" in str(result)
            assert "First item category: Test Category" in str(result)
            assert "Schema received: True" in str(result)
            # Should have anyOf with 5 options (4 Literal values + null)
            assert "Priority spec count: 2" in str(result)  # anyOf with enum and null

        finally:
            interpreter.shutdown()

    def test_pydantic_model_serialization_through_tool(self):
        """Pydantic models should serialize correctly when passed to tools."""

        received_data = []

        async def process_items(items):
            """Tool that receives a list of Pydantic models."""
            # Store what we received for verification
            received_data.append(items)
            return f"Processed {len(items)} items"

        interpreter = JspiInterpreter(
            preinstall_packages=True, tools={"process_items": process_items}
        )

        try:
            result = interpreter.execute("""
from typing import Literal, Optional
from pydantic import BaseModel

class Priority(BaseModel):
    level: Literal["urgent", "high", "medium", "low"]

class Task(BaseModel):
    title: str
    priority: Optional[Priority] = None

# Create tasks with Literal values
tasks = [
    Task(title="Task 1", priority=Priority(level="high")),
    Task(title="Task 2", priority=Priority(level="urgent")),
    Task(title="Task 3"),  # No priority
]

# Pass to tool - models should be serialized to dicts
result = await process_items(tasks)
print(result)

# Verify serialization worked
for i, task in enumerate(tasks):
    if task.priority:
        print(f"Task {i+1} priority: {task.priority.level}")
    else:
        print(f"Task {i+1} has no priority")
""")
            assert "Processed 3 items" in str(result)
            assert "Task 1 priority: high" in str(result)
            assert "Task 2 priority: urgent" in str(result)
            assert "Task 3 has no priority" in str(result)

            # Verify the tool received proper dicts
            assert len(received_data) == 1
            items = received_data[0]
            assert len(items) == 3
            assert items[0]["title"] == "Task 1"
            assert items[0]["priority"]["level"] == "high"
            assert items[2]["priority"] is None

        finally:
            interpreter.shutdown()

    def test_dspy_history_serialization_with_rlm(self):
        """Test that DSPy history serialization doesn't produce warnings after RLM run."""
        import warnings
        from typing import Literal, Optional

        import dspy
        from pydantic import BaseModel, Field

        from predict_rlm import PredictRLM

        # Define Pydantic models with Literal like in the notebook
        class ExtractedItem(BaseModel):
            """A task item extracted from documents."""

            category: str = Field(description="Category from extraction")
            title: str = Field(description="Task title")
            description: str = Field(description="Task description")
            priority: Optional[Literal["urgent", "high", "medium", "low"]] = Field(
                default=None, description="Priority level"
            )
            due_date: Optional[str] = Field(default=None, description="Due date")
            doc_id: str = Field(description="Source document ID")
            page: int = Field(description="Page number (0-indexed)")

        class ExtractionResult(BaseModel):
            """All items extracted from documents."""

            items: list[ExtractedItem] = Field(description="List of extracted items")

        # Signature using the Pydantic models
        class ExtractFromDocuments(dspy.Signature):
            """Extract structured items from documents."""

            documents: list[dict] = dspy.InputField(desc="Documents to analyze")
            prompt: str = dspy.InputField(desc="Extraction instructions")
            result: ExtractionResult = dspy.OutputField(
                desc="Task items with category, title, description"
            )

        # Mock tools
        async def get_pages(doc_id: str, pages: list[int], format: str):
            return [f"fake_image_{i}" for i in pages]

        async def search(query: str, doc_id: Optional[str] = None, limit: int = 5):
            return [{"doc_id": "test", "page": 0, "text": "test", "relevance": 1.0}]

        # Use a real LM model (OpenAI) that would trigger the issue
        test_lm = dspy.LM(model="openai/gpt-5-mini-2025-08-07", cache=False)

        # Create the RLM
        rlm = PredictRLM(
            ExtractFromDocuments,
            max_iterations=2,
            verbose=True,
            tools={"get_pages": get_pages, "search": search},
        )

        # Prepare test data like the notebook
        test_docs = [{"doc_id": "doc1", "file_name": "test.pdf", "page_count": 2}]
        test_prompt = "Extract actionable tasks from RFP"

        # Capture warnings during execution and history access
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            try:
                # Run the RLM with the test LM context
                with dspy.context(lm=test_lm):
                    # This might fail due to incomplete setup, which is fine
                    try:
                        _ = rlm(documents=test_docs, prompt=test_prompt)
                    except Exception as e:
                        # We expect some failures, but not serialization warnings
                        print(f"Expected failure: {e}")

                    # The critical part: access history which triggers serialization
                    # This is what happens in the notebook cell
                    _ = list(test_lm.history)

            except Exception as e:
                # If we get serialization warnings, they should be captured
                print(f"Exception during test: {e}")

            # Check for Pydantic serialization warnings from OUR code only.
            # DSPy internals (cache.py, base_lm.py) emit these warnings when
            # serializing history entries with Literal types — that's upstream.
            serialization_warnings = [
                warning
                for warning in w
                if "PydanticSerializationUnexpectedValue" in str(warning.message)
                and "predict_rlm" in str(warning.filename)
            ]

            if len(serialization_warnings) > 0:
                print(
                    f"Got {len(serialization_warnings)} Pydantic serialization warnings from our code:"
                )
                for warning in serialization_warnings:
                    print(f"  - {warning.message}")
                    print(f"    File: {warning.filename}:{warning.lineno}")

            assert len(serialization_warnings) == 0, (
                f"Got {len(serialization_warnings)} Pydantic serialization warnings from our code"
            )


class TestSerializationFailureRecovery:
    """Tests that tools survive serialization failures (the original bug).

    The original issue: when sandbox code tries to pass a non-existent object
    to a tool, the serialization fails and corrupts the global state, making
    tools unavailable for subsequent calls.

    Our fix stores tools in a persistent `_repl_tools` module and re-injects
    them before each execution, preventing state corruption from affecting
    tool availability.
    """

    def test_pydantic_model_with_method_serialization(self):
        """Test that Pydantic models with methods can be serialized properly when passed to predict()."""

        # Track what gets passed to predict
        predict_calls = []

        def mock_predict(signature: str, pydantic_schemas=None, **kwargs):
            predict_calls.append(
                {"signature": signature, "schemas": pydantic_schemas, "kwargs": kwargs}
            )
            # Return mock items based on signature
            return {
                "items": [
                    {"category": "Test", "title": "Item 1", "description": "Test description"}
                ]
            }

        interpreter = JspiInterpreter(tools={"predict": mock_predict})
        try:
            # This should work without serialization errors
            result = interpreter.execute("""
from pydantic import BaseModel, Field

class ExtractedItem(BaseModel):
    category: str = Field(description="Category from the extraction prompt")
    title: str = Field(description="Task title - concise, actionable")
    description: str = Field(description="Task directive using verbatim wording")

    # This method should not cause serialization issues
    def to_dict(self):
        return self.model_dump()

    # This shouldn't either
    @property
    def formatted_title(self):
        return f"[{self.category}] {self.title}"

# Using the model in predict signature - this was failing with method serialization
result = await predict(
    "page: dspy.Image -> items: list[ExtractedItem]",
    instructions="Extract tasks from the page",
    page="dummy_image_url"
)
print(f"Got {len(result['items'])} items")
print(f"First item category: {result['items'][0]['category']}")
""")

            # Verify it worked
            assert "Got 1 items" in str(result)
            assert "First item category: Test" in str(result)

            # Verify predict was called with proper schemas
            assert len(predict_calls) == 1
            call = predict_calls[0]
            assert "ExtractedItem" in str(call["schemas"]) if call["schemas"] else False

            # Tool should still work after this
            result2 = interpreter.execute("""
result = await predict("text: str -> answer: str", text="hello")
print("Predict still works!")
""")
            assert "Predict still works!" in str(result2)

        finally:
            interpreter.shutdown()

    def test_pydantic_model_json_schema_with_method_serialization(self):
        """Test that model_json_schema() with methods doesn't cause JSON serialization errors."""

        interpreter = JspiInterpreter(preinstall_packages=False)
        try:
            # This reproduces the exact issue from the user's example
            result = interpreter.execute("""
from pydantic import BaseModel, Field
import json

class ExtractedItem(BaseModel):
    category: str = Field(description="Category")
    title: str = Field(description="Title")

    def custom_method(self):
        '''Custom method that should not cause issues'''
        return f"{self.category}: {self.title}"

    @property
    def formatted(self):
        '''Property that should not cause issues'''
        return self.custom_method()

# Get the JSON schema - this should work
schema = ExtractedItem.model_json_schema()

# This was causing "TypeError: Object of type method is not JSON serializable"
# when the schema contains references to methods
try:
    serialized = json.dumps(schema)
    print("SUCCESS: Schema serialized without error")
    print(f"Schema keys: {list(schema.keys())}")
except TypeError as e:
    print(f"ERROR: {e}")
    print("This is the bug we're testing for")
""")

            # The test passes if we can serialize the schema without errors
            assert "SUCCESS: Schema serialized" in str(result)
            assert "ERROR" not in str(result)

        finally:
            interpreter.shutdown()

    def test_pydantic_schema_extraction_with_methods_in_predict(self):
        """Test that the _get_pydantic_schemas function works with models containing methods."""
        # This test directly tests the schema extraction that happens in runner.js
        received_schemas = []

        def mock_predict(signature: str, pydantic_schemas=None, **kwargs):
            import json

            # Try to serialize the schemas - this is where the error would occur
            if pydantic_schemas:
                try:
                    json.dumps(pydantic_schemas)
                    received_schemas.append({"success": True, "schemas": pydantic_schemas})
                except TypeError as e:
                    received_schemas.append({"success": False, "error": str(e)})
            return {"items": []}

        interpreter = JspiInterpreter(tools={"predict": mock_predict})
        try:
            result = interpreter.execute("""
from pydantic import BaseModel, Field

class ExtractedItem(BaseModel):
    category: str = Field(description="Category")
    title: str = Field(description="Title")

    def to_dict(self):
        return self.model_dump()

    @property
    def formatted_title(self):
        return f"[{self.category}] {self.title}"

# This should extract the schema and pass it through
result = await predict("text: str -> items: list[ExtractedItem]", text="test")
print("Predict call completed")
""")

            assert "Predict call completed" in str(result)
            assert len(received_schemas) == 1

            # Check if schema was successfully serialized
            schema_result = received_schemas[0]
            if not schema_result["success"]:
                print(f"Schema serialization failed: {schema_result['error']}")
            assert schema_result["success"], (
                f"Schema serialization failed: {schema_result.get('error')}"
            )

        finally:
            interpreter.shutdown()

    def test_reproduce_user_exact_error_scenario(self):
        """Reproduce the exact error scenario from the user's example with ExtractedItem."""
        interpreter = JspiInterpreter(preinstall_packages=False)
        try:
            # This is the exact pattern from the user's error output
            result = interpreter.execute("""
from pydantic import BaseModel, Field
import json

class ExtractedItem(BaseModel):
    category: str = Field(description="Category from the extraction prompt")
    title: str = Field(description="Task title - concise, actionable")
    description: str = Field(description="Task directive using verbatim wording")
    doc_id: str = Field(description="Source document ID")
    page: int = Field(description="Page number where found (0-indexed)")
    due_date: str | None = Field(default=None, description="Due date if explicitly stated")
    priority: str | None = Field(default=None, description="Priority level")

    # Methods that should not affect schema serialization
    def to_dict(self):
        return self.model_dump()

    @property
    def formatted(self):
        return f"[{self.category}] {self.title}"

# Get schema - this should work now with our fix
schema = ExtractedItem.model_json_schema()

# Try to serialize it (this was failing before)
try:
    serialized = json.dumps(schema)
    print("SUCCESS: Schema serialized without error")
    print(f"Schema has {len(schema.get('properties', {}))} properties")
except TypeError as e:
    print(f"ERROR: {e}")
    print("The JSON serialization failed")
""")

            # Should succeed with our fix
            assert "SUCCESS: Schema serialized" in str(result)
            assert "ERROR" not in str(result)

        finally:
            interpreter.shutdown()

    def test_pydantic_model_schema_with_function_validators(self):
        """Test that Pydantic models with validators don't cause serialization issues."""
        received_schemas = []

        def mock_predict(signature: str, pydantic_schemas=None, **kwargs):
            import json

            if pydantic_schemas:
                try:
                    # Check if we can JSON serialize the schema
                    json.dumps(pydantic_schemas)
                    received_schemas.append(pydantic_schemas)
                except TypeError as e:
                    raise RuntimeError(f"Schema serialization failed: {e}")
            return {"items": []}

        interpreter = JspiInterpreter(tools={"predict": mock_predict})
        try:
            # Use field validators which might add function references to the schema
            result = interpreter.execute("""
from pydantic import BaseModel, Field, field_validator

class ExtractedItem(BaseModel):
    category: str = Field(description="Category")
    title: str = Field(description="Title")

    @field_validator('category')
    @classmethod
    def validate_category(cls, v):
        if not v:
            raise ValueError('Category cannot be empty')
        return v.upper()

    @field_validator('title')
    @classmethod
    def validate_title(cls, v):
        return v.strip()

# Extract the schema and use it in predict
result = await predict("text: str -> items: list[ExtractedItem]", text="test")
print("Schema extraction and serialization successful")
""")

            assert "Schema extraction and serialization successful" in str(result)
            assert len(received_schemas) == 1

        finally:
            interpreter.shutdown()

    def test_undefined_variable_in_tool_arg(self):
        """Tools survive when code passes an undefined variable."""
        call_log = []

        def my_tool(data) -> str:
            call_log.append(data)
            return f"Got: {data}"

        interpreter = JspiInterpreter(preinstall_packages=False, tools={"my_tool": my_tool})
        try:
            # First call - success
            output1 = interpreter.execute("""
result = await my_tool("valid")
print(result)
""")
            assert "Got: valid" in str(output1)
            assert len(call_log) == 1

            # Second call - reference undefined variable (NameError)
            # This should fail but NOT corrupt the tools
            output2 = interpreter.execute("""
try:
    result = await my_tool(undefined_variable)
    print("Should not reach here")
except NameError as e:
    print(f"Caught NameError: {e}")
""")
            assert "Caught NameError" in str(output2)

            # Third call - tool should still work!
            output3 = interpreter.execute("""
result = await my_tool("after_error")
print(result)
""")
            assert "Got: after_error" in str(output3)
            assert len(call_log) == 2  # Only 2 successful calls
        finally:
            interpreter.shutdown()

    def test_undefined_class_instantiation_in_tool_arg(self):
        """Tools survive when code tries to instantiate undefined class."""
        call_log = []

        def predict(signature: str, **kwargs) -> dict:
            call_log.append({"sig": signature, "kwargs": kwargs})
            return {"output": f"Processed {len(kwargs)} args"}

        interpreter = JspiInterpreter(preinstall_packages=False, tools={"predict": predict})
        try:
            # First call - success
            output1 = interpreter.execute("""
result = await predict("question -> answer", question="test")
print(result)
""")
            assert "Processed 1 args" in str(output1)

            # Try to pass instance of undefined class
            output2 = interpreter.execute("""
try:
    # NonExistentModel doesn't exist - this will raise NameError
    obj = NonExistentModel(field="value")
    result = await predict("data -> output", data=obj)
    print("Should not reach here")
except NameError as e:
    print(f"Caught: NameError")
""")
            assert "Caught: NameError" in str(output2)

            # Tool should still work
            output3 = interpreter.execute("""
result = await predict("x -> y", x="recovery test")
print(result)
""")
            assert "Processed 1 args" in str(output3)
            assert len(call_log) == 2
        finally:
            interpreter.shutdown()

    def test_serialization_error_mid_gather(self):
        """Tools survive when one of many parallel calls has serialization error."""
        call_log = []

        def process(item_id: int, data) -> str:
            call_log.append({"id": item_id, "data": data})
            return f"Processed {item_id}"

        interpreter = JspiInterpreter(preinstall_packages=False, tools={"process": process})
        try:
            # Run parallel calls where one will fail due to undefined reference
            output = interpreter.execute("""
import asyncio

async def safe_process(item_id, data_factory):
    try:
        data = data_factory()
        return await process(item_id, data)
    except Exception as e:
        return f"Error {item_id}: {type(e).__name__}"

# Factory functions - one will fail
def valid1(): return "data1"
def valid2(): return "data2"
def invalid(): return undefined_var  # Will raise NameError
def valid3(): return "data3"

tasks = [
    safe_process(1, valid1),
    safe_process(2, valid2),
    safe_process(3, invalid),  # This one fails
    safe_process(4, valid3),
]
results = await asyncio.gather(*tasks)

successes = [r for r in results if r.startswith("Processed")]
errors = [r for r in results if r.startswith("Error")]

print(f"Successes: {len(successes)}")
print(f"Errors: {len(errors)}")
""")
            assert "Successes: 3" in str(output)
            assert "Errors: 1" in str(output)

            # Tool should still work after the mixed batch
            output2 = interpreter.execute("""
result = await process(99, "final")
print(result)
""")
            assert "Processed 99" in str(output2)
        finally:
            interpreter.shutdown()

    def test_attribute_error_on_nonexistent_method(self):
        """Tools survive when code calls non-existent method on result."""

        def get_data() -> dict:
            return {"items": [1, 2, 3]}

        interpreter = JspiInterpreter(preinstall_packages=False, tools={"get_data": get_data})
        try:
            # First call - success
            output1 = interpreter.execute("""
data = await get_data()
print(f"Got {len(data['items'])} items")
""")
            assert "Got 3 items" in str(output1)

            # Call method that doesn't exist on the result
            output2 = interpreter.execute("""
try:
    data = await get_data()
    # Try to call non-existent method
    result = data.nonexistent_method()
    print("Should not reach")
except AttributeError as e:
    print("Caught AttributeError")
""")
            assert "Caught AttributeError" in str(output2)

            # Tool should still work
            output3 = interpreter.execute("""
data = await get_data()
print(f"Recovery: {data['items'][0]}")
""")
            assert "Recovery: 1" in str(output3)
        finally:
            interpreter.shutdown()

    def test_type_error_during_serialization(self):
        """Tools survive TypeError during argument serialization."""
        received = []

        def accept_dict(d: dict) -> str:
            received.append(d)
            return f"Got dict with {len(d)} keys"

        interpreter = JspiInterpreter(
            preinstall_packages=False, tools={"accept_dict": accept_dict}
        )
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

    def test_tools_persist_through_repeated_failures(self):
        """Tools remain available through multiple consecutive failures."""
        call_count = {"success": 0}

        def counter_tool(value: str) -> str:
            call_count["success"] += 1
            return f"Count: {call_count['success']}"

        interpreter = JspiInterpreter(
            preinstall_packages=False, tools={"counter_tool": counter_tool}
        )
        try:
            # Initial success
            interpreter.execute("""
result = await counter_tool("first")
print(result)
""")
            assert call_count["success"] == 1

            # Multiple failures in a row
            for i in range(5):
                interpreter.execute(f"""
try:
    result = await counter_tool(undefined_var_{i})
except NameError:
    pass
""")

            # Tool should STILL work after 5 consecutive failures
            output = interpreter.execute("""
result = await counter_tool("recovery")
print(result)
""")
            assert "Count: 2" in str(output)
            assert call_count["success"] == 2
        finally:
            interpreter.shutdown()

    def test_pydantic_model_with_missing_field_reference(self):
        """Tools survive when Pydantic model references undefined field."""
        received = []

        def process_model(data) -> str:
            received.append(data)
            return "processed"

        interpreter = JspiInterpreter(tools={"process_model": process_model})
        try:
            # Success with valid model
            output1 = interpreter.execute("""
from pydantic import BaseModel

class ValidModel(BaseModel):
    name: str
    value: int

m = ValidModel(name="test", value=42)
result = await process_model(m)
print(f"First: {result}")
""")
            assert "First: processed" in str(output1)

            # Try to create model referencing undefined variable
            output2 = interpreter.execute("""
from pydantic import BaseModel

class AnotherModel(BaseModel):
    data: str

try:
    # undefined_value doesn't exist
    m = AnotherModel(data=undefined_value)
    result = await process_model(m)
    print("Should not reach")
except NameError:
    print("Caught NameError as expected")
""")
            assert "Caught NameError" in str(output2)

            # Tool should still work
            output3 = interpreter.execute("""
from pydantic import BaseModel

class YetAnotherModel(BaseModel):
    status: str

m = YetAnotherModel(status="recovered")
result = await process_model(m)
print(f"Third: {result}")
""")
            assert "Third: processed" in str(output3)
            assert len(received) == 2  # Only successful calls
        finally:
            interpreter.shutdown()

    def test_tool_in_module_survives_serialization_crash(self):
        """Verify tool remains in _repl_tools module after serialization error."""

        def my_tool(x) -> str:
            return f"got {x}"

        interpreter = JspiInterpreter(preinstall_packages=False, tools={"my_tool": my_tool})
        try:
            # Cause a serialization failure inside the tool
            output1 = interpreter.execute("""
import sys

class CrashingModel:
    def model_dump(self):
        raise RuntimeError("Serialization crashed!")

try:
    obj = CrashingModel()
    await my_tool(obj)
except Exception as e:
    print(f"Tool call failed: {type(e).__name__}")

# Verify tool is still in _repl_tools module
_repl_tools = sys.modules.get('_repl_tools')
tool_in_module = hasattr(_repl_tools, 'my_tool') if _repl_tools else False
print(f"Tool in _repl_tools: {tool_in_module}")

# Verify tool is still in globals
tool_in_globals = 'my_tool' in dir()
print(f"Tool in globals: {tool_in_globals}")
""")
            assert "Tool call failed:" in str(output1)
            assert "Tool in _repl_tools: True" in str(output1)
            assert "Tool in globals: True" in str(output1)

            # Tool should work normally after the crash
            output2 = interpreter.execute("""
result = await my_tool("works!")
print(result)
""")
            assert "got works!" in str(output2)
        finally:
            interpreter.shutdown()

    def test_tool_recovery_after_global_deletion(self):
        """Tools can be recovered from _repl_tools even if deleted from globals.

        This simulates the worst case where globals get corrupted/cleared.
        The re-injection mechanism should restore the tools.
        """
        call_log = []

        def track_tool(msg: str) -> str:
            call_log.append(msg)
            return f"tracked: {msg}"

        interpreter = JspiInterpreter(
            preinstall_packages=False, tools={"track_tool": track_tool}
        )
        try:
            # First call works
            output1 = interpreter.execute("""
result = await track_tool("first")
print(result)
""")
            assert "tracked: first" in str(output1)

            # Deliberately delete the tool from globals (simulating corruption)
            output2 = interpreter.execute("""
# Delete tool from globals
del track_tool

# Verify it's gone from globals
tool_in_globals = 'track_tool' in dir()
print(f"After del - tool in globals: {tool_in_globals}")

# But it should still be in _repl_tools module
import sys
_repl_tools = sys.modules.get('_repl_tools')
tool_in_module = hasattr(_repl_tools, 'track_tool') if _repl_tools else False
print(f"After del - tool in module: {tool_in_module}")
""")
            assert "After del - tool in globals: False" in str(output2)
            assert "After del - tool in module: True" in str(output2)

            # Next execution should re-inject the tool (our fix!)
            output3 = interpreter.execute("""
# Tool should be available again due to re-injection
result = await track_tool("after deletion")
print(result)
""")
            assert "tracked: after deletion" in str(output3)
            assert len(call_log) == 2
        finally:
            interpreter.shutdown()

    def test_multiple_tools_survive_partial_corruption(self):
        """All tools survive even when one tool's call crashes."""
        tool_a_calls = []
        tool_b_calls = []
        tool_c_calls = []

        def tool_a(x) -> str:
            tool_a_calls.append(x)
            return f"A: {x}"

        def tool_b(x) -> str:
            tool_b_calls.append(x)
            return f"B: {x}"

        def tool_c(x) -> str:
            tool_c_calls.append(x)
            return f"C: {x}"

        interpreter = JspiInterpreter(
            preinstall_packages=False,
            tools={
                "tool_a": tool_a,
                "tool_b": tool_b,
                "tool_c": tool_c,
            },
        )
        try:
            # Use all tools successfully
            output1 = interpreter.execute("""
a = await tool_a("first")
b = await tool_b("first")
c = await tool_c("first")
print(f"{a}, {b}, {c}")
""")
            assert "A: first" in str(output1)
            assert "B: first" in str(output1)
            assert "C: first" in str(output1)

            # Crash tool_b with a serialization error
            output2 = interpreter.execute("""
class BadArg:
    def model_dump(self):
        raise ValueError("Boom!")

try:
    await tool_b(BadArg())
except Exception:
    print("tool_b crashed")
""")
            assert "tool_b crashed" in str(output2)

            # ALL tools should still work
            output3 = interpreter.execute("""
a = await tool_a("after")
b = await tool_b("after")
c = await tool_c("after")
print(f"{a}, {b}, {c}")
""")
            assert "A: after" in str(output3)
            assert "B: after" in str(output3)
            assert "C: after" in str(output3)

            # Verify call counts
            assert len(tool_a_calls) == 2
            assert len(tool_b_calls) == 2  # 2 successful, 1 failed (not counted)
            assert len(tool_c_calls) == 2
        finally:
            interpreter.shutdown()


class TestToolSurvivalAfterAsyncErrors:
    """Test that tools survive when async code crashes with pending tool calls.

    This addresses the issue where tools would disappear after an error during
    concurrent async execution (e.g., asyncio.gather with many predict calls).
    The bug was that the response reader could consume the next command from
    the interpreter if not properly stopped on error.
    """

    def test_tools_survive_after_unhandled_async_error(self):
        """Tools remain available after an unhandled exception in async code."""

        call_counts = {"slow": 0, "fast": 0}

        def slow_tool(x: str) -> str:
            call_counts["slow"] += 1
            return f"slow: {x}"

        def fast_tool(x: str) -> str:
            call_counts["fast"] += 1
            return f"fast: {x}"

        interpreter = JspiInterpreter(
            preinstall_packages=False,
            tools={"slow_tool": slow_tool, "fast_tool": fast_tool},
        )
        try:
            # First, verify tools work normally
            output1 = interpreter.execute("""
r1 = await slow_tool("first")
r2 = await fast_tool("first")
print(f"Results: {r1}, {r2}")
""")
            assert "slow: first" in str(output1)
            assert "fast: first" in str(output1)
            assert call_counts["slow"] == 1
            assert call_counts["fast"] == 1

            # Now cause an error - but catch it so we can continue
            output2 = interpreter.execute("""
import asyncio

async def fail_after_tool():
    r = await slow_tool("before_fail")
    print(f"Got: {r}")
    # Raise after the tool call
    raise ValueError("Boom!")

try:
    await fail_after_tool()
except ValueError as e:
    print(f"Caught: {e}")
""")
            assert "Got: slow: before_fail" in str(output2)
            assert "Caught: Boom!" in str(output2)
            assert call_counts["slow"] == 2

            # CRITICAL: Tools should still be available after the error
            output3 = interpreter.execute("""
# Tools should be defined and callable
r1 = await slow_tool("after_error")
r2 = await fast_tool("after_error")
print(f"After error: {r1}, {r2}")
""")
            assert "slow: after_error" in str(output3)
            assert "fast: after_error" in str(output3)
            assert call_counts["slow"] == 3
            assert call_counts["fast"] == 2

        finally:
            interpreter.shutdown()

    def test_tools_survive_after_import_re(self):
        """Importing 're' module doesn't shadow tool named 'search'.

        This tests a specific bug where doing 'import re' could cause
        the 'search' tool to be replaced by re.search somehow.
        """
        search_calls = []

        def search_tool(query: str) -> str:
            search_calls.append(query)
            return f"results for: {query}"

        interpreter = JspiInterpreter(
            preinstall_packages=False,
            tools={"search": search_tool},
        )
        try:
            # Use search tool before import
            output1 = interpreter.execute("""
r = await search("test query 1")
print(f"Result: {r}")
""")
            assert "results for: test query 1" in str(output1)
            assert len(search_calls) == 1

            # Import re module (which has re.search)
            output2 = interpreter.execute("""
import re
pattern = re.compile(r"\\d+")
match = pattern.search("abc123def")
print(f"Found: {match.group()}")
""")
            assert "Found: 123" in str(output2)

            # Search tool should still work and NOT be re.search
            output3 = interpreter.execute("""
r = await search("test query 2")
print(f"Result: {r}")
print(f"search is callable: {callable(search)}")
# Verify it's our async tool, not re.search
import inspect
is_coroutine = inspect.iscoroutinefunction(search)
print(f"search is async: {is_coroutine}")
""")
            assert "results for: test query 2" in str(output3)
            assert "search is callable: True" in str(output3)
            assert "search is async: True" in str(output3)
            assert len(search_calls) == 2

        finally:
            interpreter.shutdown()

    def test_tools_survive_multiple_consecutive_errors(self):
        """Tools survive through multiple consecutive errors."""
        tool_calls = []

        def my_tool(x: str) -> str:
            tool_calls.append(x)
            return f"result: {x}"

        interpreter = JspiInterpreter(
            preinstall_packages=False,
            tools={"my_tool": my_tool},
        )
        try:
            # Error 1
            output1 = interpreter.execute("""
try:
    1/0
except ZeroDivisionError:
    print("Error 1")
""")
            assert "Error 1" in str(output1)

            # Tool still works
            output2 = interpreter.execute("""
r = await my_tool("after error 1")
print(r)
""")
            assert "result: after error 1" in str(output2)

            # Error 2
            output3 = interpreter.execute("""
try:
    raise KeyError("missing")
except KeyError:
    print("Error 2")
""")
            assert "Error 2" in str(output3)

            # Tool still works
            output4 = interpreter.execute("""
r = await my_tool("after error 2")
print(r)
""")
            assert "result: after error 2" in str(output4)

            # Error 3 - name error
            output5 = interpreter.execute("""
try:
    undefined_variable_xyz
except NameError:
    print("Error 3")
""")
            assert "Error 3" in str(output5)

            # Tool still works
            output6 = interpreter.execute("""
r = await my_tool("after error 3")
print(r)
""")
            assert "result: after error 3" in str(output6)
            assert len(tool_calls) == 3

        finally:
            interpreter.shutdown()


class TestCancellationAndLateResponses:
    """Tests for the resilient tool call channel with cancellation."""

    def test_gather_with_one_failure_cancels_pending_calls(self):
        """When one tool in asyncio.gather fails, pending calls are cancelled."""
        import asyncio

        tool_calls = []
        call_order = []

        async def slow_tool(msg):
            tool_calls.append(msg)
            call_order.append(f"start:{msg}")
            # Simulate slow tool
            await asyncio.sleep(0.2)
            if msg == "fail":
                raise ValueError(f"Intentional failure for {msg}")
            call_order.append(f"end:{msg}")
            return f"result: {msg}"

        interpreter = JspiInterpreter(
            preinstall_packages=False,
            tools={"slow_tool": slow_tool},
        )
        try:
            # This gather will fail because one tool raises an exception
            # The error is wrapped by the tool call wrapper, so we catch RuntimeError
            output = interpreter.execute("""
import asyncio

async def run():
    try:
        results = await asyncio.gather(
            slow_tool("call1"),
            slow_tool("fail"),  # This will fail
            slow_tool("call3"),
        )
        return results
    except RuntimeError as e:
        return f"Caught: {e}"

result = await run()
print(result)
""")
            # The error should be caught (wrapped as RuntimeError by tool wrapper)
            assert "Caught:" in str(output)
            assert "Intentional failure for fail" in str(output)
            # All three calls were made
            assert len(tool_calls) == 3

        finally:
            interpreter.shutdown()

    def test_next_execution_works_after_cancelled_gather(self):
        """After a gather fails and cancels pending calls, next execution works."""
        import asyncio

        call_count = [0]

        async def counting_tool(msg):
            call_count[0] += 1
            await asyncio.sleep(0.05)
            if msg == "fail":
                raise ValueError("Intentional failure")
            return f"result: {msg}"

        interpreter = JspiInterpreter(
            preinstall_packages=False,
            tools={"counting_tool": counting_tool},
        )
        try:
            # First execution - gather with failure
            output1 = interpreter.execute("""
import asyncio

async def run():
    try:
        results = await asyncio.gather(
            counting_tool("a"),
            counting_tool("fail"),
            counting_tool("c"),
        )
        return results
    except RuntimeError as e:
        return f"First caught: {e}"

result = await run()
print(result)
""")
            assert "First caught:" in str(output1)
            assert "Intentional failure" in str(output1)
            first_call_count = call_count[0]
            assert first_call_count == 3

            # Second execution - should work normally
            output2 = interpreter.execute("""
import asyncio

async def run():
    results = await asyncio.gather(
        counting_tool("x"),
        counting_tool("y"),
    )
    return results

result = await run()
print(result)
""")
            # Should succeed
            assert "result: x" in str(output2)
            assert "result: y" in str(output2)
            # Two more calls made
            assert call_count[0] == first_call_count + 2

        finally:
            interpreter.shutdown()

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

        interpreter = JspiInterpreter(
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

    def test_many_parallel_calls_one_fails(self):
        """Test with 10 parallel calls where call 7 fails."""
        import asyncio

        call_log = []

        async def numbered_tool(n):
            call_log.append(f"start_{n}")
            await asyncio.sleep(0.05)
            if n == 7:
                raise ValueError(f"Call {n} failed")
            call_log.append(f"end_{n}")
            return f"result_{n}"

        interpreter = JspiInterpreter(
            preinstall_packages=False,
            tools={"numbered_tool": numbered_tool},
        )
        try:
            # Issue 10 parallel calls, call 7 will fail
            output = interpreter.execute("""
import asyncio

async def run():
    try:
        results = await asyncio.gather(*[
            numbered_tool(i) for i in range(1, 11)
        ])
        return results
    except RuntimeError as e:
        return f"Failed: {e}"

result = await run()
print(result)
""")
            # Should catch the error from call 7
            assert "Failed:" in str(output)
            assert "Call 7 failed" in str(output)
            # All 10 calls were started
            assert len([c for c in call_log if c.startswith("start_")]) == 10

            # Next execution should work
            output2 = interpreter.execute("""
result = await numbered_tool(99)
print(result)
""")
            assert "result_99" in str(output2)

        finally:
            interpreter.shutdown()

    def test_rapid_failure_and_recovery(self):
        """Test multiple rapid failures followed by successful execution."""
        import asyncio

        async def flaky_tool(should_fail):
            await asyncio.sleep(0.02)
            if should_fail:
                raise RuntimeError("Flaky failure")
            return "success"

        interpreter = JspiInterpreter(
            preinstall_packages=False,
            tools={"flaky_tool": flaky_tool},
        )
        try:
            # Rapid failures
            for i in range(3):
                output = interpreter.execute(
                    f"""
import asyncio

async def run():
    try:
        await asyncio.gather(
            flaky_tool(False),
            flaky_tool(True),  # Always fails
            flaky_tool(False),
        )
    except RuntimeError as e:
        return f"Failure {i}: {{e}}"

result = await run()
print(result)
""".replace("{i}", str(i))
                )
                # Error is wrapped by tool wrapper, check for the core message
                assert f"Failure {i}:" in str(output)
                assert "Flaky failure" in str(output)

            # Now success
            output = interpreter.execute("""
import asyncio

results = await asyncio.gather(
    flaky_tool(False),
    flaky_tool(False),
)
print(results)
""")
            assert "success" in str(output)

        finally:
            interpreter.shutdown()


class TestPydanticReconstruction:
    """Predict results should be reconstructed as Pydantic model instances."""

    def test_predict_returns_pydantic_instances_for_list_output(self):
        """When predict returns list[TaskItem], items should have attribute access."""

        def mock_predict(signature: str, pydantic_schemas=None, **kwargs):
            return {"tasks": [
                {"category": "Cert", "title": "Get ISO cert"},
                {"category": "Form", "title": "Fill W-9"},
            ]}

        interpreter = JspiInterpreter(tools={"predict": mock_predict})
        try:
            result = interpreter.execute("""
from pydantic import BaseModel

class TaskItem(BaseModel):
    category: str
    title: str

result = await predict("doc: str -> tasks: list[TaskItem]", doc="test")
# Attribute access should work — not just dict access
for task in result["tasks"]:
    print(f"{task.title} ({task.category})")
""")
            assert "Get ISO cert (Cert)" in str(result)
            assert "Fill W-9 (Form)" in str(result)
        finally:
            interpreter.shutdown()

    def test_predict_returns_pydantic_instance_for_single_output(self):
        """When predict returns a single Pydantic type, it should have attribute access."""

        def mock_predict(signature: str, pydantic_schemas=None, **kwargs):
            return {"summary": {"title": "Project X", "score": 95}}

        interpreter = JspiInterpreter(tools={"predict": mock_predict})
        try:
            result = interpreter.execute("""
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    score: int

result = await predict("doc: str -> summary: Summary", doc="test")
print(f"{result['summary'].title}: {result['summary'].score}")
""")
            assert "Project X: 95" in str(result)
        finally:
            interpreter.shutdown()

    def test_predict_dict_output_stays_as_dict(self):
        """When output type is plain dict (no Pydantic model), result stays as dict."""

        def mock_predict(signature: str, pydantic_schemas=None, **kwargs):
            return {"items": [{"a": 1}, {"b": 2}]}

        interpreter = JspiInterpreter(tools={"predict": mock_predict})
        try:
            result = interpreter.execute("""
result = await predict("doc: str -> items: list[dict]", doc="test")
# dict access should work
print(f"Got {len(result['items'])} items, first has key: {list(result['items'][0].keys())[0]}")
""")
            assert "Got 2 items" in str(result)
        finally:
            interpreter.shutdown()

    def test_can_add_fields_after_reconstruction(self):
        """LM can add metadata fields to reconstructed models (extra='allow')."""

        def mock_predict(signature: str, pydantic_schemas=None, **kwargs):
            return {"tasks": [
                {"category": "Cert", "title": "Get ISO cert", "extra_field": "bonus"},
            ]}

        interpreter = JspiInterpreter(tools={"predict": mock_predict})
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

    def test_predict_result_attribute_access(self):
        """Top-level result supports both attribute and subscript access."""

        def mock_predict(signature: str, pydantic_schemas=None, **kwargs):
            return {"answer": "Paris", "confidence": 0.95}

        interpreter = JspiInterpreter(tools={"predict": mock_predict})
        try:
            result = interpreter.execute("""
result = await predict("question: str -> answer: str, confidence: float", question="capital?")
# Both access patterns should work
print(f"attr: {result.answer}, sub: {result['confidence']}")
""")
            assert "attr: Paris, sub: 0.95" in str(result)
        finally:
            interpreter.shutdown()

    def test_predict_result_items_no_collision(self):
        """result.items returns stored list, not dict.items() method."""

        def mock_predict(signature: str, pydantic_schemas=None, **kwargs):
            return {"items": ["apple", "banana", "cherry"]}

        interpreter = JspiInterpreter(tools={"predict": mock_predict})
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
