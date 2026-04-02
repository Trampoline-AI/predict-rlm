"""
Tests verifying tools survive long RLM traces (bug reproduction validated).

BACKGROUND:
During long RLM traces with many iterations and parallel tool calls, tools
would sometimes become unavailable (NameError: 'predict' is not defined).

BUG REPRODUCTION:
The bug was successfully reproduced by creating a version of runner.js WITHOUT
the following fixes:
1. No `_repl_tools` module persistence
2. No re-injection of tools before each execution
3. No `await responseReaderPromise` after code execution

With the unfixed runner:
- Single iteration: PASSES
- Multiple iterations: HANGS on second iteration

The hang occurs because `responseReader` is still blocked on `stdin.next()`
when the main loop tries to read the next code block. The `responseReader`
"steals" the code block (treating it as a tool response), causing deadlock.

FIXES APPLIED (in runner.js):
1. Store tools in `_repl_tools` module in `sys.modules` (survives globals corruption)
2. Re-inject tools before each execution from `registeredTools` array
3. Always `await responseReaderPromise` after code execution to ensure clean handoff

These tests verify the fixes work correctly by simulating the production pattern:
1. Many iterations (like RLM's 15-20 iterations)
2. Parallel tool calls via asyncio.gather()
3. Exceptions that are caught inside the sandbox code
4. Checks that tools remain available after errors
"""

import pytest

from predict_rlm.interpreter import JspiInterpreter

pytestmark = pytest.mark.integration


class TestToolsExistInNamespace:
    """Tests that explicitly check if tools exist in the sandbox namespace."""

    def test_tools_in_globals_across_iterations(self):
        """
        Explicitly check if tools exist in globals() across iterations.
        This is the actual bug pattern - tools disappear from namespace.
        """
        call_count = 0

        async def predict(item: str) -> dict:
            nonlocal call_count
            call_count += 1
            return {"item": item, "count": call_count}

        interp = JspiInterpreter(
            tools={"predict": predict},
            preinstall_packages=False,
        )

        try:
            for i in range(20):
                # Explicitly check if 'predict' exists in namespace BEFORE calling it
                code = f"""
import sys

# Check multiple places where tool might exist
in_globals = 'predict' in globals()
in_dir = 'predict' in dir()
in_repl_tools = '_repl_tools' in sys.modules and hasattr(sys.modules['_repl_tools'], 'predict')

if not in_globals and not in_dir:
    if in_repl_tools:
        print(f"Iteration {i}: TOOLS_LOST_FROM_GLOBALS but recoverable from _repl_tools")
    else:
        print(f"Iteration {i}: TOOLS_COMPLETELY_LOST")
        SUBMIT("TOOLS_LOST")
else:
    # Tools exist, use them
    result = await predict("item_{i}")
    print(f"Iteration {i}: predict returned {{result}}")
"""
                result = interp.execute(code)
                output = str(result)

                if "TOOLS_COMPLETELY_LOST" in output:
                    pytest.fail(f"BUG REPRODUCED: Tools completely lost at iteration {i}!")

                if "TOOLS_LOST_FROM_GLOBALS" in output:
                    print(f"Note: Tools lost from globals but recovered at iteration {i}")

            print(f"All {call_count} iterations completed with tools intact")

        finally:
            interp.shutdown()

    def test_tools_after_heavy_async_corruption(self):
        """
        Try to corrupt globals with heavy async operations, then check tool existence.
        """
        call_count = 0

        async def predict(item: str) -> dict:
            import asyncio

            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return {"item": item}

        interp = JspiInterpreter(
            tools={"predict": predict},
            preinstall_packages=False,
        )

        try:
            for i in range(10):
                # Heavy async work that might corrupt state
                code = f"""
import asyncio
import sys

# Do lots of async work
async def heavy_work():
    tasks = [predict(f"item_{{j}}") for j in range(5)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Trigger some errors
    try:
        x = undefined_var_{i}
    except NameError:
        pass

    return results

results = await heavy_work()
print(f"Iteration {i}: got {{len(results)}} results")

# Now check if tools still exist
in_globals = 'predict' in globals()
in_repl_tools = '_repl_tools' in sys.modules and hasattr(sys.modules['_repl_tools'], 'predict')

if not in_globals:
    if in_repl_tools:
        print(f"GLOBALS_CORRUPTED_BUT_RECOVERED")
    else:
        print(f"TOOLS_COMPLETELY_LOST")
        SUBMIT("TOOLS_LOST")
"""
                result = interp.execute(code)
                output = str(result)

                if "TOOLS_COMPLETELY_LOST" in output:
                    pytest.fail(
                        f"BUG REPRODUCED: Tools lost after heavy async at iteration {i}!"
                    )

            print(f"All iterations completed, {call_count} total predict calls")

        finally:
            interp.shutdown()


class TestToolsSurviveExceptions:
    """Tests that tools survive when exceptions are caught inside the sandbox."""

    def test_tools_available_after_caught_exception(self):
        """
        Reproduce: Tool call succeeds, then exception is caught, tools still work.

        This simulates the RLM pattern where exceptions are caught and handled.
        """
        results = []

        async def slow_tool(value: str) -> str:
            results.append(f"slow_tool called with: {value}")
            return f"processed: {value}"

        interp = JspiInterpreter(
            tools={"slow_tool": slow_tool},
            preinstall_packages=False,
        )

        try:
            # Iteration 1: Call tool, catch exception
            code1 = """
import asyncio

# Call the tool successfully
result = await slow_tool("iteration1")
print(f"Got: {result}")

# Raise and catch an exception
try:
    raise ValueError("Model returned invalid response")
except ValueError as e:
    print(f"Caught error: {e}")
"""
            result1 = interp.execute(code1)
            assert "Got: processed: iteration1" in str(result1)
            assert "Caught error" in str(result1)

            # Verify tool was called
            assert len(results) == 1

            # Iteration 2: Tools should still work
            code2 = """
try:
    result = await slow_tool("iteration2")
    print(f"SUCCESS: {result}")
except NameError as e:
    print(f"TOOLS_LOST: {e}")
"""
            result2 = interp.execute(code2)
            output = str(result2)

            if "TOOLS_LOST" in output:
                pytest.fail(
                    f"BUG REPRODUCED: Tools were lost after exception! Output: {output}"
                )

            assert "SUCCESS" in output
            assert len(results) == 2

        finally:
            interp.shutdown()

    def test_rapid_iterations_with_errors(self):
        """
        Multiple rapid iterations with errors - stress test.
        """
        call_count = 0

        async def count_tool() -> int:
            nonlocal call_count
            call_count += 1
            return call_count

        interp = JspiInterpreter(
            tools={"count_tool": count_tool},
            preinstall_packages=False,
        )

        try:
            for i in range(10):
                code = f"""
try:
    result = await count_tool()
    print(f"Iteration {i}: count = {{result}}")
    # Raise and catch an error
    raise RuntimeError("Iteration {i} error")
except RuntimeError as e:
    print(f"Caught: {{e}}")
"""
                result = interp.execute(code)
                assert f"Iteration {i}" in str(result)
                assert "Caught" in str(result)

            # Final check: tools should still work
            final_code = """
try:
    result = await count_tool()
    print(f"FINAL: count = {result}")
except NameError as e:
    print(f"TOOLS_LOST: {e}")
"""
            final_result = interp.execute(final_code)
            output = str(final_result)

            if "TOOLS_LOST" in output:
                pytest.fail(f"BUG REPRODUCED: Tools lost after {call_count} iterations!")

            assert "FINAL" in output
            assert call_count == 11, f"Expected 11 calls, got {call_count}"

        finally:
            interp.shutdown()

    def test_parallel_tool_calls_with_error_processing(self):
        """
        Exception occurs while processing results from parallel tool calls.
        This is the most likely scenario for the production bug.
        """
        call_log = []

        async def predict(item: str) -> dict:
            import asyncio

            await asyncio.sleep(0.01)  # Simulate async work
            call_log.append(f"predict({item})")
            return {"item": item, "value": f"extracted_{item}"}

        interp = JspiInterpreter(
            tools={"predict": predict},
            preinstall_packages=False,
        )

        try:
            # Pattern from RLM: parallel predict calls, then error processing
            code1 = """
import asyncio

# Start multiple predict calls
tasks = [
    predict("item1"),
    predict("item2"),
    predict("item3"),
]

# Gather them
results = await asyncio.gather(*tasks)
print(f"Got {len(results)} results")

# Error processing the results (caught)
try:
    x = results[0]["nonexistent_field"]  # KeyError
except KeyError as e:
    print(f"Caught KeyError: {e}")
"""
            result1 = interp.execute(code1)
            assert "Got 3 results" in str(result1)
            assert "Caught KeyError" in str(result1)

            # Check that tools were called
            assert len(call_log) == 3

            # Now try next iteration - tools might be lost!
            code2 = """
try:
    result = await predict("iteration2_item")
    print(f"SUCCESS: {result}")
except NameError as e:
    print(f"TOOLS_LOST: {e}")
"""
            result2 = interp.execute(code2)
            output = str(result2)

            if "TOOLS_LOST" in output:
                pytest.fail("BUG REPRODUCED: Tools lost after parallel calls + exception!")

            assert "SUCCESS" in output
            assert len(call_log) == 4

        finally:
            interp.shutdown()


class TestProductionScenarioReproduction:
    """
    Reproduce the exact production scenario from dspy-playground.py
    """

    def test_rlm_extractor_pattern_20_iterations(self):
        """
        Simulate the actual RLM extractor pattern:
        1. 20 iterations (like real RLM)
        2. Each iteration calls predict() and other tools
        3. Some iterations have processing errors (caught)
        4. Tools should survive all iterations
        """
        iteration_count = 0

        async def predict(extraction_type: str) -> dict:
            nonlocal iteration_count
            iteration_count += 1
            return {
                "items": [{"name": f"item_{iteration_count}", "value": iteration_count * 10}]
            }

        async def search(query: str) -> list:
            return [{"page": 1, "text": f"Found: {query}"}]

        async def get_pages() -> list:
            return [{"page": 1}, {"page": 2}]

        interp = JspiInterpreter(
            tools={"predict": predict, "search": search, "get_pages": get_pages},
            preinstall_packages=False,
        )

        tools_lost_at = None

        try:
            for i in range(20):
                if i % 3 == 2:
                    # Error iteration
                    code = f"""
import asyncio

# Call predict
result = await predict("iteration_{i}")
print(f"Iteration {i}: predict returned {{len(result.get('items', []))}} items")

# Simulate processing error (caught)
try:
    bad_value = result["items"][0]["nonexistent_field"]
except KeyError as e:
    print(f"Caught KeyError: {{e}}")
"""
                else:
                    # Normal iteration with multiple tool calls
                    code = f"""
import asyncio

# Multiple tool calls
try:
    results = await asyncio.gather(
        predict("iteration_{i}"),
        search("query_{i}"),
    )
    print(f"Iteration {i}: predict={{results[0]}}, search={{results[1]}}")

    pages = await get_pages()
    print(f"Pages: {{pages}}")
except NameError as e:
    print(f"TOOLS_LOST at iteration {i}: {{e}}")
"""

                result = interp.execute(code)
                output = str(result)

                # Check for tools lost
                if "TOOLS_LOST" in output:
                    tools_lost_at = i
                    break

            if tools_lost_at is not None:
                pytest.fail(f"BUG REPRODUCED: Tools lost at iteration {tools_lost_at}!")

            # Final verification
            final_code = """
try:
    p = await predict("final")
    s = await search("final")
    g = await get_pages()
    print(f"FINAL SUCCESS: predict={p}, search={s}, pages={g}")
except NameError as e:
    print(f"TOOLS_LOST: {e}")
"""
            final_result = interp.execute(final_code)
            output = str(final_result)

            if "TOOLS_LOST" in output:
                pytest.fail("BUG REPRODUCED: Tools lost after 20 iterations!")

            assert "FINAL SUCCESS" in output
            print(f"All tools survived {iteration_count} predict calls across 20 iterations")

        finally:
            interp.shutdown()

    def test_heavy_parallel_load(self):
        """
        Heavy parallel load - many concurrent predict calls per iteration.
        This stress tests the concurrent tool call handling.
        """
        call_count = 0

        async def predict(item: dict) -> dict:
            import asyncio

            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)  # Simulate latency
            return {"extracted": item.get("name", "unknown"), "count": call_count}

        interp = JspiInterpreter(
            tools={"predict": predict},
            preinstall_packages=False,
        )

        try:
            for iteration in range(5):
                # Each iteration does 10 parallel predict calls
                code = f"""
import asyncio

items = [dict(name=f"item_{{i}}", iteration={iteration}) for i in range(10)]

try:
    results = await asyncio.gather(*[predict(item) for item in items])
    print(f"Iteration {iteration}: got {{len(results)}} results")

    # Processing might fail
    if {iteration} == 2:
        raise ValueError("Simulated processing error")
except ValueError as e:
    print(f"Caught: {{e}}")
except NameError as e:
    print(f"TOOLS_LOST at iteration {iteration}: {{e}}")
"""
                result = interp.execute(code)
                output = str(result)

                if "TOOLS_LOST" in output:
                    pytest.fail(
                        f"BUG REPRODUCED: Tools lost at iteration {iteration} with heavy parallel load!"
                    )

                assert f"Iteration {iteration}" in output

            # Final check
            final_code = """
try:
    result = await predict({"name": "final_test"})
    print(f"FINAL SUCCESS: {result}")
except NameError as e:
    print(f"TOOLS_LOST: {e}")
"""
            final_result = interp.execute(final_code)
            assert "FINAL SUCCESS" in str(final_result)

            # 5 iterations * 10 calls + 1 final = 51
            assert call_count == 51, f"Expected 51 calls, got {call_count}"

        finally:
            interp.shutdown()
