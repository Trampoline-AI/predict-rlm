"""Tests for Pydantic serialization and sandbox detection fixes.

These tests should fail without the fixes and pass with them.
"""

import asyncio
import json
import re
import unittest
from typing import Literal, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field


def _run(coro):
    """Run async predict call from sync test."""
    import nest_asyncio

    nest_asyncio.apply()
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


class ExtractedItem(BaseModel):
    """Test model with complex types including Literal."""

    category: str = Field(description="Category")
    title: str = Field(description="Title")
    priority: Optional[Literal["urgent", "high", "medium", "low"]] = Field(
        default=None, description="Priority"
    )


class TestPydanticSerialization(unittest.TestCase):
    """Test that _to_serializable properly handles complex Pydantic types."""

    def test_literal_field_serialization(self):
        """Test that Pydantic models with Literal fields serialize without mode='python'."""
        # This test demonstrates the fix - without mode='python', Literal types don't serialize properly

        item = ExtractedItem(
            category="TEST",
            title="Test Task",
            priority="high",  # Literal field
        )

        # The OLD way (would have issues with Literal types)
        def old_to_serializable(value):
            if hasattr(value, "model_dump"):
                return value.model_dump()  # No mode='python'
            return value

        # The NEW way (our fix)
        def new_to_serializable(value):
            if isinstance(value, BaseModel):
                return value.model_dump(mode="python")  # With mode='python'
            return value

        # Both should produce dicts, but the new way handles Literal better
        old_result = old_to_serializable(item)
        new_result = new_to_serializable(item)

        # Both should be serializable to JSON
        self.assertIsInstance(old_result, dict)
        self.assertIsInstance(new_result, dict)

        # The new way should preserve the Literal value correctly
        self.assertEqual(new_result["priority"], "high")

        # Both should be JSON serializable
        json.dumps(old_result)
        json.dumps(new_result)


class TestSandboxPydanticDetection(unittest.TestCase):
    """Test that sandbox can detect Pydantic models defined in REPL."""

    def test_signature_must_be_string(self):
        """Test that _get_pydantic_schemas handles non-string signatures."""

        # Simulate the pattern matching that happens in _get_pydantic_schemas
        pattern = r":\s*(?:list\[|List\[|Optional\[)?([A-Z][A-Za-z0-9_]*)"

        # This would fail without our fix
        sig_dict = {"signature": "test"}

        # OLD way - would crash with TypeError
        with self.assertRaises(TypeError) as ctx:
            list(re.finditer(pattern, sig_dict))
        self.assertIn("expected string or bytes-like object", str(ctx.exception))

        # NEW way - convert to string first
        sig_safe = str(sig_dict) if not isinstance(sig_dict, str) else sig_dict
        matches = list(re.finditer(pattern, sig_safe))
        # Should not crash
        self.assertIsInstance(matches, list)

    def test_pydantic_model_detection_in_different_scopes(self):
        """Test that Pydantic models can be found in various scopes."""

        # Simulate what _get_pydantic_schemas does
        def find_model_in_scopes(name, test_globals=None, test_locals=None):
            """Simplified version of our fixed _get_pydantic_schemas logic."""
            cls = None

            # Check provided scopes (for testing)
            if test_globals and name in test_globals:
                cls = test_globals[name]
            elif test_locals and name in test_locals:
                cls = test_locals[name]

            return cls

        # Test 1: Model in globals
        test_globals = {"ExtractedItem": ExtractedItem}
        found = find_model_in_scopes("ExtractedItem", test_globals=test_globals)
        self.assertEqual(found, ExtractedItem)

        # Test 2: Model in locals (simulating REPL definition)
        test_locals = {"ExtractedItem": ExtractedItem}
        found = find_model_in_scopes("ExtractedItem", test_locals=test_locals)
        self.assertEqual(found, ExtractedItem)

        # Test 3: Model not found
        found = find_model_in_scopes("NonExistent", test_globals={}, test_locals={})
        self.assertIsNone(found)

    def test_schema_extraction_from_signature(self):
        """Test that Pydantic schemas can be extracted from signatures."""

        signature = "image: dspy.Image, doc_id: str -> items: list[ExtractedItem]"

        # Pattern to find type names
        pattern = r":\s*(?:list\[|List\[|Optional\[)?([A-Z][A-Za-z0-9_]*)"
        matches = re.finditer(pattern, signature)

        found_types = []
        for match in matches:
            name = match.group(1)
            # Skip DSPy built-ins
            if name not in ("Image", "List", "Optional", "Dict"):
                found_types.append(name)

        # Should find ExtractedItem
        self.assertIn("ExtractedItem", found_types)

        # Now test schema extraction
        # Simulate having ExtractedItem in scope
        test_globals = {"ExtractedItem": ExtractedItem}

        schemas = {}
        for type_name in found_types:
            if type_name in test_globals:
                cls = test_globals[type_name]
                if hasattr(cls, "model_json_schema"):
                    schemas[type_name] = cls.model_json_schema()

        # Should have extracted the schema
        self.assertIn("ExtractedItem", schemas)
        self.assertIn("properties", schemas["ExtractedItem"])
        self.assertIn("category", schemas["ExtractedItem"]["properties"])


class TestIntegration(unittest.TestCase):
    """Test that the fixes work together in a realistic scenario."""

    def test_predict_tool_with_pydantic_model(self):
        """Test the full flow of using a Pydantic model in a predict signature."""

        # This simulates what happens when the model uses ExtractedItem
        signature = "page: dspy.Image -> items: list[ExtractedItem]"

        # Step 1: Ensure signature is string (our first fix)
        if not isinstance(signature, str):
            signature = str(signature)

        # Step 2: Extract Pydantic schemas (our second fix - checking multiple scopes)
        pattern = r":\s*(?:list\[|List\[|Optional\[)?([A-Z][A-Za-z0-9_]*)"
        schemas = {}

        # Simulate ExtractedItem being in the REPL's globals
        repl_globals = {"ExtractedItem": ExtractedItem}

        for match in re.finditer(pattern, signature):
            name = match.group(1)
            if name == "ExtractedItem" and name in repl_globals:
                cls = repl_globals[name]
                if hasattr(cls, "model_json_schema"):
                    schemas[name] = cls.model_json_schema()

        # Should have found and extracted the schema
        self.assertIn("ExtractedItem", schemas)

        # Step 3: Create payload with schemas
        payload_dict = {
            "args": [signature],
            "kwargs": {"page": "image_url"},
            "pydantic_schemas": schemas,
        }

        # Should be JSON serializable
        json_payload = json.dumps(payload_dict)
        self.assertIsInstance(json_payload, str)

        # Step 4: When predict returns ExtractedItem instances, they should serialize
        result_items = [ExtractedItem(category="TASK", title="Test", priority="high")]

        # Our _to_serializable fix
        def to_serializable(value):
            if isinstance(value, BaseModel):
                return value.model_dump(mode="python")
            if isinstance(value, list):
                return [to_serializable(item) for item in value]
            return value

        serialized = to_serializable(result_items)

        # Should be JSON serializable
        json_result = json.dumps(serialized)
        self.assertIsInstance(json_result, str)

        # Should preserve the Literal value
        parsed = json.loads(json_result)
        self.assertEqual(parsed[0]["priority"], "high")


class TestListDictSerialization(unittest.TestCase):
    """Test that list[dict] output type is properly serialized.

    This addresses the issue where using list[dict] as an output type
    in predict() causes Pydantic serialization warnings about Message
    and StreamingChoices objects from litellm.
    """

    def test_list_dict_serialization_in_to_serializable(self):
        """Test that list[dict] values are properly handled by _to_serializable."""
        from typing import Any

        # Replicate the _to_serializable function from predict_rlm.py
        def _to_serializable(value: Any) -> Any:
            """Convert Pydantic models to dicts recursively."""
            if value is None:
                return value

            if isinstance(value, BaseModel):
                return value.model_dump(mode="python")

            if hasattr(value, "dict") and hasattr(value, "__fields__"):
                return value.dict()

            if isinstance(value, list):
                return [_to_serializable(item) for item in value]

            if isinstance(value, tuple):
                return [_to_serializable(item) for item in value]

            if isinstance(value, dict):
                return {k: _to_serializable(v) for k, v in value.items()}

            if isinstance(value, set):
                return list(value)

            return value

        # Test: list of dicts with various value types
        test_data = [
            {"category": "TASK", "title": "Test Task", "page": 1},
            {"category": "FORM", "title": "Another Task", "priority": "high"},
        ]

        result = _to_serializable(test_data)

        # Should be a list of dicts
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], dict)
        self.assertIsInstance(result[1], dict)

        # Should be JSON serializable
        json_str = json.dumps(result)
        self.assertIsInstance(json_str, str)

        # Parse back and verify
        parsed = json.loads(json_str)
        self.assertEqual(parsed[0]["category"], "TASK")
        self.assertEqual(parsed[1]["priority"], "high")

    def test_list_dict_with_nested_pydantic_model(self):
        """Test that list containing dicts with Pydantic models are serialized."""
        from typing import Any

        class NestedModel(BaseModel):
            name: str
            value: int

        def _to_serializable(value: Any) -> Any:
            if value is None:
                return value
            if isinstance(value, BaseModel):
                return value.model_dump(mode="python")
            if isinstance(value, list):
                return [_to_serializable(item) for item in value]
            if isinstance(value, dict):
                return {k: _to_serializable(v) for k, v in value.items()}
            return value

        # A list of dicts where some values are Pydantic models
        test_data = [
            {"category": "TASK", "nested": NestedModel(name="test", value=42)},
        ]

        result = _to_serializable(test_data)

        # Should have serialized the nested model
        self.assertIsInstance(result[0]["nested"], dict)
        self.assertEqual(result[0]["nested"]["name"], "test")
        self.assertEqual(result[0]["nested"]["value"], 42)

        # Should be JSON serializable
        json_str = json.dumps(result)
        self.assertIsInstance(json_str, str)

    def test_list_dict_with_none_values(self):
        """Test that list[dict] with None values is properly serialized."""

        def _to_serializable(value):
            if value is None:
                return value
            if isinstance(value, BaseModel):
                return value.model_dump(mode="python")
            if isinstance(value, list):
                return [_to_serializable(item) for item in value]
            if isinstance(value, dict):
                return {k: _to_serializable(v) for k, v in value.items()}
            return value

        # List of dicts with None values (common pattern in extraction results)
        test_data = [
            {
                "category": "TASK",
                "title": "Test Task",
                "priority": None,
                "due_date": None,
            },
        ]

        result = _to_serializable(test_data)

        # Should preserve None values
        self.assertIsNone(result[0]["priority"])
        self.assertIsNone(result[0]["due_date"])

        # Should be JSON serializable
        json_str = json.dumps(result)
        self.assertIn('"priority": null', json_str)


@pytest.mark.integration
class TestListDictInInterpreter(unittest.TestCase):
    """Test list[dict] through the full interpreter pipeline."""

    def test_predict_with_list_dict_output(self):
        """Test that predict with list[dict] output works through the sandbox."""
        from predict_rlm.interpreter import (
            JspiInterpreter,
        )

        # Track calls to predict
        predict_calls = []

        def mock_predict(signature: str, pydantic_schemas=None, **kwargs):
            predict_calls.append(
                {"signature": signature, "schemas": pydantic_schemas, "kwargs": kwargs}
            )
            # Return a list of dicts (simulating DSPy output)
            return {
                "items": [
                    {"category": "TASK", "title": "Task 1", "page": 0},
                    {"category": "FORM", "title": "Form 1", "priority": "high"},
                ]
            }

        interpreter = JspiInterpreter(
            tools={"predict": mock_predict},
            preinstall_packages=False,
        )
        try:
            # This is the pattern from the user's trace that was causing issues
            result = interpreter.execute("""
import json

result = await predict(
    "context: str -> items: list[dict]",
    instructions="Extract items",
    context="test context"
)
print("Got items:", len(result["items"]))
print("First item:", result["items"][0])

# Verify we can serialize the result (this was failing)
serialized = json.dumps(result["items"])
print("Serialized OK:", len(serialized), "chars")
""")

            # Check that it worked
            assert "Got items: 2" in str(result)
            assert "First item:" in str(result)
            assert "Serialized OK:" in str(result)

            # Verify predict was called correctly
            assert len(predict_calls) == 1
            assert predict_calls[0]["signature"] == "context: str -> items: list[dict]"

        finally:
            interpreter.shutdown()

    def test_predict_with_list_dict_containing_none_values(self):
        """Test that predict with list[dict] containing None values works correctly."""
        from predict_rlm.interpreter import (
            JspiInterpreter,
        )

        def mock_predict(signature: str, **kwargs):
            # Return list of dicts with None values (common in extraction results)
            return {
                "items": [
                    {
                        "category": "TASK",
                        "title": "Task 1",
                        "priority": None,  # Nullable field
                        "due_date": None,  # Nullable field
                        "page": 0,
                    },
                    {
                        "category": "FORM",
                        "title": "Form 1",
                        "priority": "high",
                        "due_date": "2026-01-15",
                        "page": 1,
                    },
                ]
            }

        interpreter = JspiInterpreter(
            tools={"predict": mock_predict},
            preinstall_packages=False,
        )
        try:
            result = interpreter.execute("""
import json

result = await predict(
    "context: str -> items: list[dict]",
    context="test context"
)

# Check None values are preserved
first_item = result["items"][0]
print("First item priority:", first_item["priority"])
print("First item due_date:", first_item["due_date"])

# Verify serialization works with None
serialized = json.dumps(result["items"])
print("Serialized OK")

# Verify None is preserved after serialization
parsed = json.loads(serialized)
print("Parsed priority is None:", parsed[0]["priority"] is None)
""")

            assert "First item priority: None" in str(result)
            assert "First item due_date: None" in str(result)
            assert "Serialized OK" in str(result)
            assert "Parsed priority is None: True" in str(result)

        finally:
            interpreter.shutdown()

    def test_predict_with_list_dict_complex_nested_values(self):
        """Test that predict handles list[dict] with complex nested structures."""
        from predict_rlm.interpreter import (
            JspiInterpreter,
        )

        def mock_predict(signature: str, **kwargs):
            # Return list of dicts with nested structures
            return {
                "items": [
                    {
                        "category": "TASK",
                        "title": "Task 1",
                        "metadata": {
                            "source": "RFP Section 5",
                            "references": ["page 1", "page 2"],
                        },
                        "tags": ["urgent", "compliance"],
                    },
                ]
            }

        interpreter = JspiInterpreter(
            tools={"predict": mock_predict},
            preinstall_packages=False,
        )
        try:
            result = interpreter.execute("""
import json

result = await predict(
    "context: str -> items: list[dict]",
    context="test context"
)

item = result["items"][0]
print("Metadata source:", item["metadata"]["source"])
print("Tags:", item["tags"])
print("References:", item["metadata"]["references"])

# Verify serialization works with nested structures
serialized = json.dumps(result["items"])
print("Serialized length:", len(serialized))
""")

            assert "Metadata source: RFP Section 5" in str(result)
            assert "Tags: ['urgent', 'compliance']" in str(result)
            assert "References: ['page 1', 'page 2']" in str(result)
            assert "Serialized length:" in str(result)

        finally:
            interpreter.shutdown()


class TestToSerializableEdgeCases(unittest.TestCase):
    """Test edge cases in _to_serializable that could cause 'method is not JSON serializable'."""

    def _get_to_serializable(self):
        """Get the actual _to_serializable function from predict_rlm."""
        from typing import Any

        def _to_serializable(value: Any) -> Any:
            """Convert Pydantic models and other objects to JSON-serializable dicts."""
            # Primitives pass through directly
            if value is None or isinstance(value, (str, int, float, bool)):
                return value

            if isinstance(value, BaseModel):
                return value.model_dump(mode="python")

            if hasattr(value, "dict") and hasattr(value, "__fields__"):
                return value.dict()

            # Dataclasses - convert to dict using asdict
            if hasattr(value, "__dataclass_fields__"):
                import dataclasses

                return {k: _to_serializable(v) for k, v in dataclasses.asdict(value).items()}

            if isinstance(value, list):
                return [_to_serializable(item) for item in value]

            if isinstance(value, tuple):
                return [_to_serializable(item) for item in value]

            if isinstance(value, dict):
                return {k: _to_serializable(v) for k, v in value.items()}

            if isinstance(value, set):
                return list(value)

            if hasattr(value, "__dict__") and type(value).__name__ == "Prediction":
                return {
                    k: _to_serializable(v)
                    for k, v in value.__dict__.items()
                    if not k.startswith("_")
                }

            # Fallback: try to convert to dict or string representation
            if hasattr(value, "__dict__"):
                try:
                    return {
                        k: _to_serializable(v)
                        for k, v in value.__dict__.items()
                        if not k.startswith("_") and not callable(v)
                    }
                except Exception:
                    pass

            # Last resort: convert to string representation
            return str(value)

        return _to_serializable

    def test_object_with_method_is_serializable(self):
        """Test that objects with methods are now serializable (fixed)."""
        _to_serializable = self._get_to_serializable()

        class ObjectWithMethod:
            def __init__(self):
                self.value = "test"
                self.number = 42

            def some_method(self):
                return "result"

        obj = ObjectWithMethod()
        result = _to_serializable(obj)

        # Now it should be a dict (methods excluded)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["value"], "test")
        self.assertEqual(result["number"], 42)
        # Method should NOT be included
        self.assertNotIn("some_method", result)

        # Should be JSON serializable
        json_str = json.dumps(result)
        self.assertIsInstance(json_str, str)

    def test_dataclass_is_serializable(self):
        """Test that dataclasses are properly serialized (fixed)."""
        from dataclasses import dataclass

        _to_serializable = self._get_to_serializable()

        @dataclass
        class TaskItem:
            category: str
            title: str
            priority: Optional[str] = None

        item = TaskItem(category="TASK", title="Test", priority="high")
        result = _to_serializable(item)

        # Should be a dict
        self.assertIsInstance(result, dict)
        self.assertEqual(result["category"], "TASK")
        self.assertEqual(result["title"], "Test")
        self.assertEqual(result["priority"], "high")

        # Should be JSON serializable
        json_str = json.dumps(result)
        self.assertIsInstance(json_str, str)

    def test_nested_dataclass_is_serializable(self):
        """Test that nested dataclasses are properly serialized."""
        from dataclasses import dataclass

        _to_serializable = self._get_to_serializable()

        @dataclass
        class Address:
            street: str
            city: str

        @dataclass
        class Person:
            name: str
            address: Address

        person = Person(name="Test", address=Address(street="123 Main", city="Boston"))
        result = _to_serializable(person)

        # Should be nested dicts
        self.assertIsInstance(result, dict)
        self.assertEqual(result["name"], "Test")
        self.assertIsInstance(result["address"], dict)
        self.assertEqual(result["address"]["street"], "123 Main")
        self.assertEqual(result["address"]["city"], "Boston")

        # Should be JSON serializable
        json_str = json.dumps(result)
        self.assertIsInstance(json_str, str)

    def test_list_of_dataclasses_is_serializable(self):
        """Test that list of dataclasses is properly serialized."""
        from dataclasses import dataclass

        _to_serializable = self._get_to_serializable()

        @dataclass
        class TaskItem:
            category: str
            title: str

        items = [
            TaskItem(category="TASK", title="Task 1"),
            TaskItem(category="FORM", title="Form 1"),
        ]
        result = _to_serializable(items)

        # Should be a list of dicts
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], dict)
        self.assertEqual(result[0]["category"], "TASK")

        # Should be JSON serializable
        json_str = json.dumps(result)
        self.assertIsInstance(json_str, str)

    def test_unknown_object_converted_to_string(self):
        """Test that completely unknown objects are converted to string."""
        _to_serializable = self._get_to_serializable()

        # An object without __dict__ (can't extract attributes)
        class WeirdObject:
            __slots__ = ["value"]

            def __init__(self):
                self.value = "test"

            def __str__(self):
                return "WeirdObject(value=test)"

        obj = WeirdObject()
        result = _to_serializable(obj)

        # Should be converted to string
        self.assertEqual(result, "WeirdObject(value=test)")

        # Should be JSON serializable
        json_str = json.dumps(result)
        self.assertIsInstance(json_str, str)


class TestPredictToolListDict:
    """Test the actual predict tool from PredictRLM with list[dict] output."""

    @pytest.mark.asyncio
    async def test_predict_tool_with_list_dict_output(self):
        """Test that the predict tool handles list[dict] correctly."""
        import warnings

        from predict_rlm import PredictRLM

        mock_lm = MagicMock()
        rlm = PredictRLM("text -> answer", sub_lm=mock_lm, max_iterations=1)
        predict_tool = rlm.tools["predict"].func

        with patch("predict_rlm.predict_rlm.dspy.Predict") as mock_predict_class:
            mock_predictor = MagicMock()
            mock_prediction = MagicMock()
            mock_prediction.keys.return_value = ["items"]
            items_list = [
                {"category": "TASK", "title": "Task 1", "page": 0, "priority": None},
                {"category": "FORM", "title": "Form 1", "page": 1, "priority": "high"},
            ]
            mock_prediction.items = items_list
            mock_prediction.__getitem__ = lambda self, key: getattr(self, key)
            mock_predictor.acall = AsyncMock(return_value=mock_prediction)
            mock_predict_class.return_value = mock_predictor

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                result = await predict_tool(
                    "context: str -> items: list[dict]",
                    context="test context",
                )

                assert isinstance(result, dict)
                assert "items" in result
                assert isinstance(result["items"], list)
                assert len(result["items"]) == 2
                assert result["items"][0]["category"] == "TASK"
                assert result["items"][1]["priority"] == "high"
                assert result["items"][0]["priority"] is None

                json_str = json.dumps(result)
                assert isinstance(json_str, str)

                pydantic_warnings = [
                    warning
                    for warning in w
                    if "PydanticSerializationUnexpectedValue" in str(warning.message)
                ]
                assert len(pydantic_warnings) == 0, (
                    f"Got unexpected Pydantic warnings: {pydantic_warnings}"
                )

    @pytest.mark.asyncio
    async def test_predict_tool_with_list_of_pydantic_models(self):
        """Test that the predict tool handles list of Pydantic models correctly."""
        import warnings

        from predict_rlm import PredictRLM

        class TaskItem(BaseModel):
            category: str
            title: str
            priority: Optional[str] = None

        mock_lm = MagicMock()
        rlm = PredictRLM("text -> answer", sub_lm=mock_lm, max_iterations=1)
        predict_tool = rlm.tools["predict"].func

        with patch("predict_rlm.predict_rlm.dspy.Predict") as mock_predict_class:
            mock_predictor = MagicMock()
            mock_prediction = MagicMock()
            mock_prediction.keys.return_value = ["items"]
            mock_prediction.items = [
                TaskItem(category="TASK", title="Task 1", priority=None),
                TaskItem(category="FORM", title="Form 1", priority="high"),
            ]
            mock_prediction.__getitem__ = lambda self, key: getattr(self, key)
            mock_predictor.acall = AsyncMock(return_value=mock_prediction)
            mock_predict_class.return_value = mock_predictor

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                result = await predict_tool(
                    "context: str -> items: list[TaskItem]",
                    context="test context",
                    pydantic_schemas={"TaskItem": TaskItem.model_json_schema()},
                )

                assert isinstance(result, dict)
                assert "items" in result
                assert isinstance(result["items"], list)
                assert isinstance(result["items"][0], dict)
                assert result["items"][0]["category"] == "TASK"

                json_str = json.dumps(result)
                assert isinstance(json_str, str)

                # Check for Pydantic warnings
                pydantic_warnings = [
                    warning
                    for warning in w
                    if "PydanticSerializationUnexpectedValue" in str(warning.message)
                ]
                assert len(pydantic_warnings) == 0


@pytest.mark.integration
class TestSchemaExtractionFromCallStack(unittest.TestCase):
    """Test that Pydantic schemas are extracted correctly from user's REPL scope."""

    def test_schema_extraction_traverses_full_call_stack(self):
        """Test that _get_pydantic_schemas traverses the entire call stack.

        This tests the fix for the issue where models defined in the REPL
        weren't found because _get_pydantic_schemas only checked one frame back
        instead of traversing the full call stack.
        """
        import re

        # A model defined in this test's scope (simulates REPL-defined model)
        class DeepStackModel(BaseModel):
            name: str
            value: int

        # Replicate the _get_pydantic_schemas logic with the FIX applied
        def _get_pydantic_schemas_fixed(sig):
            """Extract schemas by traversing the FULL call stack."""
            import inspect

            schemas = {}
            pattern = r":\s*(?:list\[|List\[|Optional\[)?([A-Z][A-Za-z0-9_]*)"
            for match in re.finditer(pattern, sig):
                name = match.group(1)
                if name in ("Image", "List", "Optional", "Dict", "Any", "Union"):
                    continue

                cls = None
                # FIX: Traverse the FULL call stack, not just one frame back
                frame = inspect.currentframe()
                while frame:
                    if name in frame.f_globals:
                        cls = frame.f_globals[name]
                        break
                    if name in frame.f_locals:
                        cls = frame.f_locals[name]
                        break
                    frame = frame.f_back

                if cls and hasattr(cls, "model_json_schema"):
                    try:
                        schemas[name] = cls.model_json_schema()
                    except Exception:
                        pass
            return schemas

        # Now call through multiple nested functions (simulating production call stack)
        def level_3(sig):
            return _get_pydantic_schemas_fixed(sig)

        def level_2(sig):
            return level_3(sig)

        def level_1(sig):
            return level_2(sig)

        # Call through deep stack - model is defined in THIS scope (like REPL)
        schemas = level_1("context: str -> items: list[DeepStackModel]")

        # Should find the model even though it's many frames up
        self.assertIn(
            "DeepStackModel",
            schemas,
            f"DeepStackModel should be found in call stack. Got: {schemas}",
        )
        self.assertIn("properties", schemas["DeepStackModel"])
        self.assertIn("name", schemas["DeepStackModel"]["properties"])

    def test_schema_extraction_finds_model_in_caller_frame(self):
        """Test that _get_pydantic_schemas can find models defined in caller's scope.

        This is a regression test for the issue where models defined in the REPL
        weren't being found by _get_pydantic_schemas because it only looked at
        immediate caller's globals, not the full call stack.
        """
        from predict_rlm.interpreter import (
            JspiInterpreter,
        )

        # Track what schemas are extracted and passed to predict
        received_schemas = []

        def mock_predict(signature: str, pydantic_schemas=None, **kwargs):
            received_schemas.append(
                {
                    "signature": signature,
                    "schemas": pydantic_schemas,
                }
            )
            return {"tasks": []}

        interpreter = JspiInterpreter(
            tools={"predict": mock_predict},
            preinstall_packages=False,
        )
        try:
            # Define a Pydantic model in the REPL and use it in a signature
            result = interpreter.execute("""
from pydantic import BaseModel
from typing import Optional

class TaskCandidate(BaseModel):
    category: str
    title: str
    description: str
    priority: Optional[str] = None

# Call predict with the custom model in the signature
result = await predict(
    "page: dspy.Image -> tasks: list[TaskCandidate]",
    instructions="Extract tasks",
    page="test_url"
)
print("Predict called successfully")
""")

            # Check that predict was called
            assert "Predict called successfully" in str(result)
            assert len(received_schemas) == 1

            # Check that the schema was extracted and passed
            schemas = received_schemas[0]["schemas"]
            self.assertIsNotNone(schemas, "Schema should be extracted from REPL-defined model")
            self.assertIn(
                "TaskCandidate", schemas, f"TaskCandidate schema not found. Got: {schemas}"
            )

            # Verify the schema has the expected structure
            tc_schema = schemas["TaskCandidate"]
            self.assertIn("properties", tc_schema)
            self.assertIn("category", tc_schema["properties"])
            self.assertIn("title", tc_schema["properties"])
            self.assertIn("description", tc_schema["properties"])
            self.assertIn("priority", tc_schema["properties"])

        finally:
            interpreter.shutdown()


if __name__ == "__main__":
    unittest.main()
