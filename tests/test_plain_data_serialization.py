"""Tests for shared host-side plain-data serialization."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from predict_rlm.interpreter import JspiInterpreter
from predict_rlm.interpreters import SbxConfig, SbxInterpreter
from predict_rlm.serialization import to_plain_data


class TaskItem(BaseModel):
    title: str
    done: bool = False


def test_pydantic_v2_model_becomes_dict():
    assert to_plain_data(TaskItem(title="write tests", done=True)) == {
        "title": "write tests",
        "done": True,
    }


def test_list_of_pydantic_models_becomes_list_of_dicts():
    assert to_plain_data([
        TaskItem(title="one"),
        TaskItem(title="two", done=True),
    ]) == [
        {"title": "one", "done": False},
        {"title": "two", "done": True},
    ]


def test_pydantic_v1_style_model_becomes_dict():
    class V1StyleModel:
        __fields__ = {"name": object()}

        def dict(self):
            return {"name": "legacy"}

    assert to_plain_data(V1StyleModel()) == {"name": "legacy"}


def test_nested_containers_are_recursively_normalized():
    assert to_plain_data({
        "tasks": (TaskItem(title="nested"),),
        "metadata": [{"owners": ("emile", "agent")}],
    }) == {
        "tasks": [{"title": "nested", "done": False}],
        "metadata": [{"owners": ["emile", "agent"]}],
    }


def test_dataclass_becomes_dict():
    @dataclass
    class Job:
        name: str
        task: TaskItem

    assert to_plain_data(Job(name="demo", task=TaskItem(title="ship"))) == {
        "name": "demo",
        "task": {"title": "ship", "done": False},
    }


def test_set_becomes_list():
    assert to_plain_data({3, 1, 2}) == [1, 2, 3]


def test_unknown_object_is_preserved():
    class CustomObject:
        pass

    value = CustomObject()

    assert to_plain_data(value) is value


def test_jspi_variable_serializer_uses_plain_data():
    @dataclass
    class Job:
        task: TaskItem

    interpreter = object.__new__(JspiInterpreter)

    assert (
        interpreter._serialize_value(Job(task=TaskItem(title="jspi")))
        == "{'task': {'title': 'jspi', 'done': False}}"
    )


def test_sbx_input_serializer_uses_plain_data(tmp_path):
    @dataclass
    class Job:
        task: TaskItem

    interpreter = SbxInterpreter(
        config=SbxConfig(name="serialization-test"),
        preinstall_packages=False,
        _runner_command=["unused"],
        _staging_root=tmp_path / "staging",
    )

    try:
        assert interpreter._map_variable_value(Job(task=TaskItem(title="sbx"))) == {
            "task": {"title": "sbx", "done": False}
        }
    finally:
        interpreter.shutdown()
