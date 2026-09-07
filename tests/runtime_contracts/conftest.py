from __future__ import annotations

from pathlib import Path

import pytest

from .backends import RuntimeHandle, runtime_specs


def _runtime_params():
    for spec in runtime_specs():
        marks = []
        if spec.name == "jspi":
            marks.append(pytest.mark.integration)
        elif spec.name.startswith("sbx"):
            marks.append(pytest.mark.sbx)
            if spec.name == "sbx":
                marks.append(pytest.mark.integration)
        yield pytest.param(spec, id=spec.name, marks=marks)


@pytest.fixture(params=list(_runtime_params()))
def runtime(request: pytest.FixtureRequest, tmp_path: Path) -> RuntimeHandle:
    handle = request.param.make(tmp_path, request.param)
    try:
        yield handle
    finally:
        handle.shutdown()
