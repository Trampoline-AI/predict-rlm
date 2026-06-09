from __future__ import annotations

from pathlib import Path

import pytest

from .backends import RuntimeHandle, RuntimeSpec, runtime_specs


@pytest.fixture(params=runtime_specs(), ids=lambda spec: spec.name)
def runtime(request: pytest.FixtureRequest, tmp_path: Path) -> RuntimeHandle:
    spec: RuntimeSpec = request.param
    handle = spec.make(tmp_path, spec)
    try:
        yield handle
    finally:
        handle.shutdown()
