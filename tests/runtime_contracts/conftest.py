from __future__ import annotations

from pathlib import Path

import pytest

from .backends import RuntimeHandle, RuntimeSpec, runtime_specs

_HERE = Path(__file__).resolve().parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark every runtime-contract test as `sbx` so it runs only in the sbx CI job."""
    for item in items:
        try:
            if _HERE in Path(str(item.fspath)).resolve().parents:
                item.add_marker(pytest.mark.sbx)
        except (OSError, ValueError):
            continue


@pytest.fixture(params=runtime_specs(), ids=lambda spec: spec.name)
def runtime(request: pytest.FixtureRequest, tmp_path: Path) -> RuntimeHandle:
    spec: RuntimeSpec = request.param
    handle = spec.make(tmp_path, spec)
    try:
        yield handle
    finally:
        handle.shutdown()
