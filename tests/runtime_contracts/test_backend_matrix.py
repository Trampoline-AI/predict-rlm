from __future__ import annotations

from .backends import CAPABILITIES, runtime_specs


def test_runtime_specs_have_unique_maintained_target_names() -> None:
    specs = runtime_specs()

    assert [spec.name for spec in specs] == [
        "jspi",
        "python-runner/direct-process",
        "sbx",
        "internal/python-runner-jsonrpc",
    ]
    assert len({spec.name for spec in specs}) == len(specs)


def test_runtime_specs_advertise_only_known_capabilities() -> None:
    unknown = {
        capability
        for spec in runtime_specs()
        for capability in spec.capabilities
        if capability not in CAPABILITIES
    }

    assert unknown == set()


def test_legacy_environment_api_paths_are_not_primitive_targets() -> None:
    target_names = {spec.name for spec in runtime_specs()}

    assert "python-runner/environment-api" not in target_names
    assert "harbor/environment-api" not in target_names
    assert "docker-container-adapter" not in target_names
