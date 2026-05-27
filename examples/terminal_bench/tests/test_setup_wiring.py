from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_DIR = Path(__file__).resolve().parent.parent


def _project_dependencies(pyproject_path: Path) -> list[str]:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    return list(data.get("project", {}).get("dependencies", []))


def _all_project_dependencies(pyproject_path: Path) -> list[str]:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    dependencies = list(data.get("project", {}).get("dependencies", []))
    for optional_dependencies in data.get("project", {}).get("optional-dependencies", {}).values():
        dependencies.extend(optional_dependencies)
    return dependencies


def test_example_pyproject_installs_terminal_bench_rlm_without_terminal_bench_dependency() -> None:
    pyproject = EXAMPLE_DIR / "pyproject.toml"

    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    assert data["project"]["name"] == "terminal-bench-rlm"
    assert data["project"]["requires-python"] == ">=3.12"
    assert data["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "terminal_bench_rlm"
    ]
    dependencies = "\n".join(_project_dependencies(pyproject))
    assert "predict-rlm[codex-lm,gepa,gepa-viz]" in dependencies
    assert "terminal-bench" not in dependencies
    assert "terminal-bench" not in "\n".join(
        _all_project_dependencies(REPO_ROOT / "pyproject.toml")
    )


def test_example_pyproject_exposes_rlm_gepa_script() -> None:
    pyproject = EXAMPLE_DIR / "pyproject.toml"

    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    assert data["project"]["scripts"]["rlm-gepa"] == "terminal_bench_rlm.gepa:main"


def test_setup_script_contains_venv_install_and_import_verification_contract() -> None:
    script = EXAMPLE_DIR / "scripts" / "setup_terminal_bench.sh"

    text = script.read_text(encoding="utf-8")

    assert "set -euo pipefail" in text
    assert 'PYTHON_VERSION="${PYTHON_VERSION:-3.12}"' in text
    assert 'TERMINAL_BENCH_VERSION="${TERMINAL_BENCH_VERSION:-0.2.18}"' in text
    assert 'TB_VENV="${TB_VENV:-${EXAMPLE_DIR}/.terminal-bench-venv}"' in text
    assert 'uv venv --python "${PYTHON_VERSION}" "${TB_VENV}"' in text
    assert '"terminal-bench==${TERMINAL_BENCH_VERSION}"' in text
    assert '-e "${REPO_ROOT}[codex-lm]"' in text
    assert '-e "${EXAMPLE_DIR}"' in text
    assert "import terminal_bench" in text
    assert "import predict_rlm" in text
    assert "import dspy_codex_lm" in text
    assert "import terminal_bench_rlm.tools.tbench_agent as tbench_agent" in text
    assert "TerminalBenchRLMAgent" in text


def test_makefile_exposes_setup_smoke_and_test_targets() -> None:
    text = (EXAMPLE_DIR / "Makefile").read_text(encoding="utf-8")

    assert ".PHONY: setup smoke test" in text
    assert "scripts/setup_terminal_bench.sh" in text
    assert "TB_VENV ?= .terminal-bench-venv" in text
    assert "$(TB_VENV)/bin/python" in text
    assert "scripts/smoke_three_tasks.py --json" in text
    assert "uv run pytest tests -q" in text
