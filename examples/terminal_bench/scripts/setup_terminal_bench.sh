#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${EXAMPLE_DIR}/../.." && pwd)"

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TERMINAL_BENCH_VERSION="${TERMINAL_BENCH_VERSION:-0.2.18}"
TB_VENV="${TB_VENV:-${EXAMPLE_DIR}/.terminal-bench-venv}"
VENV_PYTHON="${TB_VENV}/bin/python"

uv venv --python "${PYTHON_VERSION}" "${TB_VENV}"
uv pip install --python "${VENV_PYTHON}" \
    -e "${REPO_ROOT}" \
    -e "${EXAMPLE_DIR}" \
    "terminal-bench==${TERMINAL_BENCH_VERSION}"

"${VENV_PYTHON}" - <<'PY'
import terminal_bench
import predict_rlm
import terminal_bench_rlm.tools.tbench_agent as tbench_agent

agent_class = tbench_agent.TerminalBenchRLMAgent
if agent_class.name() != "predict-rlm":
    raise RuntimeError(f"unexpected agent name: {agent_class.name()!r}")
if not issubclass(agent_class, tbench_agent.TerminalBenchRLMBaseAgent):
    raise RuntimeError("TerminalBenchRLMAgent must inherit TerminalBenchRLMBaseAgent")

print("Verified imports: terminal_bench, predict_rlm, terminal_bench_rlm.tools.tbench_agent")
print("Terminal-Bench import path: terminal_bench_rlm.tools.tbench_agent:TerminalBenchRLMAgent")
PY
