#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
RUN_ID="${RUN_ID:-real-smoke-3-oracle-tmux}"
RESULTS="runs/${RUN_ID}/results.json"
RUN_DIR="runs/${RUN_ID}"
LOG="runs/_ops/${RUN_ID}/harness.log"
TB_PID_FILE="runs/_ops/${RUN_ID}/tb.pid"

while true; do
  tmp="$(mktemp)"
  {
    echo "Terminal-Bench real smoke: ${RUN_ID}"
    echo "refreshed_at=$(date '+%Y-%m-%dT%H:%M:%S%z')"
    echo
    if [ -f "${RESULTS}" ]; then
      .terminal-bench-venv/bin/python - <<PY
import json
from pathlib import Path
p = Path('${RESULTS}')
data = json.loads(p.read_text())
print(f"results={p}")
print(f"accuracy={data.get('accuracy')} resolved={data.get('n_resolved')} unresolved={data.get('n_unresolved')}")
print("tasks:")
for row in data.get('results', []):
    print(f"- {row.get('task_id')}: resolved={row.get('is_resolved')} failure={row.get('failure_mode')}")
PY
    else
      echo "results=waiting (${RESULTS})"
      if [ -d "${RUN_DIR}" ]; then
        echo "run_files:"
        find "${RUN_DIR}" -maxdepth 2 -type f | sed 's#^#- #' | head -20
      fi
      echo
      echo "active build/processes:"
      ps -eo pid,ppid,stat,etime,%cpu,%mem,command \
        | grep -E "tb run|docker compose|docker-buildx|${RUN_ID}" \
        | grep -v grep \
        | sed -E 's#^#- #' \
        | head -12 || true
      echo
      echo "docker containers:"
      docker ps --format '- {{.Names}} {{.Status}} {{.Image}}' | head -10 || true
      echo
      if [ -f "${LOG}" ]; then
        echo "harness_log_mtime=$(stat -f '%Sm' -t '%Y-%m-%dT%H:%M:%S%z' "${LOG}" 2>/dev/null || true)"
        echo "recent harness log:"
        tail -n 8 "${LOG}" | sed -E 's/\x1b\[[0-9;]*[A-Za-z]//g' | tr '\r' '\n' | tail -n 8
      else
        echo "harness_log=waiting (${LOG})"
      fi
    fi
  } > "${tmp}"
  clear
  cat "${tmp}"
  rm -f "${tmp}"
  sleep 5
done
