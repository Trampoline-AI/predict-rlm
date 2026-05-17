#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_ID="${RUN_ID:-real-smoke-oracle-tmux}"
N_TASKS="${N_TASKS:-3}"
N_CONCURRENT="${N_CONCURRENT:-1}"
OPS_DIR="runs/_ops/${RUN_ID}"
LOG_PATH="${OPS_DIR}/harness.log"
mkdir -p "${OPS_DIR}"

{
  echo "started_at=$(date '+%Y-%m-%dT%H:%M:%S%z')"
  echo "run_id=${RUN_ID}"
  echo "n_tasks=${N_TASKS}"
  echo "n_concurrent=${N_CONCURRENT}"
  echo "workdir=$(pwd)"
  echo "docker=$(docker info --format 'Docker {{.ServerVersion}} ready; containers={{.Containers}} images={{.Images}}')"
  echo "command=.terminal-bench-venv/bin/tb run --agent oracle --dataset terminal-bench-core==0.1.1 --n-tasks ${N_TASKS} --n-concurrent ${N_CONCURRENT} --run-id ${RUN_ID} --no-upload-results"
} | tee "${LOG_PATH}"

.terminal-bench-venv/bin/tb run \
  --agent oracle \
  --dataset terminal-bench-core==0.1.1 \
  --n-tasks "${N_TASKS}" \
  --n-concurrent "${N_CONCURRENT}" \
  --run-id "${RUN_ID}" \
  --no-upload-results 2>&1 | tee -a "${LOG_PATH}"
status=${PIPESTATUS[0]}

echo "ended_at=$(date '+%Y-%m-%dT%H:%M:%S%z') status=${status}" | tee -a "${LOG_PATH}"
exit "${status}"
