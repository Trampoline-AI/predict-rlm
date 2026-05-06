#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
APPWORLD_VERSION="${APPWORLD_VERSION:-0.1.3.post1}"
APPWORLD_VENV="${APPWORLD_VENV:-$ROOT_DIR/.appworld-venv}"
APPWORLD_ROOT="${APPWORLD_ROOT:-$ROOT_DIR}"

uv sync
uv venv "$APPWORLD_VENV" --python "$PYTHON_VERSION"
"$APPWORLD_VENV/bin/pip" install "appworld==$APPWORLD_VERSION"
"$APPWORLD_VENV/bin/appworld" install
"$APPWORLD_VENV/bin/appworld" download data --root "$APPWORLD_ROOT"

cat <<EOF

AppWorld setup is ready.
EOF
