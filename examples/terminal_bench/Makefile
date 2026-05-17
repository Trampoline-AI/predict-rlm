.PHONY: setup smoke test

TB_VENV ?= .terminal-bench-venv

setup:
	scripts/setup_terminal_bench.sh

smoke:
	$(TB_VENV)/bin/python scripts/smoke_three_tasks.py --json

test:
	uv run pytest tests -q
