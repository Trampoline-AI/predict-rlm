.PHONY: test test-unit test-integration \
        test-core test-sbx test-gepa test-codex-lm \
        test-integration-jspi test-integration-sbx

# Extra args after the target are forwarded to pytest, e.g.
#   make test-core tests/test_predict_rlm.py -k schema
ARGS = $(filter-out $@,$(MAKECMDGOALS))

# Run pytest against whatever is currently installed.
test:
	uv run pytest $(ARGS)

# All unit (non-integration) suites. Installs every extra so sbx/gepa/codex-lm
# tests actually run rather than skipping.
test-unit:
	uv run --all-extras pytest -m "not integration" $(ARGS)

# All integration suites (real backends). Real-SBX tests still need the sbx CLI
# + `sbx login` + PREDICT_RLM_RUN_SBX_TESTS=1 (see test-integration-sbx).
test-integration:
	uv run --all-extras pytest -m "integration" $(ARGS)

# --- Per-suite targets (one per CI job; each installs only what it needs) ---

# Core: no extras. Pure host-side logic; proves the package works standalone
# (passes with neither Deno nor websockets present).
test-core:
	uv run pytest -m "not integration and not sbx and not gepa and not codex_lm" $(ARGS)

# SBX/supervisor backend unit tests against the local runner/supervisor seam.
# [sbx] extra = websockets; does NOT need the real Docker Sandboxes service.
test-sbx:
	uv run --extra sbx pytest -m "sbx and not integration" $(ARGS)

# rlm_gepa subsystem ([gepa] extra).
test-gepa:
	uv run --extra gepa pytest -m "gepa and not integration" $(ARGS)

# dspy_codex_lm subsystem ([codex-lm] extra).
test-codex-lm:
	uv run --extra codex-lm pytest -m "codex_lm and not integration" $(ARGS)

# Real Deno/WASM (JSPI) integration tests. Deno is a default dependency, so no
# extra is required.
test-integration-jspi:
	uv run pytest -m "integration and not sbx" $(ARGS)

# Real Docker Sandboxes integration tests. Requires the sbx CLI and `sbx login`.
test-integration-sbx:
	PREDICT_RLM_RUN_SBX_TESTS=1 uv run --extra sbx pytest -m "integration and sbx" $(ARGS)

# Swallow forwarded pytest args (extra make goals) so they don't error as targets.
%:
	@:
