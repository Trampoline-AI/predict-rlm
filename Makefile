.PHONY: test test-unit test-integration

test:
	uv run pytest $(filter-out $@,$(MAKECMDGOALS))

test-unit:
	uv run pytest -m "not integration" $(filter-out $@,$(MAKECMDGOALS))

test-integration:
	uv run pytest -m "integration" $(filter-out $@,$(MAKECMDGOALS))
