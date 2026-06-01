.PHONY: setup download-runs unpack-runs bundle-runs

RUNS_BUNDLE_REPO ?= Trampoline-AI/predict-rlm-appworld-runs
RUNS_BUNDLE ?= runs/runs.bundle

setup:
	bash scripts/setup_appworld_data.sh

download-runs:
	hf download $(RUNS_BUNDLE_REPO) runs.bundle \
		--repo-type dataset \
		--local-dir runs

unpack-runs: $(RUNS_BUNDLE)
	@test -n "$(RUNS_BUNDLE_PASSWORD)" || \
		(echo "Set RUNS_BUNDLE_PASSWORD before unpacking runs.bundle" >&2; exit 1)
	mkdir -p runs
	openssl enc -d -aes-256-cbc -pbkdf2 \
		-pass env:RUNS_BUNDLE_PASSWORD \
		-in $(RUNS_BUNDLE) \
		| tar -xzf - -C runs

bundle-runs:
	@test -n "$(RUNS_BUNDLE_PASSWORD)" || \
		(echo "Set RUNS_BUNDLE_PASSWORD before creating runs.bundle" >&2; exit 1)
	tar \
		--exclude='./runs.bundle' \
		--exclude='./README.md' \
		-C runs \
		-czf - . \
		| openssl enc -aes-256-cbc -salt -pbkdf2 \
			-pass env:RUNS_BUNDLE_PASSWORD \
			-out $(RUNS_BUNDLE)
