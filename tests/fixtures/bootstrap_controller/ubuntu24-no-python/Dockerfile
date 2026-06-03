FROM ubuntu:24.04

ENV ROOT=/tmp/predict_rlm_controller

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY src/predict_rlm/remote/bootstrap_controller.sh /tmp/bootstrap_controller.sh

RUN mkdir -p "$ROOT/repo" \
    && printf '%s\n' 'def ok(): return "ok"' > "$ROOT/repo/fake_controller_repo.py" \
    && printf '%s\n' \
        'from setuptools import setup' \
        'setup(name="fake-controller-repo", version="0.0.0", py_modules=["fake_controller_repo"], extras_require={"codex-lm": []})' \
        > "$ROOT/repo/setup.py"

RUN sh /tmp/bootstrap_controller.sh --root "$ROOT" --repo "$ROOT/repo" --extra '[codex-lm]' --python 3.12 > /tmp/bootstrap.log 2>&1 \
    || { cat /tmp/bootstrap.log; exit 1; } \
    && grep -q 'repairing Python/pip/venv with package manager: apt' /tmp/bootstrap.log \
    && "$ROOT/.venv/bin/python" -c 'import fake_controller_repo; print(fake_controller_repo.ok())'
