FROM ubuntu:24.04

ENV ROOT=/tmp/predict_rlm_controller

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates python3 python3-pip \
    && useradd --create-home --shell /bin/sh app \
    && rm -rf /var/lib/apt/lists/*

COPY src/predict_rlm/remote/bootstrap_controller.sh /tmp/bootstrap_controller.sh

USER app

RUN mkdir -p "$ROOT/repo"

RUN if sh /tmp/bootstrap_controller.sh --root "$ROOT" --repo "$ROOT/repo" --python 3.12 > /tmp/bootstrap.log 2>&1; then \
        cat /tmp/bootstrap.log; exit 1; \
    fi \
    && grep -q 'root privileges are required to install Python prerequisites with apt-get' /tmp/bootstrap.log \
    && grep -q 'is_root=false' /tmp/bootstrap.log
