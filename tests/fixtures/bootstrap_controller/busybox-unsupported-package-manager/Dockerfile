FROM busybox:1.36.1

ENV ROOT=/tmp/predict_rlm_controller

COPY src/predict_rlm/remote/bootstrap_controller.sh /tmp/bootstrap_controller.sh

RUN mkdir -p "$ROOT/repo"

RUN if sh /tmp/bootstrap_controller.sh --root "$ROOT" --repo "$ROOT/repo" --python 3.12 > /tmp/bootstrap.log 2>&1; then \
        cat /tmp/bootstrap.log; exit 1; \
    fi \
    && grep -q 'unsupported package manager and unable to install Python prerequisites' /tmp/bootstrap.log \
    && grep -q 'package_manager=unsupported' /tmp/bootstrap.log
