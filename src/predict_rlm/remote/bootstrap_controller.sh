#!/bin/sh
set -eu

ROOT="/tmp/predict_rlm_controller"
REPO=""
EXTRA=""
REQUESTED_PYTHON="3.12"

usage() {
    cat >&2 <<'EOF'
usage: bootstrap_controller.sh --root PATH --repo PATH [--extra EXTRA] [--python VERSION]
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --root)
            [ "$#" -ge 2 ] || { usage; exit 2; }
            ROOT=$2
            shift 2
            ;;
        --repo)
            [ "$#" -ge 2 ] || { usage; exit 2; }
            REPO=$2
            shift 2
            ;;
        --extra)
            [ "$#" -ge 2 ] || { usage; exit 2; }
            EXTRA=$2
            shift 2
            ;;
        --python)
            [ "$#" -ge 2 ] || { usage; exit 2; }
            REQUESTED_PYTHON=$2
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "bootstrap_controller.sh: unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [ -z "$REPO" ]; then
    echo "bootstrap_controller.sh: --repo is required" >&2
    usage
    exit 2
fi

UV_BOOTSTRAP="$ROOT/uv-bootstrap"
CONTROLLER_VENV="$ROOT/.venv"
PROBE_DIR="$ROOT/venv-probe.$$"
PROBE_LOG="$ROOT/venv-probe.$$.log"
INSTALL_LOG="$ROOT/install-python.$$.log"
UV_COMMAND="$UV_BOOTSTRAP/bin/python -m uv"

mkdir -p "$ROOT"

cleanup_probe() {
    rm -rf "$PROBE_DIR"
}

python_version() {
    if command -v python3 >/dev/null 2>&1; then
        python3 -c 'import sys; print("%s.%s" % sys.version_info[:2])' 2>/dev/null || true
    fi
}

pip_status() {
    if command -v python3 >/dev/null 2>&1; then
        python3 -m pip --version 2>&1 || true
    else
        echo "python3 is not on PATH"
    fi
}

detect_package_manager() {
    if command -v apt-get >/dev/null 2>&1; then
        echo "apt"
    elif command -v apk >/dev/null 2>&1; then
        echo "apk"
    else
        echo "unsupported"
    fi
}

probe_venv() {
    cleanup_probe
    : > "$PROBE_LOG"
    if ! command -v python3 >/dev/null 2>&1; then
        echo "python3 is not on PATH" > "$PROBE_LOG"
        return 1
    fi
    if ! python3 -m pip --version >> "$PROBE_LOG" 2>&1; then
        echo "python3 -m pip is not available" >> "$PROBE_LOG"
        return 1
    fi
    if python3 -m venv "$PROBE_DIR" > "$PROBE_LOG" 2>&1; then
        cleanup_probe
        return 0
    fi
    cleanup_probe
    return 1
}

apt_install_python() {
    export DEBIAN_FRONTEND=noninteractive
    apt-get update

    packages="python3 python3-pip python3-venv"
    version=$(python_version)
    if [ -n "$version" ] && apt-cache show "python$version-venv" >/dev/null 2>&1; then
        packages="$packages python$version-venv"
    fi

    apt-get install -y $packages

    version=$(python_version)
    if [ -n "$version" ] && apt-cache show "python$version-venv" >/dev/null 2>&1; then
        apt-get install -y "python$version-venv"
    fi
}

apk_install_python() {
    apk add --no-cache python3 py3-pip py3-virtualenv
}

repair_python() {
    pm=$(detect_package_manager)
    echo "bootstrap_controller.sh: repairing Python/pip/venv with package manager: $pm" >&2
    case "$pm" in
        apt)
            apt_install_python > "$INSTALL_LOG" 2>&1
            ;;
        apk)
            apk_install_python > "$INSTALL_LOG" 2>&1
            ;;
        *)
            echo "bootstrap_controller.sh: python3/pip/venv unavailable and no supported package manager found" >&2
            return 1
            ;;
    esac
}

diagnostics() {
    reason=$1
    pm=$(detect_package_manager)
    echo "bootstrap_controller.sh: $reason" >&2
    echo "bootstrap_controller.sh: diagnostics follow" >&2
    echo "package_manager=$pm" >&2
    echo "python_path=$(command -v python3 2>/dev/null || true)" >&2
    echo "python_version=$(python_version)" >&2
    echo "pip_status=$(pip_status)" >&2
    if [ -f /etc/os-release ]; then
        echo "--- /etc/os-release ---" >&2
        cat /etc/os-release >&2
    else
        echo "--- /etc/os-release missing ---" >&2
    fi
    if [ -s "$INSTALL_LOG" ]; then
        echo "--- package repair output ---" >&2
        cat "$INSTALL_LOG" >&2
    fi
    if [ -s "$PROBE_LOG" ]; then
        echo "--- failing venv probe: python3 -m venv $PROBE_DIR ---" >&2
        cat "$PROBE_LOG" >&2
    fi
}

if ! probe_venv; then
    if ! repair_python; then
        diagnostics "could not repair Python/pip/venv"
        exit 1
    fi
    if ! probe_venv; then
        diagnostics "venv probe still fails after package repair"
        exit 1
    fi
fi

rm -rf "$UV_BOOTSTRAP" "$CONTROLLER_VENV"
python3 -m venv "$UV_BOOTSTRAP"
"$UV_BOOTSTRAP/bin/python" -m pip install --disable-pip-version-check uv
$UV_COMMAND venv --seed --python "$REQUESTED_PYTHON" "$CONTROLLER_VENV"
"$CONTROLLER_VENV/bin/python" -m pip install --disable-pip-version-check -e "$REPO$EXTRA"
