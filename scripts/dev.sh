#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

# Keep runtime files writable in cloud environments as well as local checkouts.
export XDG_CACHE_HOME="$project_root/.cache"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$XDG_CACHE_HOME/python}"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export IPYTHONDIR="$XDG_CACHE_HOME/ipython"
export JUPYTER_CONFIG_DIR="$XDG_CACHE_HOME/jupyter/config"
export JUPYTER_DATA_DIR="$XDG_CACHE_HOME/jupyter/data"
export JUPYTER_RUNTIME_DIR="$XDG_CACHE_HOME/jupyter/runtime"
export PATH="$project_root/.venv/bin:$PATH"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$IPYTHONDIR" \
    "$JUPYTER_CONFIG_DIR" "$JUPYTER_DATA_DIR" "$JUPYTER_RUNTIME_DIR"
chmod 700 "$JUPYTER_RUNTIME_DIR"

action="${1:-check}"
case "$action" in
    setup|check|data|lab) ;;
    *)
        printf 'Usage: bash scripts/dev.sh {setup|check|data|lab}\n' >&2
        exit 2
        ;;
esac

if [[ "$action" == setup && ! -x .venv/bin/python ]]; then
    if command -v uv >/dev/null 2>&1; then
        uv venv --seed --python "$(cat .python-version)" .venv
    elif command -v python3.14 >/dev/null 2>&1; then
        python3.14 -m venv .venv
    else
        printf 'Install uv or Python 3.14, then run setup again.\n' >&2
        exit 1
    fi
fi

if [[ ! -x .venv/bin/python ]]; then
    printf 'Run bash scripts/dev.sh setup first.\n' >&2
    exit 1
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 14), "Python 3.14 is required; found " + sys.version'

if [[ "$action" == setup ]]; then
    .venv/bin/python -m pip install --disable-pip-version-check --no-input -r requirements.txt
    .venv/bin/python -m ipykernel install --sys-prefix --name python3 \
        --display-name 'ElectricityLoad (Python 3.14)'
fi

case "$action" in
    setup|check)
        .venv/bin/python -m pip check
        MPLBACKEND=Agg .venv/bin/python -m unittest discover -s tests -v
        ;;
    data)
        .venv/bin/python scripts/download_data.py
        ;;
    lab)
        exec .venv/bin/python -m jupyterlab --no-browser \
            --ServerApp.ip=127.0.0.1 --ServerApp.port=8888 \
            --ServerApp.port_retries=0 --ServerApp.root_dir="$project_root"
        ;;
esac
