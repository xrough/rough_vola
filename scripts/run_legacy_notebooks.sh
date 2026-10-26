#!/usr/bin/env bash
# Execute archived recipe notebooks, attempting the full set and reporting failures.
#
#   bash scripts/run_notebooks.sh --legacy                    # archived recipes
#   RVLAB_FORCE_SYNTHETIC=1 bash scripts/run_notebooks.sh --legacy
#   bash scripts/run_notebooks.sh --legacy notebooks/legacy/forecasting_the_smile.ipynb
#   bash scripts/run_notebooks.sh --legacy --output-dir /tmp/executed-notebooks
#
# Notebooks are committed WITH their outputs so they read on GitHub, hence the
# in-place default. --output-dir preserves inputs for read-only/Kaggle mounts.

set -euo pipefail

PY="${RVLAB_PYTHON:-python3}"
REQUESTED_KERNEL="${RVLAB_KERNEL:-}"
KERNEL=""
KERNEL_PREFIX=""
LOG=""
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

cleanup() {
    [ -z "$LOG" ] || rm -f -- "$LOG"
    # KERNEL_PREFIX is assigned only from mktemp below, never from user input.
    [ -z "$KERNEL_PREFIX" ] || rm -rf -- "$KERNEL_PREFIX"
}
trap cleanup EXIT

EXECUTED_NOTEBOOK_DIR="${RVLAB_EXECUTED_NOTEBOOK_DIR:-}"
if [ "${1:-}" = "--output-dir" ]; then
    if [ $# -lt 2 ] || [ -z "$2" ]; then
        echo "--output-dir requires a writable directory" >&2
        exit 2
    fi
    EXECUTED_NOTEBOOK_DIR="$2"
    shift 2
fi

if ! command -v "$PY" >/dev/null 2>&1 && [ ! -x "$PY" ]; then
    echo "Python interpreter not found: $PY" >&2
    echo "Set RVLAB_PYTHON to the interpreter with rvlab's notebook dependencies." >&2
    exit 2
fi

# Establish writable, run-scoped state before importing the notebook stack.
# This keeps preflight, font caches, kernelspecs and connection files away from
# read-only home/input mounts and removes everything together on exit.
KERNEL_PREFIX="$(mktemp -d "${TMPDIR:-/tmp}/rvlab_kernel.XXXXXX")"
KERNEL_PREFIX="$(cd "$KERNEL_PREFIX" && pwd -P)"
export IPYTHONDIR="$KERNEL_PREFIX/ipython"
export JUPYTER_CONFIG_DIR="$KERNEL_PREFIX/jupyter-config"
export JUPYTER_RUNTIME_DIR="$KERNEL_PREFIX/runtime"
export MPLCONFIGDIR="$KERNEL_PREFIX/matplotlib"
export XDG_CACHE_HOME="$KERNEL_PREFIX/cache"
export JOBLIB_TEMP_FOLDER="$KERNEL_PREFIX/joblib"
for state_dir in "$IPYTHONDIR" "$JUPYTER_CONFIG_DIR" "$JUPYTER_RUNTIME_DIR" \
                 "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$JOBLIB_TEMP_FOLDER"; do
    if ! mkdir -p "$state_dir" || [ ! -w "$state_dir" ]; then
        echo "Runner state directory is not writable: $state_dir" >&2
        exit 2
    fi
done

# Fail before touching notebook outputs when the execution stack is missing.
if ! "$PY" -c '
import importlib

required = (
    "ipykernel", "nbconvert", "jupyter_client", "numpy", "scipy", "pandas",
    "sklearn", "statsmodels", "matplotlib", "joblib", "seaborn", "pyarrow",
    "optuna", "tidyfinance", "xgboost", "lightgbm",
)
failed = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        failed.append(f"{name} ({type(exc).__name__}: {exc})")
if failed:
    raise SystemExit("unusable notebook packages: " + "; ".join(failed))
'; then
    echo "Notebook preflight failed for interpreter '$PY'." >&2
    echo "Install .[notebooks] or set RVLAB_PYTHON explicitly." >&2
    exit 2
fi

# Make kernel selection authoritative. With no override, install a private,
# short-lived kernelspec whose argv[0] is exactly the interpreter just
# preflighted. An explicitly selected kernelspec is accepted only when its
# command resolves to that same interpreter — including generic `python3`
# commands, which are resolved through PATH rather than assumed equivalent.
if [ -n "$REQUESTED_KERNEL" ]; then
    if ! "$PY" -c '
from pathlib import Path
import shutil
import sys
from jupyter_client.kernelspec import KernelSpecManager

kernel = KernelSpecManager().get_kernel_spec(sys.argv[1])
command = kernel.argv[0]
resolved = shutil.which(command) if not Path(command).is_absolute() else command
if not resolved:
    raise SystemExit(f"kernel command does not resolve: {command}")
kernel_python = Path(resolved).expanduser().resolve()
runner_python = Path(sys.executable).resolve()
if kernel_python != runner_python:
    raise SystemExit(
        f"kernel {sys.argv[1]!r} resolves {command!r} to {kernel_python}, "
        f"not RVLAB_PYTHON {runner_python}")
' "$REQUESTED_KERNEL"; then
        echo "Notebook preflight failed for interpreter '$PY' and kernel '$REQUESTED_KERNEL'." >&2
        echo "Unset RVLAB_KERNEL to use an ephemeral kernel pinned to RVLAB_PYTHON." >&2
        exit 2
    fi
    KERNEL="$REQUESTED_KERNEL"
    KERNEL_LABEL="kernel $KERNEL (verified)"
else
    KERNEL="rvlab-runner-$$"
    export JUPYTER_PATH="$KERNEL_PREFIX/share/jupyter${JUPYTER_PATH:+:$JUPYTER_PATH}"
    if ! "$PY" -m ipykernel install --prefix "$KERNEL_PREFIX" --name "$KERNEL" \
            --display-name "RVLab runner ($$)" >/dev/null; then
        echo "Could not create the temporary kernel for '$PY'." >&2
        exit 2
    fi
    KERNEL_LABEL="ephemeral kernel $KERNEL"
fi

# Filenames carry no numeric prefix, so reading order lives in notebooks/legacy/.order —
# one filename per line, and the single place the sequence is written down.
if [ $# -gt 0 ]; then
    NOTEBOOKS=("$@")
else
    NOTEBOOKS=()
    while IFS= read -r line; do
        [ -n "$line" ] && NOTEBOOKS+=("notebooks/legacy/$line")
    done < notebooks/legacy/.order
fi

MODE="real data"
case "${RVLAB_FORCE_SYNTHETIC:-}" in
    ""|0|false|False) ;;
    *) MODE="SYNTHETIC fallback" ;;
esac
echo "Executing ${#NOTEBOOKS[@]} notebooks with $PY / $KERNEL_LABEL  [$MODE]"
if [ -n "$EXECUTED_NOTEBOOK_DIR" ]; then
    if ! mkdir -p "$EXECUTED_NOTEBOOK_DIR" || [ ! -w "$EXECUTED_NOTEBOOK_DIR" ]; then
        echo "Executed-notebook output directory is not writable: $EXECUTED_NOTEBOOK_DIR" >&2
        exit 2
    fi
    EXECUTED_NOTEBOOK_DIR="$(cd "$EXECUTED_NOTEBOOK_DIR" && pwd)"
    echo "Executed notebooks will be written to $EXECUTED_NOTEBOOK_DIR"
    OUTPUT_ARGS=(--output-dir="$EXECUTED_NOTEBOOK_DIR")
else
    OUTPUT_ARGS=(--inplace)
fi
echo

LOG="$(mktemp "${TMPDIR:-/tmp}/rvlab_nbconvert.XXXXXX.log")"

failed=0
for nb in "${NOTEBOOKS[@]}"; do
    if [ ! -f "$nb" ]; then
        echo "  missing notebook: $nb" >&2
        failed=$((failed + 1))
        continue
    fi
    printf '%-58s' "  $(basename "$nb")"
    start=$(date +%s)
    if "$PY" -m jupyter nbconvert --to notebook --execute "${OUTPUT_ARGS[@]}" \
            --ExecutePreprocessor.timeout=180 \
            --ExecutePreprocessor.kernel_name="$KERNEL" \
            "$nb" >"$LOG" 2>&1; then
        echo "ok   $(( $(date +%s) - start ))s"
    else
        echo "FAIL $(( $(date +%s) - start ))s"
        tail -25 "$LOG" | sed 's/^/      /'
        failed=$((failed + 1))
    fi
done

echo
if [ "$failed" -gt 0 ]; then
    echo "$failed of ${#NOTEBOOKS[@]} notebooks failed"
    exit 1
fi
echo "all ${#NOTEBOOKS[@]} notebooks executed cleanly"
