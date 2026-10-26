#!/usr/bin/env bash
set -euo pipefail
PY="${RVLAB_PYTHON:-python3}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
exec "$PY" "$REPO/scripts/run_notebooks.py" "$@"
