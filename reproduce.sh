#!/usr/bin/env bash
# Reproduce every reported table and figure (R7). Wrapper around reproduce.py:
#   ./reproduce.sh            # fresh venv (.venv-repro), pinned packages, data download, full run
#   ./reproduce.sh --no-venv  # use the current Python environment
# Extra arguments after the flags are passed to reproduce.py (e.g. --phases R4).
# Works in Git Bash on Windows, and on Linux / macOS.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python}
if [[ "${1:-}" == "--no-venv" ]]; then
  shift
else
  if [[ ! -d .venv-repro ]]; then
    "$PY" -m venv .venv-repro
  fi
  if [[ -x .venv-repro/Scripts/python ]]; then PY=.venv-repro/Scripts/python; else PY=.venv-repro/bin/python; fi
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q -r requirements-lock.txt
fi

want=$(cat .python-version)
have=$("$PY" -c 'import platform; print(platform.python_version())')
[[ "$have" == "$want" ]] || echo "warning: Python $have, the results were produced with $want" >&2

"$PY" -m src.data.download
"$PY" reproduce.py "$@"
