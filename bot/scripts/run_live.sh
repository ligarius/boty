#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
# Allow overriding the interpreter while falling back to python when python3 is unavailable.
"$PYTHON_BIN" -m bot.scripts.guard_live
