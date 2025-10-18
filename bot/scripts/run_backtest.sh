#!/usr/bin/env bash
set -euo pipefail
SYMBOL=${1:-BTCUSDT}
TIMEFRAME=${2:-1m}
START=${3:-"2023-01-01"}
END=${4:-"2023-02-01"}
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
# Allow overriding the interpreter while falling back to python when python3 is unavailable.
"$PYTHON_BIN" -m bot.scripts.run_backtest_cli "$SYMBOL" "$TIMEFRAME" "$START" "$END"
