#!/usr/bin/env bash
set -euo pipefail
SYMBOL=${1:-BTCUSDT}
TIMEFRAME=${2:-1m}
START=${3:-"2023-01-01"}
END=${4:-"2023-02-01"}
python -m bot.scripts.run_backtest_cli "$SYMBOL" "$TIMEFRAME" "$START" "$END"
