#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=bot/scripts/_runner.sh
source "$SCRIPT_DIR/_runner.sh"

SYMBOL=${1:-BTCUSDT}
TIMEFRAME=${2:-1m}
START=${3:-"2023-01-01"}
END=${4:-"2023-02-01"}

run_python_module bot.scripts.run_backtest_cli "$SYMBOL" "$TIMEFRAME" "$START" "$END"
