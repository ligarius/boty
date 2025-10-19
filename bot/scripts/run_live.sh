#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=bot/scripts/_runner.sh
source "$SCRIPT_DIR/_runner.sh"

run_python_module bot.scripts.guard_live
