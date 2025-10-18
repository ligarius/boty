#!/usr/bin/env bash
set -euo pipefail
celery -A bot.exec.celery_app.celery_app worker --loglevel=info
