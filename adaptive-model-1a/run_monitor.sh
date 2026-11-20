#!/bin/bash
# Periodic training health check — run by launchd every 5 minutes.
# Appends output to runs/monitor.log with timestamps.
#
# Usage:
#   ./run_monitor.sh              # Run once (cron/launchd mode)
#   ./run_monitor.sh --watch 300  # Live watch mode (manual)

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="./venv/bin/python"
LOG_FILE="runs/monitor.log"
PID_FILE="runs/training.pid"

# Only run if training is active
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "" >> "$LOG_FILE"
    echo "===== Monitor check: $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$LOG_FILE"
    "$PYTHON" monitor_training.py >> "$LOG_FILE" 2>&1
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training not running, skipping check" >> "$LOG_FILE"
fi
