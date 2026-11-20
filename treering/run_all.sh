#!/bin/bash
# Runs every stream and seed ONE AT A TIME. Never start these in parallel: see safety.py.
cd "$(dirname "$0")"
PY=../adaptive-model-1a/.venv/bin/python
for job in "skills 0" "text 0" "skills 1" "skills 2" "text 1"; do
  set -- $job
  echo "=== $(date '+%F %T') starting $1 seed $2" >> logs/run_all.log
  "$PY" run.py --benchmark "$1" --seed "$2" >> "logs/$1_seed$2.log" 2>&1
  echo "=== $(date '+%F %T') finished $1 seed $2 (exit $?)" >> logs/run_all.log
done
echo "=== $(date '+%F %T') ALL DONE" >> logs/run_all.log
