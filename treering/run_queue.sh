#!/bin/bash
# Remaining seeds, ONE AT A TIME (see safety.py). Variants already shown to be dead ends on
# skills seed 0 are skipped here; their seed-0 results stay in the report.
cd "$(dirname "$0")"
PY=../adaptive-model-1a/.venv/bin/python
SKIP="treering_dreams,treering_splice,selfdream_splice,ewc_1000,compaction"
for job in "text 0" "skills 1" "skills 2" "text 1"; do
  set -- $job
  echo "=== $(date '+%F %T') starting $1 seed $2" >> logs/run_all.log
  "$PY" run.py --benchmark "$1" --seed "$2" --skip "$SKIP" >> "logs/$1_seed$2.log" 2>&1
  echo "=== $(date '+%F %T') finished $1 seed $2 (exit $?)" >> logs/run_all.log
done
echo "=== $(date '+%F %T') QUEUE DONE" >> logs/run_all.log
