#!/bin/bash
# Launch training as a fully daemonized process.
#
# Uses the classic Unix double-fork pattern via a tiny Python helper
# to create a true daemon that is completely detached from the
# launching process tree. This prevents ANY cleanup mechanism
# from killing the training process.
#
# Usage:
#   ./launch_training.sh                          # Fresh training run
#   ./launch_training.sh --resume runs/XXX/interrupted  # Resume from checkpoint
#
# Monitor: tail -f runs/training.log
# Stop:    kill $(cat runs/training.pid)

set -euo pipefail

cd "$(dirname "$0")"

PYTHON="./venv/bin/python"
LOG_FILE="runs/training.log"
PID_FILE="runs/training.pid"

mkdir -p runs

# Double-fork daemon: the grandchild process has init (PID 1) as parent,
# its own session ID, and its own process group. No ancestor can reach it.
"$PYTHON" -c "
import os, sys

# First fork
pid = os.fork()
if pid > 0:
    # Parent exits immediately — Claude Code sees this process end
    sys.exit(0)

# Child: create new session (new SID + new PGID)
os.setsid()

# Second fork
pid = os.fork()
if pid > 0:
    # First child exits — grandchild is now a true daemon
    sys.exit(0)

# Grandchild: redirect stdio to log file
log_fd = os.open('$LOG_FILE', os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
os.dup2(log_fd, 1)  # stdout -> log
os.dup2(log_fd, 2)  # stderr -> log
os.close(log_fd)
devnull = os.open(os.devnull, os.O_RDONLY)
os.dup2(devnull, 0)  # stdin -> /dev/null
os.close(devnull)

# Write PID file
with open('$PID_FILE', 'w') as f:
    f.write(str(os.getpid()))

# Set unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

# Exec the training script (replaces this process)
os.execv('$PYTHON', [
    '$PYTHON', 'mlx_train.py',
    '--tokenizer-path', 'runs/tokenizer_32k.json',
    '--total-steps', '60000',
] + sys.argv[1:])
" "$@"

# Read the PID that the daemon wrote
sleep 1
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    echo "Training launched (PID: $PID)"
    echo "  Log:  tail -f $LOG_FILE"
    echo "  Stop: kill \$(cat $PID_FILE)"
else
    echo "ERROR: PID file not created"
    exit 1
fi
