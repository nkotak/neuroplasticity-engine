"""Start a command as a fully detached background process that survives the launching session.

Purpose
    Forks twice and starts a new session, so the command has no tie to the launcher, and
    keeps the Mac awake while it runs. Used for run_all.sh (the runs, one at a time) and
    guard.py (the external memory kill switch).

Docs
    Python os.fork / os.setsid: https://docs.python.org/3/library/os.html

Sample input
    python launch.py logs/run_all.out /bin/bash run_all.sh

Expected output
    Prints "launched ..." and returns at once; the command's output goes to the log file.
"""
import os
import sys

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: launch.py LOGFILE COMMAND [ARGS...]")
    log, cmd = sys.argv[1], sys.argv[2:]
    if os.fork() > 0:
        os.wait()
        print(f"launched: {' '.join(cmd)} -> {log}")
        sys.exit(0)
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
    os.execv("/usr/bin/caffeinate", ["caffeinate", "-i"] + cmd)
