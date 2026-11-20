"""External kill switch for this project's runs. Uses no MLX and almost no memory.

Purpose
    Watches every "run.py --benchmark" process. If one exceeds LIMIT_GB of physical
    footprint, or the system's free-memory level falls below MIN_FREE_PERCENT, it kills
    all of them and writes why to logs/guard.log. Exits when no run is left.

Docs
    macOS memory status: `sysctl kern.memorystatus_level` (percent of memory available).

Sample input
    python guard.py

Expected output
    Lines in logs/guard.log such as "ok: 1 run, largest 3.9 GB, system 71% free".
"""
import ctypes
import os
import signal
import subprocess
import time

LIMIT_GB = 20.0
MIN_FREE_PERCENT = 25
GB = 2 ** 30


class _RusageInfoV0(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in (
        "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups", "ri_pageins",
        "ri_wired_size", "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime", "ri_proc_exit_abstime")]


def footprint_gb(pid: int) -> float:
    info = _RusageInfoV0()
    lib = ctypes.CDLL("/usr/lib/libproc.dylib")
    return info.ri_phys_footprint / GB if lib.proc_pid_rusage(pid, 0, ctypes.byref(info)) == 0 else 0.0


def run_pids() -> list:
    out = subprocess.run(["pgrep", "-f", "run.py --benchmark"], capture_output=True, text=True).stdout.split()
    return [int(p) for p in out if int(p) != os.getpid()]


def free_percent() -> int:
    out = subprocess.run(["sysctl", "-n", "kern.memorystatus_level"], capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else 100


def log(msg: str) -> None:
    os.makedirs("logs", exist_ok=True)
    with open("logs/guard.log", "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


if __name__ == "__main__":
    log("guard started")
    quiet = 0
    while quiet < 6:  # exit after 30 s with no run to watch
        pids = run_pids()
        if not pids:
            quiet += 1
            time.sleep(5)
            continue
        quiet = 0
        sizes = {p: footprint_gb(p) for p in pids}
        free = free_percent()
        if max(sizes.values()) > LIMIT_GB or free < MIN_FREE_PERCENT:
            for p in pids:
                try:
                    os.kill(p, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            log(f"KILLED {len(pids)} run(s): largest {max(sizes.values()):.1f} GB (limit {LIMIT_GB}), system {free}% free (minimum {MIN_FREE_PERCENT})")
        elif int(time.time()) % 60 < 5:
            log(f"ok: {len(pids)} run(s), largest {max(sizes.values()):.1f} GB, system {free}% free")
        time.sleep(5)
    log("guard exiting: no runs left")
