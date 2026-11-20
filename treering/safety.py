"""Hard memory limits for every MLX process in this project.

Purpose
    On 2026-11-19 and 2026-11-20 this project's runs crashed the machine twice. MLX keeps
    freed GPU buffers in a cache whose default limit here is 91 GB per process; code whose
    array shapes changed every step filled it, several processes ran at once, and the Mac
    ran out of memory and was force-restarted by the kernel watchdog.

    Importing this module caps MLX's working memory and its buffer cache. `check()` aborts
    the process with a clear error long before it can threaten the machine. Every entry
    point imports model.py, which imports this, so no run can start without the caps.

Docs
    MLX memory management: https://ml-explore.github.io/mlx/build/html/python/memory_management.html

Sample input
    import safety; safety.check("after dreaming")

Expected output
    Returns the MLX memory in use, in GB; raises MemoryError above ABORT_GB.
"""
import ctypes
import os

import mlx.core as mx

GB = 2 ** 30
MEMORY_LIMIT_GB = 12.0   # MLX's own working-memory guideline
CACHE_LIMIT_GB = 1.0     # freed buffers kept for reuse; this is what grew to ~70 GB
ABORT_GB = 16.0          # the process stops itself above this


def apply_limits(memory_gb: float = MEMORY_LIMIT_GB, cache_gb: float = CACHE_LIMIT_GB) -> None:
    mx.set_memory_limit(int(memory_gb * GB))
    mx.set_cache_limit(int(cache_gb * GB))


def used_gb() -> float:
    """Memory MLX currently holds: live arrays plus cached free buffers."""
    return (mx.get_active_memory() + mx.get_cache_memory()) / GB


class _RusageInfoV0(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in (
        "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups", "ri_pageins",
        "ri_wired_size", "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime", "ri_proc_exit_abstime")]


def footprint_gb(pid: int = 0) -> float:
    """Physical memory footprint of a process as the kernel accounts it (includes GPU buffers).

    This is the figure that reached 68-79 GB per process in the two crashes.
    """
    info = _RusageInfoV0()
    lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    if lib.proc_pid_rusage(pid or os.getpid(), 0, ctypes.byref(info)) != 0:
        return -1.0
    return info.ri_phys_footprint / GB


def check(where: str = "", abort_gb: float = ABORT_GB) -> float:
    used, foot = used_gb(), footprint_gb()
    if used > abort_gb or foot > abort_gb:
        raise MemoryError(f"memory guard tripped {where}: MLX holds {used:.1f} GB, process footprint {foot:.1f} GB, limit {abort_gb} GB")
    return max(used, foot)


apply_limits()


if __name__ == "__main__":
    import sys

    failures, total_tests = [], 0

    total_tests += 1  # the caps are actually in force
    prev_mem = mx.set_memory_limit(int(MEMORY_LIMIT_GB * GB)) / GB
    prev_cache = mx.set_cache_limit(int(CACHE_LIMIT_GB * GB)) / GB
    if abs(prev_mem - MEMORY_LIMIT_GB) > 0.01 or abs(prev_cache - CACHE_LIMIT_GB) > 0.01:
        failures.append(f"limits not applied on import: memory {prev_mem:.1f} GB, cache {prev_cache:.1f} GB")

    total_tests += 1  # shape-changing work must not grow the cache past its cap
    for n in range(1, 400):
        x = mx.random.normal((n * 40, 256))
        mx.eval((x @ x.T).sum())
    cache = mx.get_cache_memory() / GB
    if cache > CACHE_LIMIT_GB + 0.25:
        failures.append(f"cache grew to {cache:.2f} GB despite the {CACHE_LIMIT_GB} GB cap")

    total_tests += 1  # the guard trips when asked to enforce an impossible limit
    try:
        check("self-test", abort_gb=0.0)
        failures.append("guard did not trip at a 0 GB limit")
    except MemoryError:
        pass

    total_tests += 1
    if check("self-test") > ABORT_GB:
        failures.append("normal use is already over the abort limit")

    total_tests += 1  # the kernel footprint must see GPU memory, or the guard would be blind
    big = mx.zeros((3 * GB // 4,), dtype=mx.float32) + 1.0   # hold 3 GB
    mx.eval(big)
    held = footprint_gb()
    if held < 2.5:
        failures.append(f"footprint is {held:.2f} GB while holding a 3 GB array: the guard cannot see GPU memory")
    del big
    mx.clear_cache()
    print(f"cache after shape churn {cache:.2f} GB (cap {CACHE_LIMIT_GB}); footprint while holding a 3 GB array {held:.2f} GB")

    if failures:
        print(f"VALIDATION FAILED - {len(failures)} of {total_tests} tests failed:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print(f"VALIDATION PASSED - all {total_tests} tests produced expected results")
