"""CPU core pinning so the NCNN detector worker can't starve the camera
capture + main pipeline loop.

On a 4-core Pi Zero 2W, NanoDet inference (2 NCNN threads, ~220 ms) run
flat-out and the OS spreads them across all 4 cores, descheduling the camera
capture thread and main loop — measured wall-clock ticks ballooned to ~90 ms
(content refresh ~7 FPS) even though the main-loop *work* is only ~3-5 ms.

Pinning the detector to a dedicated core set (and everything else to the
remaining cores) removes that contention. Linux-only (`os.sched_setaffinity`);
silent no-op on macOS dev hosts and on boards with < 4 cores.
"""
from __future__ import annotations
import os
from typing import Optional, Set, Tuple


def compute_split(enabled: bool) -> Tuple[Optional[Set[int]], Optional[Set[int]]]:
    """Return (pipeline_cores, detector_cores), or (None, None) to skip pinning.

    Splits cores in half: lower half = pipeline (capture + loop + output),
    upper half = detector (NCNN). 4 cores -> pipeline {0,1}, detector {2,3}.
    """
    if not enabled or not hasattr(os, "sched_setaffinity"):
        return None, None
    n = os.cpu_count() or 1
    if n < 4:
        return None, None
    half = n // 2
    pipeline_cores = set(range(0, n - half))
    detector_cores = set(range(n - half, n))
    return pipeline_cores, detector_cores


def pin_current_thread(cores: Optional[Set[int]]) -> None:
    """Pin the calling thread (and pthreads it later spawns, e.g. NCNN's pool)
    to `cores`. No-op if cores is None or the platform lacks the syscall."""
    if cores and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, cores)
        except (OSError, ValueError):
            pass  # affinity is an optimization, never fatal
