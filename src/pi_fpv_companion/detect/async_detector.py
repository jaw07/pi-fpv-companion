"""Async wrapper around a Detector: runs inference in a worker thread.

The Pipeline submits frames on its `detect_period_frames` cadence; the worker
processes them as fast as it can; the Pipeline polls for completed results
every frame and feeds them to the tracker when available.

This decouples detector latency from main-loop tick rate. On Pi Zero 2W with
NanoDet @ 256 (221 ms inference), the main loop ticks at 30 FPS continuously
instead of stalling every detect cycle.

NCNN inference releases the GIL during the C++ forward pass, so two A53 cores
end up doing useful work in parallel (main loop + inference worker), while
NCNN's own internal threads use the remaining two.

Latest-wins semantics: if the Pipeline submits a new frame while the worker is
still chewing on the previous one, the previous pending input is discarded —
only the most recent submission matters. Stale submissions waste cycles.
"""
from __future__ import annotations
import threading
from typing import List, Optional

from pi_fpv_companion.detect.base import Detector
from pi_fpv_companion.types import Detection


class AsyncDetector:
    def __init__(self, detector: Detector, cpu_affinity=None) -> None:
        self._detector = detector
        self._cpu_affinity = cpu_affinity
        self._pending_input: Optional[object] = None
        self._latest_output: Optional[List[Detection]] = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._busy = False
        self._error: Optional[BaseException] = None
        self._error_logged = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="async-detector")
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def submit(self, image: object) -> None:
        """Queue an image for detection. Replaces any previous pending input."""
        with self._lock:
            self._pending_input = image
        self._wake.set()

    def poll(self) -> Optional[List[Detection]]:
        """Return the latest completed detection list, or None.

        Failure model (deliberate, for a flight system): if the worker thread
        died on a detector fault, this does NOT raise. A detector bug must not
        take down the pilot's composite video feed. Instead it logs once and
        returns None forever — the tracker then loses its target, the safety
        gate mutes guidance (zeroes intent), and the pilot keeps manual
        control with the video + overlay still live. Loud (logged + visible
        via the muted-overlay reason) but non-fatal.

        `worker_died` exposes the state for tests / status.
        """
        with self._lock:
            if self._error is not None:
                should_log = not self._error_logged
                self._error_logged = True
                err = self._error
                result = None
            else:
                should_log = False
                err = None
                result = self._latest_output
                self._latest_output = None
        if should_log:
            print(
                f"WARN: async detector worker died ({err!r}); continuing "
                f"without detections — guidance will mute on target loss, "
                f"video/overlay unaffected"
            )
        return result

    def worker_died(self) -> bool:
        with self._lock:
            return self._error is not None

    def is_busy(self) -> bool:
        """True if the worker is currently mid-inference. Useful for tests."""
        return self._busy

    def _run(self) -> None:
        # Pin this worker (and NCNN's lazily-spawned pool threads, which inherit
        # this thread's affinity) to the detector core set BEFORE the first
        # detect() so the pool is created already pinned.
        from pi_fpv_companion.cpu_affinity import pin_current_thread
        pin_current_thread(self._cpu_affinity)
        while not self._stop.is_set():
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            if self._stop.is_set():
                return
            with self._lock:
                image = self._pending_input
                self._pending_input = None
            if image is None:
                continue
            self._busy = True
            try:
                detections = self._detector.detect(image)
            except Exception as e:
                # Don't let a detector fault silently kill the worker — record
                # it so poll() can surface it and the pipeline fails loud.
                with self._lock:
                    self._error = e
                return
            finally:
                self._busy = False
            with self._lock:
                self._latest_output = detections
