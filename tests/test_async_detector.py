"""Tests for the AsyncDetector worker. Uses time.sleep + threading.Event
synchronization to avoid flaky timing assertions where possible."""
from __future__ import annotations
import threading
import time
from typing import List

import numpy as np

from pi_fpv_companion.detect.async_detector import AsyncDetector
from pi_fpv_companion.types import Detection


class _SlowDetector:
    """Detector stub with controllable latency. Posts a known detection list per call."""
    def __init__(self, latency_s: float = 0.02) -> None:
        self.latency_s = latency_s
        self.call_count = 0
        self.completed = threading.Event()

    def detect(self, image) -> List[Detection]:
        time.sleep(self.latency_s)
        self.call_count += 1
        det = Detection(
            x=float(self.call_count * 10), y=10.0, w=20.0, h=20.0,
            confidence=0.9, class_id=0, class_name="t",
        )
        self.completed.set()
        return [det]


def _image():
    return np.zeros((10, 10, 3), dtype=np.uint8)


def test_start_then_stop_cleanly():
    d = _SlowDetector()
    ad = AsyncDetector(d)
    ad.start()
    ad.stop()
    # No hangs, no exceptions


def test_submit_then_poll_returns_result():
    d = _SlowDetector(latency_s=0.01)
    ad = AsyncDetector(d)
    ad.start()
    try:
        ad.submit(_image())
        assert d.completed.wait(timeout=1.0), "worker never completed"
        # Give the worker a moment to post the result after detect() returns
        for _ in range(50):
            result = ad.poll()
            if result is not None:
                break
            time.sleep(0.005)
        assert result is not None
        assert len(result) == 1
        assert result[0].x == 10.0
    finally:
        ad.stop()


def test_poll_returns_none_when_no_work_submitted():
    d = _SlowDetector()
    ad = AsyncDetector(d)
    ad.start()
    try:
        time.sleep(0.05)
        assert ad.poll() is None
    finally:
        ad.stop()


def test_poll_clears_result_slot():
    d = _SlowDetector(latency_s=0.01)
    ad = AsyncDetector(d)
    ad.start()
    try:
        ad.submit(_image())
        # Wait for result to land
        for _ in range(50):
            result = ad.poll()
            if result is not None:
                break
            time.sleep(0.005)
        assert result is not None
        # Second poll without new submit returns None
        assert ad.poll() is None
    finally:
        ad.stop()


def test_latest_wins_when_submits_outpace_worker():
    """If the worker is still processing image A and we submit B, the next
    cycle should process B (latest), not the original A again."""
    d = _SlowDetector(latency_s=0.05)
    ad = AsyncDetector(d)
    ad.start()
    try:
        # Submit a burst — worker can only handle one at a time
        for _ in range(5):
            ad.submit(_image())
        # Wait long enough for worker to finish at least one call
        time.sleep(0.3)
        # call_count should be small (probably 2: one immediate, one after burst settled)
        assert d.call_count <= 3, f"worker ran detect {d.call_count}× — should drop stale submissions"
    finally:
        ad.stop()


class _ExplodingDetector:
    def detect(self, image):
        raise RuntimeError("simulated NCNN fault")


def test_worker_death_is_non_fatal(capsys):
    """A dead detector worker must NOT raise — that would kill the pilot's
    composite video feed. poll() returns None, worker_died() True, logged once."""
    ad = AsyncDetector(_ExplodingDetector())
    ad.start()
    try:
        ad.submit(_image())
        deadline = time.time() + 2.0
        while time.time() < deadline and not ad.worker_died():
            time.sleep(0.01)
        assert ad.worker_died(), "worker should have recorded the fault"
        assert ad.poll() is None          # must NOT raise
        assert ad.poll() is None          # stays None
        assert "worker died" in capsys.readouterr().out   # loud
        for _ in range(5):                # but logged only once
            ad.poll()
        assert "worker died" not in capsys.readouterr().out
    finally:
        ad.stop()
