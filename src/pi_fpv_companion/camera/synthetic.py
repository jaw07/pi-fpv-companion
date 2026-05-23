"""Synthetic camera for Pi-free development.

Generates frames showing a single red "target" rectangle that wanders sinusoidally.
Emits detections alongside frames, modeling the IMX500 path where dense per-frame
detections come straight from the sensor.

Use `render_at(t)` for deterministic tests; use `frames()` for the real-time
generator a `Pipeline` consumes.
"""
from __future__ import annotations
import math
import time
from typing import Iterator, Optional

import numpy as np

from pi_fpv_companion.camera.base import FrameBundle
from pi_fpv_companion.types import Detection


class SyntheticCamera:
    def __init__(
        self,
        width: int = 720,
        height: int = 576,
        fps: int = 20,
        target_size_px: int = 60,
        x_amplitude_frac: float = 0.35,
        y_amplitude_frac: float = 0.15,
        x_period_s: float = 4.0,
        y_period_s: float = 5.5,
    ) -> None:
        self._width = width
        self._height = height
        self._fps = fps
        self._frame_period = 1.0 / fps
        self._target_size = target_size_px
        self._ax = x_amplitude_frac * width
        self._ay = y_amplitude_frac * height
        self._wx = 2 * math.pi / x_period_s
        self._wy = 2 * math.pi / y_period_s
        self._start_time: Optional[float] = None
        self._running = False

    def open(self) -> None:
        self._start_time = time.monotonic()
        self._running = True

    def close(self) -> None:
        self._running = False

    def render_at(self, t: float) -> FrameBundle:
        cx = self._width / 2 + self._ax * math.sin(self._wx * t)
        cy = self._height / 2 + self._ay * math.sin(self._wy * t)

        img = np.full((self._height, self._width, 3), 64, dtype=np.uint8)
        size = self._target_size
        x1 = max(0, int(cx - size / 2))
        y1 = max(0, int(cy - size / 2))
        x2 = min(self._width, int(cx + size / 2))
        y2 = min(self._height, int(cy + size / 2))
        img[y1:y2, x1:x2] = (0, 0, 255)  # BGR red

        det = Detection(
            x=float(cx), y=float(cy),
            w=float(size), h=float(size),
            confidence=0.95, class_id=0, class_name="target",
        )
        return FrameBundle(
            image=img,
            width=self._width,
            height=self._height,
            timestamp=t if self._start_time is None else (self._start_time + t),
            detections=[det],
        )

    def frames(self) -> Iterator[FrameBundle]:
        if not self._running:
            self.open()
        assert self._start_time is not None
        next_due = self._start_time
        while self._running:
            now = time.monotonic()
            if now < next_due:
                time.sleep(next_due - now)
            t = time.monotonic() - self._start_time
            yield self.render_at(t)
            next_due += self._frame_period
