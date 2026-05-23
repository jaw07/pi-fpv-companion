"""Plays a video file as a Camera. Yields raw frames; the Pipeline runs the
detector on the configured cadence.

Loops the video on EOF so demos can run indefinitely.
"""
from __future__ import annotations
import time
from typing import Iterator, Optional

import cv2

from pi_fpv_companion.camera.base import FrameBundle


class FileCamera:
    def __init__(
        self,
        path: str,
        fps_override: Optional[float] = None,
        loop: bool = True,
    ) -> None:
        self._path = path
        self._fps_override = fps_override
        self._loop = loop
        self._cap = None
        self._fps = 20.0
        self._running = False

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open video {self._path!r}")
        self._fps = self._fps_override or self._cap.get(cv2.CAP_PROP_FPS) or 20.0
        self._running = True

    def close(self) -> None:
        self._running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def frames(self) -> Iterator[FrameBundle]:
        if not self._running:
            self.open()
        period = 1.0 / self._fps
        next_due = time.monotonic()
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                if not self._loop:
                    return
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            now = time.monotonic()
            if now < next_due:
                time.sleep(next_due - now)
            h, w = frame.shape[:2]
            yield FrameBundle(
                image=frame, width=w, height=h,
                timestamp=time.monotonic(), detections=[],
            )
            next_due += period
