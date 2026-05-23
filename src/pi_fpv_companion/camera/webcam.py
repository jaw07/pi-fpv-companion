"""OpenCV VideoCapture-backed camera.

Wraps the laptop webcam (or any V4L2/AVFoundation source). Yields raw frames;
the Pipeline runs the detector on the configured cadence.
"""
from __future__ import annotations
import time
from typing import Iterator

import cv2

from pi_fpv_companion.camera.base import FrameBundle


class WebcamCamera:
    def __init__(
        self,
        device: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        self._device = device
        self._width = width
        self._height = height
        self._fps = fps
        self._cap = None
        self._running = False

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self._device)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open webcam device {self._device}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        self._running = True

    def close(self) -> None:
        self._running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def frames(self) -> Iterator[FrameBundle]:
        if not self._running:
            self.open()
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            yield FrameBundle(
                image=frame, width=w, height=h,
                timestamp=time.monotonic(), detections=[],
            )
