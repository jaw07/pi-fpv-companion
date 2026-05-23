"""Haar-cascade face detector using OpenCV's bundled `haarcascade_frontalface_default.xml`.

Real neural-free detection that works out of the box — useful for end-to-end
pipeline validation on real webcam footage before a proper NCNN model is wired up.

Latency reference points:
    Mac M-series @ 640x480       ~3-6 ms
    Pi Zero 2W   @ 640x480 (est) ~30-60 ms (10x slower, NEON helps marginally)
    Pi Zero 2W   @ 320x240 (est) ~8-20 ms (use `downscale=0.5`)

Drop-in replaceable by `NanoDetDetector` / `Yolov8Detector` — same `Detector` Protocol.
"""
from __future__ import annotations
from typing import List

import cv2

from pi_fpv_companion.types import Detection


class HaarFaceDetector:
    def __init__(
        self,
        scale_factor: float = 1.2,
        min_neighbors: int = 5,
        min_size_px: int = 40,
        downscale: float = 1.0,        # 0.5 = detect at half res, scale results back up
    ) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"haar cascade load failed: {cascade_path}")
        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        self._min_size_px = min_size_px
        self._downscale = downscale

    def detect(self, image) -> List[Detection]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self._downscale != 1.0:
            gray = cv2.resize(
                gray,
                (int(gray.shape[1] * self._downscale), int(gray.shape[0] * self._downscale)),
            )
        rects = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=(self._min_size_px, self._min_size_px),
        )
        upscale = 1.0 / self._downscale if self._downscale != 1.0 else 1.0
        out: List[Detection] = []
        for (x, y, w, h) in rects:
            x, y, w, h = x * upscale, y * upscale, w * upscale, h * upscale
            out.append(Detection(
                x=float(x + w / 2), y=float(y + h / 2),
                w=float(w), h=float(h),
                confidence=0.9,        # Haar gives no scalar confidence; use a fixed sentinel
                class_id=0, class_name="face",
            ))
        return out
