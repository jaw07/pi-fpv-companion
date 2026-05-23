"""Detector abstraction.

Implementations:

  NanoDetDetector    — NanoDet-Plus via NCNN on the Pi CPU. Pareto-optimal for Zero 2W.
  Yolov8Detector     — YOLOv8 via NCNN (alternative to NanoDet)
  HaarFaceDetector   — OpenCV bundled Haar cascade; webcam dev validation
  ColorBlobDetector  — HSV mask + contour finder; no model file required, dev only

On the IMX500 path no Detector is used — detections are produced by the camera
itself and surface inline in the FrameBundle. Pipeline runs a Detector only
when the bundle's detections list is empty.
"""
from __future__ import annotations
from typing import List, Protocol

from pi_fpv_companion.types import Detection


class Detector(Protocol):
    def detect(self, image: object) -> List[Detection]:
        """Run detection on one BGR uint8 image. Returns 0+ Detections."""
        ...
