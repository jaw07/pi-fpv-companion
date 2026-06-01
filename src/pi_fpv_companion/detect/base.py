"""Detector abstraction.

Implementations (dev/sim only — the flight camera does inference on-sensor):

  HaarFaceDetector   — OpenCV bundled Haar cascade; webcam dev validation
  ColorBlobDetector  — HSV mask + contour finder; no model file (Gazebo SITL sim)
  ArucoDetector      — fiducial markers; Webots/SITL bring-up

On the IMX500 (flight) path no Detector is used — detections are produced by the
camera itself and surface inline in the FrameBundle. Pipeline runs a Detector
only when the bundle's detections list is empty (file/webcam dev cameras).
"""
from __future__ import annotations
from typing import List, Protocol

from pi_fpv_companion.types import Detection


class Detector(Protocol):
    def detect(self, image: object) -> List[Detection]:
        """Run detection on one BGR uint8 image. Returns 0+ Detections."""
        ...
