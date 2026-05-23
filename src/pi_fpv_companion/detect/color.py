"""Color-blob detector — HSV threshold + contour finding.

Cheap, runs in ~1-2 ms on Mac (probably ~10 ms on Pi for VGA). Doesn't require
any model file, which makes it perfect for early Pi-free development.

Drop-in replaceable by `NanoDetDetector` or `Yolov8Detector` — same Detector Protocol.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from pi_fpv_companion.types import Detection


@dataclass(frozen=True)
class HsvRange:
    """Inclusive HSV bounds. OpenCV HSV: H 0-179, S 0-255, V 0-255."""
    h_lo: int
    h_hi: int
    s_lo: int = 80
    s_hi: int = 255
    v_lo: int = 80
    v_hi: int = 255


# Wraps the hue axis (red is at both ends).
RED_RANGES: Tuple[HsvRange, HsvRange] = (
    HsvRange(h_lo=0, h_hi=10),
    HsvRange(h_lo=170, h_hi=179),
)


class ColorBlobDetector:
    def __init__(
        self,
        ranges: Tuple[HsvRange, ...] = RED_RANGES,
        min_area_px: int = 200,
        class_name: str = "target",
    ) -> None:
        self._ranges = ranges
        self._min_area = min_area_px
        self._class_name = class_name

    def detect(self, image) -> List[Detection]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = None
        for r in self._ranges:
            lo = np.array([r.h_lo, r.s_lo, r.v_lo], dtype=np.uint8)
            hi = np.array([r.h_hi, r.s_hi, r.v_hi], dtype=np.uint8)
            m = cv2.inRange(hsv, lo, hi)
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        # Small morphological close to glue near-pixels and drop speckle
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets: List[Detection] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self._min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            # Confidence = area fraction of bounding box, a rough quality proxy
            conf = min(1.0, area / max(1.0, w * h))
            dets.append(Detection(
                x=float(x + w / 2),
                y=float(y + h / 2),
                w=float(w), h=float(h),
                confidence=float(conf),
                class_id=0,
                class_name=self._class_name,
            ))
        return dets
