"""Classical OpenCV tracker wrapper. Supports KCF, MOSSE, CSRT, MedianFlow.

This is for the dev file/webcam path (re-seeded by a periodic detector). MOSSE is
the default because:
  - 20x faster than KCF on Pi-class SoCs (1.2 ms vs 22.7 ms at 720x576, measured)
  - MOSSE's scale/occlusion weaknesses are exactly what the periodic detector
    fixes when it re-seeds every N frames

Pick a different backend with `cv2_backend`:
  - "mosse"      : 1 ms,  no scale, peak FPS                 [default]
  - "kcf"        : 23 ms, no scale, decent accuracy
  - "medianflow" : 12 ms, fails on fast motion
  - "csrt"       : 200 ms — unusable on Zero 2W
"""
from __future__ import annotations
from typing import List, Optional

import cv2

from pi_fpv_companion.types import Detection, Target


_FACTORIES = {
    "mosse":      cv2.legacy.TrackerMOSSE_create,
    "kcf":        cv2.legacy.TrackerKCF_create,
    "medianflow": cv2.legacy.TrackerMedianFlow_create,
    "csrt":       cv2.legacy.TrackerCSRT_create,
}


def _closest_detection(detections: List[Detection], to: Detection) -> Detection:
    return min(detections, key=lambda d: (d.x - to.x) ** 2 + (d.y - to.y) ** 2)


class ClassicalCv2Tracker:
    """One of cv2.legacy's classical trackers, with Pipeline-friendly re-seed logic."""

    def __init__(self, cv2_backend: str = "mosse", max_lost_frames: int = 15) -> None:
        backend = cv2_backend.lower()
        if backend not in _FACTORIES:
            raise ValueError(f"unknown cv2 tracker backend {cv2_backend!r}; valid: {list(_FACTORIES)}")
        self._factory = _FACTORIES[backend]
        self._backend_name = backend
        self._max_lost = max_lost_frames
        self._cv_tracker = None
        self._target: Optional[Target] = None
        self._next_track_id: int = 1

    @property
    def backend(self) -> str:
        return self._backend_name

    def consume(
        self, image: object, detections: List[Detection], now: float
    ) -> Optional[Target]:
        if detections:
            return self._seed_from_detections(image, detections, now)
        if self._cv_tracker is None or self._target is None:
            return None
        ok, bbox = self._cv_tracker.update(image)
        if not ok:
            return self._increment_lost(now)

        x, y, w, h = bbox
        det = Detection(
            x=float(x + w / 2),
            y=float(y + h / 2),
            w=float(w),
            h=float(h),
            confidence=self._target.detection.confidence,
            class_id=self._target.detection.class_id,
            class_name=self._target.detection.class_name,
        )
        self._target = Target(
            detection=det,
            track_id=self._target.track_id,
            lost_frames=0,
            timestamp=now,
        )
        return self._target

    def _seed_from_detections(
        self, image: object, detections: List[Detection], now: float
    ) -> Optional[Target]:
        if self._target is not None:
            seed = _closest_detection(detections, self._target.detection)
            track_id = self._target.track_id
        else:
            seed = max(detections, key=lambda d: d.confidence)
            track_id = self._next_track_id
            self._next_track_id += 1

        x = int(seed.x - seed.w / 2)
        y = int(seed.y - seed.h / 2)
        w = int(seed.w)
        h = int(seed.h)
        if w < 4 or h < 4:
            return self._target

        self._cv_tracker = self._factory()
        self._cv_tracker.init(image, (x, y, w, h))
        self._target = Target(
            detection=seed,
            track_id=track_id,
            lost_frames=0,
            timestamp=now,
        )
        return self._target

    def _increment_lost(self, now: float) -> Optional[Target]:
        if self._target is None:
            return None
        new_lost = self._target.lost_frames + 1
        if new_lost > self._max_lost:
            self._target = None
            self._cv_tracker = None
            return None
        self._target = Target(
            detection=self._target.detection,
            track_id=self._target.track_id,
            lost_frames=new_lost,
            timestamp=self._target.timestamp,
        )
        return self._target

    def is_locked(self) -> bool:
        return self._target is not None and self._cv_tracker is not None

    def reset(self) -> None:
        self._target = None
        self._cv_tracker = None
