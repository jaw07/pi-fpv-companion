"""Single-target IoU associator for the IMX500 path.

When the camera produces dense per-frame detections (IMX500 sensor NPU), the
expensive part is already done. We just need to keep one consistent identity
across frames. Pick the highest-confidence detection on lock; on each frame,
associate to the detection with the highest IoU above a threshold; if no match,
count a lost frame and drop after N misses.
"""
from __future__ import annotations
from typing import List, Optional

from pi_fpv_companion.types import Detection, Target


def _iou(a: Detection, b: Detection) -> float:
    ax1, ay1 = a.x - a.w / 2, a.y - a.h / 2
    ax2, ay2 = a.x + a.w / 2, a.y + a.h / 2
    bx1, by1 = b.x - b.w / 2, b.y - b.h / 2
    bx2, by2 = b.x + b.w / 2, b.y + b.h / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


class IouAssociator:
    def __init__(self, iou_threshold: float = 0.3, max_lost_frames: int = 30) -> None:
        self._iou_threshold = iou_threshold
        self._max_lost_frames = max_lost_frames
        self._target: Optional[Target] = None
        self._next_track_id: int = 1

    def consume(
        self, image: object, detections: List[Detection], now: float
    ) -> Optional[Target]:
        if self._target is None:
            # Not locked. Acquire on the highest-confidence detection if any.
            if not detections:
                return None
            seed = max(detections, key=lambda d: d.confidence)
            self._target = Target(
                detection=seed,
                track_id=self._next_track_id,
                lost_frames=0,
                timestamp=now,
            )
            self._next_track_id += 1
            return self._target

        # Locked — associate by IoU.
        if not detections:
            return self._increment_lost(now)

        best_iou = 0.0
        best_det: Optional[Detection] = None
        for d in detections:
            iou = _iou(self._target.detection, d)
            if iou > best_iou:
                best_iou = iou
                best_det = d

        if best_det is None or best_iou < self._iou_threshold:
            return self._increment_lost(now)

        self._target = Target(
            detection=best_det,
            track_id=self._target.track_id,
            lost_frames=0,
            timestamp=now,
        )
        return self._target

    def _increment_lost(self, now: float) -> Optional[Target]:
        assert self._target is not None
        new_lost = self._target.lost_frames + 1
        if new_lost > self._max_lost_frames:
            self._target = None
            return None
        self._target = Target(
            detection=self._target.detection,
            track_id=self._target.track_id,
            lost_frames=new_lost,
            timestamp=self._target.timestamp,
        )
        return self._target

    def is_locked(self) -> bool:
        return self._target is not None

    def reset(self) -> None:
        self._target = None
