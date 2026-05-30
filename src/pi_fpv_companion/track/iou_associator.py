"""Single-target IoU associator for the IMX500 path.

When the camera produces dense per-frame detections (IMX500 sensor NPU), the
expensive part is already done. We just need to keep one consistent identity
across frames. Pick the highest-confidence detection on lock; on each frame,
associate to the detection with the highest IoU above a threshold; if no match,
count a lost frame and drop after N misses.
"""
from __future__ import annotations
import math
from dataclasses import replace
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


def best_match(pred: Detection, candidates: List[Detection],
               iou_threshold: float, max_dist_px: float) -> Optional[Detection]:
    """Pick the candidate matching a track's PREDICTED box: highest IoU, else the
    nearest centroid within the distance gate. IoU alone fails for the small boxes
    this system sees (a person at >100 m is a few px wide, so any camera rotation
    shifts the box more than its own width → zero IoU); the distance gate covers
    that, and matching the prediction (not the last position) keeps identities
    through a crossing. Shared by IouAssociator and MultiObjectTracker."""
    best, best_key = None, None
    for d in candidates:
        iou = _iou(pred, d)
        dist = math.hypot(pred.x - d.x, pred.y - d.y)
        if iou >= iou_threshold or dist <= max_dist_px:
            key = (iou, -dist)
            if best_key is None or key > best_key:
                best, best_key = d, key
    return best


class IouAssociator:
    def __init__(self, iou_threshold: float = 0.3, max_lost_frames: int = 30,
                 max_match_dist_px: float = 60.0, vel_alpha: float = 0.5) -> None:
        self._iou_threshold = iou_threshold
        self._max_lost_frames = max_lost_frames
        self._max_match_dist_px = max_match_dist_px
        self._vel_alpha = vel_alpha
        self._target: Optional[Target] = None
        self._vx = self._vy = 0.0          # px/s, for constant-velocity prediction
        self._last_t: Optional[float] = None
        self._next_track_id: int = 1

    def consume(
        self, image: object, detections: List[Detection], now: float
    ) -> Optional[Target]:
        dt = (now - self._last_t) if self._last_t is not None else 0.0
        self._last_t = now
        if self._target is None:
            # Not locked. Acquire on the highest-confidence detection if any.
            if not detections:
                return None
            seed = max(detections, key=lambda d: d.confidence)
            self._target = Target(detection=seed, track_id=self._next_track_id,
                                  lost_frames=0, timestamp=now)
            self._vx = self._vy = 0.0
            self._next_track_id += 1
            return self._target

        # Locked — match against the constant-velocity prediction (IoU or distance).
        cur = self._target.detection
        pred = replace(cur, x=cur.x + self._vx * dt, y=cur.y + self._vy * dt)
        match = best_match(pred, detections, self._iou_threshold, self._max_match_dist_px) \
            if detections else None
        if match is None:
            new_lost = self._target.lost_frames + 1
            if new_lost > self._max_lost_frames:
                self._target = None
                return None
            # Coast on the prediction (keeps moving through a brief miss).
            self._target = Target(detection=pred, track_id=self._target.track_id,
                                  lost_frames=new_lost, timestamp=self._target.timestamp)
            return self._target
        if dt > 1e-3:
            a = self._vel_alpha
            self._vx = a * (match.x - cur.x) / dt + (1 - a) * self._vx
            self._vy = a * (match.y - cur.y) / dt + (1 - a) * self._vy
        self._target = Target(detection=match, track_id=self._target.track_id,
                              lost_frames=0, timestamp=now)
        return self._target

    def is_locked(self) -> bool:
        return self._target is not None

    def reset(self) -> None:
        self._target = None
        self._vx = self._vy = 0.0
        self._last_t = None
