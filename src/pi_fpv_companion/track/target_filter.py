"""Alpha-beta target filter + track-quality / wrong-target gating.

Sits between the appearance/IoU tracker and the visual servo. The raw tracker
(MOSSE/KCF/IoU) only knows "this patch looks like the patch I locked" — it
has no motion model and cannot tell that it has drifted onto background, that
the detector handed it a different object, or that a centroid teleported
across the frame (a misdetection). For a system whose job is "fly the drone
at what it sees," acting on a confidently-wrong track is the dominant hazard
(architecture-audit.md §5).

This module adds:

  - a 4-state constant-velocity **alpha-beta filter** on the centroid -> a
    smoothed position + an image-plane velocity estimate (the velocity feeds
    the servo's feedforward term, removing the structural pursuit lag of a
    pure-P loop chasing a moving target; audit §4).
  - **innovation / motion-plausibility gating**: a measurement that lands
    implausibly far from the prediction (a teleport) is rejected as an
    outlier and degrades quality instead of being acted on.
  - **class-consistency gating**: a measurement whose class differs from the
    locked class degrades quality (the tracker re-acquired a different object).
  - a `quality` score that decays on rejections / coasting and recovers on
    consistent, plausible, confident measurements. Below the safety gate's
    floor, guidance is muted.

Pure / deterministic — fully unit-testable, no hardware.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

from pi_fpv_companion.types import Detection, FilteredTarget, Target


@dataclass(frozen=True)
class FilterConfig:
    alpha: float = 0.5            # position correction gain
    beta: float = 0.2            # velocity correction gain
    size_alpha: float = 0.3      # light low-pass on bbox size (closure proxy)
    # Motion plausibility: reject a measurement whose jump from the prediction
    # exceeds this fraction of the frame diagonal (a teleport = misdetection).
    max_jump_frac: float = 0.25
    # Quality dynamics (per-update multiplicative/additive in [0,1]):
    quality_recover: float = 0.25   # toward measurement confidence on a good update
    quality_reject_penalty: float = 0.34  # subtracted on a rejected/implausible update
    quality_coast_decay: float = 0.2      # multiplicative when coasting (no measurement)
    drop_below_quality: float = 0.05      # fully drop the track under this


class AlphaBetaTargetFilter:
    """One filter instance per locked target. Re-initializes when the tracker
    hands over a different track_id."""

    def __init__(self, cfg: Optional[FilterConfig] = None) -> None:
        self._cfg = cfg or FilterConfig()
        self._init = False
        self._track_id: int = -1
        self._class_id: int = -1
        self._x = self._y = 0.0
        self._vx = self._vy = 0.0
        self._w = self._h = 0.0
        self._quality = 0.0
        self._last_t = 0.0

    def reset(self) -> None:
        self._init = False

    def is_active(self) -> bool:
        return self._init

    def _seed(self, d: Detection, track_id: int, now: float) -> None:
        self._init = True
        self._track_id = track_id
        self._class_id = d.class_id
        self._x, self._y = d.x, d.y
        self._vx = self._vy = 0.0
        self._w, self._h = d.w, d.h
        self._quality = max(0.0, min(1.0, d.confidence))
        self._last_t = now

    def update(
        self, raw: Optional[Target], frame_w: int, frame_h: int, now: float
    ) -> Optional[FilteredTarget]:
        cfg = self._cfg
        diag = math.hypot(frame_w, frame_h)
        max_jump = cfg.max_jump_frac * diag

        # --- no measurement this tick (tracker lost / between detections) ---
        if raw is None:
            if not self._init:
                return None
            dt = max(1e-3, now - self._last_t)
            self._x += self._vx * dt
            self._y += self._vy * dt
            self._last_t = now
            self._quality *= (1.0 - cfg.quality_coast_decay)
            if self._quality < cfg.drop_below_quality:
                self._init = False
                return None
            return self._emit(now)

        d = raw.detection

        # --- (re)acquire: first lock, or tracker switched to a new id ---
        if not self._init or raw.track_id != self._track_id:
            self._seed(d, raw.track_id, now)
            return self._emit(now)

        dt = max(1e-3, now - self._last_t)
        # predict
        px = self._x + self._vx * dt
        py = self._y + self._vy * dt
        rx, ry = d.x, d.y
        innov = math.hypot(rx - px, ry - py)

        implausible_jump = innov > max_jump
        class_flip = d.class_id != self._class_id

        if implausible_jump or class_flip:
            # Reject the measurement: coast on prediction, penalize quality.
            # Do NOT pull the track toward a teleport / different object.
            self._x, self._y = px, py
            self._last_t = now
            self._quality = max(0.0, self._quality - cfg.quality_reject_penalty)
            if self._quality < cfg.drop_below_quality:
                self._init = False
                return None
            return self._emit(now)

        # accept: alpha-beta correction
        res_x = rx - px
        res_y = ry - py
        self._x = px + cfg.alpha * res_x
        self._y = py + cfg.alpha * res_y
        self._vx += (cfg.beta / dt) * res_x
        self._vy += (cfg.beta / dt) * res_y
        self._w += cfg.size_alpha * (d.w - self._w)
        self._h += cfg.size_alpha * (d.h - self._h)
        self._last_t = now
        # quality recovers toward this measurement's confidence
        target_q = max(0.0, min(1.0, d.confidence))
        self._quality += cfg.quality_recover * (target_q - self._quality)
        self._quality = max(0.0, min(1.0, self._quality))
        return self._emit(now)

    def _emit(self, now: float) -> FilteredTarget:
        return FilteredTarget(
            detection=Detection(
                x=self._x, y=self._y, w=self._w, h=self._h,
                confidence=self._quality, class_id=self._class_id,
                class_name="",
            ),
            track_id=self._track_id,
            vx_px_s=self._vx,
            vy_px_s=self._vy,
            quality=self._quality,
            timestamp=now,
        )
