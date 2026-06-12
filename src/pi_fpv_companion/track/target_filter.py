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
    # Quality = TRUST x STALENESS. Trust is a [0,1] confidence in the lock, moved
    # ONLY by real detections (recovers toward measurement confidence on a good
    # update, penalized on a rejected/implausible one). Staleness is a wall-clock
    # discount, 0.5 ** (age / halflife), where age is the time since the last
    # ACCEPTED measurement. Splitting them makes quality independent of the
    # pipeline frame rate vs the detector's rate: a detector firing at 5.5 Hz into
    # a 30 fps pipeline coasts ~5 frames between detections, and a per-frame decay
    # (the old model) would crater quality in those gaps even though the track is
    # perfectly healthy. Time-based staleness only discounts genuine age.
    quality_recover: float = 0.4    # trust moves this fraction toward meas confidence per accept
    quality_reject_penalty: float = 0.34  # trust subtracted on a rejected/implausible update
    quality_staleness_halflife_s: float = 0.8  # quality halves per this long with no real measurement
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
        self._trust = 0.0    # confidence in the lock (moved only by real detections)
        self._last_t = 0.0
        self._meas_t = 0.0   # time of the last ACCEPTED real measurement

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
        self._trust = max(0.0, min(1.0, d.confidence))
        self._last_t = now
        self._meas_t = now   # a fresh lock counts as a real measurement

    def update(
        self, raw: Optional[Target], frame_w: int, frame_h: int, now: float
    ) -> Optional[FilteredTarget]:
        cfg = self._cfg
        diag = math.hypot(frame_w, frame_h)
        max_jump = cfg.max_jump_frac * diag

        # --- no fresh measurement this tick: either the tracker returned nothing
        # (lost / between detections) OR it is coasting on a frozen box it can no
        # longer confirm (raw.lost_frames > 0). In both cases we coast on the
        # motion model and decay quality — we do NOT pull toward, recover quality
        # from, or advance the measurement clock on an unconfirmed box (audit §1/§5).
        if raw is None or raw.lost_frames > 0:
            if not self._init:
                return None
            dt = max(1e-3, now - self._last_t)
            self._x += self._vx * dt
            self._y += self._vy * dt
            self._last_t = now
            # No trust change while coasting (the box is unconfirmed — never recover
            # from it). Quality falls via time-based staleness (_emit); drop when it
            # crosses the floor. This decays by elapsed TIME, not frame count, so a
            # slow detector's between-detection gaps don't crater a healthy track.
            if self._quality(now) < cfg.drop_below_quality:
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
            # Reject the measurement: coast on prediction, penalize TRUST (an immediate
            # hit, independent of staleness). Do NOT pull toward the teleport / other
            # object, and do NOT advance _meas_t (staleness keeps growing too).
            self._x, self._y = px, py
            self._last_t = now
            self._trust = max(0.0, self._trust - cfg.quality_reject_penalty)
            if self._quality(now) < cfg.drop_below_quality:
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
        self._meas_t = now   # accepted a fresh, plausible measurement (resets staleness)
        # trust recovers toward this measurement's confidence
        target_q = max(0.0, min(1.0, d.confidence))
        self._trust += cfg.quality_recover * (target_q - self._trust)
        self._trust = max(0.0, min(1.0, self._trust))
        return self._emit(now)

    def _quality(self, now: float) -> float:
        """trust x staleness. Staleness = 0.5 ** (age / halflife), age since the last
        accepted measurement — a wall-clock discount, so the score is independent of
        how the detector's rate compares to the pipeline frame rate."""
        age = max(0.0, now - self._meas_t)
        staleness = 0.5 ** (age / self._cfg.quality_staleness_halflife_s)
        return max(0.0, min(1.0, self._trust * staleness))

    def _emit(self, now: float) -> FilteredTarget:
        q = self._quality(now)
        return FilteredTarget(
            detection=Detection(
                x=self._x, y=self._y, w=self._w, h=self._h,
                confidence=q, class_id=self._class_id,
                class_name="",
            ),
            track_id=self._track_id,
            vx_px_s=self._vx,
            vy_px_s=self._vy,
            quality=q,
            timestamp=now,
            measurement_timestamp=self._meas_t,
        )
