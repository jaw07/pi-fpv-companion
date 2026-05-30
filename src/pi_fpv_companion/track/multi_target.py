"""Multi-object IoU tracker with operator target selection.

The single-target `IouAssociator` auto-locks the highest-confidence detection and
follows it. For an operator who wants to *choose* among several detected targets,
this keeps a stable identity for **every** current detection (so a HUD can show
them all in STANDBY) and tracks which one is **selected**. The operator cycles the
selection with `cycle()` (wired to an RC channel in the pipeline); the selection
is a track id that is maintained frame-to-frame by association and is NOT reset by
the guidance-mode switch — so whatever is locked in STANDBY stays locked through
TRACK and DIVE.

  consume(...) -> the SELECTED target (or None)   # what the servo/filter consume
  .tracks      -> all current tracks              # what the HUD shows
  .cycle()     -> advance the selection to the next track (stable id order)

Greedy IoU association: each existing track claims its highest-IoU unmatched
detection above threshold; unmatched detections spawn new tracks; a track unseen
for more than `max_lost_frames` is dropped. The selected track persists while it
coasts (so a brief miss doesn't drop the lock); when it is finally dropped the
selection falls back to the highest-confidence track (re-acquire) so guidance is
never left pointing at nothing once targets are present.
"""
from __future__ import annotations
import math
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from pi_fpv_companion.types import Detection, Target
from pi_fpv_companion.track.iou_associator import _iou


class MultiObjectTracker:
    def __init__(self, iou_threshold: float = 0.3, max_lost_frames: int = 30,
                 max_match_dist_px: float = 60.0, vel_alpha: float = 0.5) -> None:
        self._iou_threshold = iou_threshold
        self._max_lost_frames = max_lost_frames
        # Associate a detection to a track when it OVERLAPS (IoU) OR its centroid is
        # within this many pixels OF THE TRACK'S PREDICTED position. IoU alone fails
        # for the small boxes this system sees (a person at >100 m is a few px wide,
        # so any camera rotation shifts the box more than its own width → zero IoU).
        # Matching against a constant-velocity PREDICTION (not the last position)
        # also keeps identities through a crossing: two targets that pass each other
        # in the image would otherwise swap ids under nearest-neighbour matching.
        self._max_match_dist_px = max_match_dist_px
        self._vel_alpha = vel_alpha
        self._vel: Dict[int, Tuple[float, float]] = {}   # track_id -> (vx, vy) px/s
        self._last_t: Optional[float] = None
        self._tracks: Dict[int, Target] = {}
        self._selected_id: Optional[int] = None
        self._next_id: int = 1
        # When True, consume() locks the highest-confidence track if nothing valid
        # is selected (acquisition / re-acquisition). The pipeline turns this OFF
        # once engaged (TRACK/DIVE) so that if the COMMITTED target is dropped the
        # aircraft holds (returns None) instead of silently swapping to a different
        # target mid-engagement.
        self.auto_acquire: bool = True

    # ---- state the pipeline / HUD read ----
    @property
    def tracks(self) -> List[Target]:
        """All current tracks, in stable (track-id) order."""
        return [self._tracks[i] for i in sorted(self._tracks)]

    @property
    def selected_id(self) -> Optional[int]:
        return self._selected_id

    def cycle(self) -> Optional[int]:
        """Advance the selection to the next track (wraps). No-op if no tracks."""
        ids = sorted(self._tracks)
        if not ids:
            self._selected_id = None
            return None
        if self._selected_id not in ids:
            self._selected_id = ids[0]
        else:
            self._selected_id = ids[(ids.index(self._selected_id) + 1) % len(ids)]
        return self._selected_id

    def select(self, track_id: Optional[int]) -> None:
        self._selected_id = track_id

    # ---- per-frame update ----
    def consume(
        self, image: object, detections: List[Detection], now: float
    ) -> Optional[Target]:
        self._associate(detections, now)
        # Acquire / re-acquire the highest-confidence track when nothing valid is
        # selected — but only while auto_acquire is on (STANDBY). When engaged it
        # stays off, so a dropped commit yields None (hold) rather than a target swap.
        if self.auto_acquire and self._selected_id not in self._tracks and self._tracks:
            self._selected_id = max(
                self._tracks, key=lambda i: self._tracks[i].detection.confidence
            )
        return self._tracks.get(self._selected_id) if self._selected_id is not None else None

    def _associate(self, detections: List[Detection], now: float) -> None:
        dt = (now - self._last_t) if self._last_t is not None else 0.0
        self._last_t = now
        unmatched = list(detections)
        # Existing tracks claim their best unmatched detection, matched against the
        # track's constant-velocity PREDICTION (carries identity through a crossing).
        for tid in sorted(self._tracks):
            cur = self._tracks[tid]
            vx, vy = self._vel.get(tid, (0.0, 0.0))
            px, py = cur.detection.x + vx * dt, cur.detection.y + vy * dt
            pred = replace(cur.detection, x=px, y=py)
            # Best match: prefer highest IoU (of the predicted box); else nearest
            # centroid to the prediction within the gate (robust for tiny boxes).
            best, best_key = None, None
            for d in unmatched:
                iou = _iou(pred, d)
                dist = math.hypot(px - d.x, py - d.y)
                if iou >= self._iou_threshold or dist <= self._max_match_dist_px:
                    key = (iou, -dist)
                    if best_key is None or key > best_key:
                        best, best_key = d, key
            if best is not None:
                unmatched.remove(best)
                if dt > 1e-3:
                    a = self._vel_alpha
                    self._vel[tid] = (
                        a * (best.x - cur.detection.x) / dt + (1 - a) * vx,
                        a * (best.y - cur.detection.y) / dt + (1 - a) * vy,
                    )
                self._tracks[tid] = Target(detection=best, track_id=tid,
                                           lost_frames=0, timestamp=now)
            else:
                lost = cur.lost_frames + 1
                if lost > self._max_lost_frames:
                    del self._tracks[tid]
                    self._vel.pop(tid, None)
                else:
                    # Coast on the prediction (so a brief miss keeps moving with it).
                    self._tracks[tid] = Target(detection=pred, track_id=tid,
                                               lost_frames=lost, timestamp=cur.timestamp)
        # Unmatched detections become new tracks (velocity seeds at zero).
        for d in unmatched:
            self._tracks[self._next_id] = Target(detection=d, track_id=self._next_id,
                                                 lost_frames=0, timestamp=now)
            self._vel[self._next_id] = (0.0, 0.0)
            self._next_id += 1
