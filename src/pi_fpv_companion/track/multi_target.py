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
coasts (so a brief miss doesn't drop the lock).

Selection stickiness — once the OPERATOR has chosen a target (cycle/select), the
choice is sticky and is never silently swapped for a different one:
  - while the chosen track coasts, it stays locked (as above);
  - if the chosen track id is finally dropped but a detection re-appears near where
    the target was, the selection RE-BINDS to that same physical target (handles
    real-world id churn: a flickery detection that re-spawns under a fresh id);
  - if nothing re-appears nearby, consume() returns None (HOLD) — it does NOT jump
    to the highest-confidence target, so a deliberate pick is never overridden.
Auto-acquire of the highest-confidence track happens ONLY before any manual pick
(initial acquisition) or after the scene fully clears.

Spurious-detection suppression (added after the first flight, where aerial false
positives appeared and coasted off-frame):
  - M-of-N confirmation: a track is only SHOWN/selectable once it has been detected
    in >= `confirm_hits` of the last `confirm_window` frames (latched once met). A
    one- or two-frame ghost never confirms, so it never reaches the HUD or the lock.
    `confirm_hits=1` disables it (default, for back-compat; production sets it >1).
  - Off-frame drop: a coasting track whose PREDICTED centre leaves the frame is
    dropped immediately instead of gliding off-screen on its velocity estimate.
"""
from __future__ import annotations
from collections import deque
from dataclasses import replace
from typing import Deque, Dict, List, Optional, Tuple

from pi_fpv_companion.types import Detection, Target
from pi_fpv_companion.track.iou_associator import best_match


class MultiObjectTracker:
    def __init__(self, iou_threshold: float = 0.3, max_lost_frames: int = 30,
                 max_match_dist_px: float = 60.0, vel_alpha: float = 0.5,
                 reacquire_radius_px: float = 90.0,
                 confirm_hits: int = 1, confirm_window: int = 5,
                 vel_max_px_s: float = 1440.0, coast_vel_decay: float = 0.8) -> None:
        self._iou_threshold = iou_threshold
        self._max_lost_frames = max_lost_frames
        # M-of-N confirmation: a track must be DETECTED (matched, not coasted) in at
        # least `confirm_hits` of the last `confirm_window` frames before it is shown
        # or can be selected. Latched once met. 1 = confirm on first detection (off).
        self._confirm_hits = max(1, confirm_hits)
        self._confirm_window = max(self._confirm_hits, confirm_window)
        self._hist: Dict[int, Deque[bool]] = {}   # track_id -> recent match history
        self._confirmed: set = set()              # track_ids that have latched confirmed
        self._frame_w: Optional[int] = None       # frame bounds for the off-frame drop
        self._frame_h: Optional[int] = None
        # When a MANUALLY-selected track is dropped, re-bind the selection to a track
        # whose centroid is within this many px of the lost target's last position
        # (re-acquire the same physical target through id churn). No nearby track ->
        # hold (None), never swap to a different target.
        self._reacquire_radius_px = reacquire_radius_px
        # Associate a detection to a track when it OVERLAPS (IoU) OR its centroid is
        # within this many pixels OF THE TRACK'S PREDICTED position. IoU alone fails
        # for the small boxes this system sees (a person at >100 m is a few px wide,
        # so any camera rotation shifts the box more than its own width → zero IoU).
        # Matching against a constant-velocity PREDICTION (not the last position)
        # also keeps identities through a crossing: two targets that pass each other
        # in the image would otherwise swap ids under nearest-neighbour matching.
        self._max_match_dist_px = max_match_dist_px
        self._vel_alpha = vel_alpha
        # Velocity sanity (bench finding: a ghost box cross-matched onto a real object
        # reads the position jump as motion — 60 px in one 33 ms frame = ~900 px/s of
        # PHANTOM velocity — and the track then streaks across the screen while the
        # object stands still):
        #   - vel_max_px_s clamps the estimate to plausible image motion (~2 frame
        #     widths/s; a real target refreshes its velocity every frame anyway), and
        #   - coast_vel_decay shrinks the velocity each UNMATCHED frame, so coasting
        #     extrapolates a few boxes' worth and flattens to a stop instead of
        #     dead-reckoning at full speed until the off-frame drop.
        self._vel_max_px_s = vel_max_px_s
        self._coast_vel_decay = coast_vel_decay
        self._vel: Dict[int, Tuple[float, float]] = {}   # track_id -> (vx, vy) px/s
        self._last_t: Optional[float] = None
        self._tracks: Dict[int, Target] = {}
        self._selected_id: Optional[int] = None
        # True once the operator has cycled/selected a target. Makes the selection
        # sticky: auto-acquire-highest no longer overrides it (see consume()).
        self._manual: bool = False
        # Last known centroid of the selected target, for proximity re-bind on drop.
        self._selected_last_pos: Optional[Tuple[float, float]] = None
        self._next_id: int = 1
        # When True, consume() locks the highest-confidence track if nothing valid
        # is selected (acquisition / re-acquisition). The pipeline turns this OFF
        # once engaged (TRACK/DIVE) so that if the COMMITTED target is dropped the
        # aircraft holds (returns None) instead of silently swapping to a different
        # target mid-engagement.
        self.auto_acquire: bool = True

    # ---- state the pipeline / HUD read ----
    def _confirmed_ids(self) -> List[int]:
        """Current track ids that have passed M-of-N confirmation, in stable order."""
        return [i for i in sorted(self._tracks) if i in self._confirmed]

    @property
    def tracks(self) -> List[Target]:
        """CONFIRMED tracks only, in stable (track-id) order — what the HUD shows.
        Unconfirmed (spurious / not-yet-established) tracks are hidden."""
        return [self._tracks[i] for i in self._confirmed_ids()]

    @property
    def selected_id(self) -> Optional[int]:
        return self._selected_id

    def cycle(self) -> Optional[int]:
        """Advance the selection to the next CONFIRMED track (wraps). No-op if none.
        Marks the selection MANUAL so it becomes sticky (consume() won't override it)."""
        ids = self._confirmed_ids()
        if not ids:
            self._selected_id = None
            return None
        if self._selected_id not in ids:
            self._selected_id = ids[0]
        else:
            self._selected_id = ids[(ids.index(self._selected_id) + 1) % len(ids)]
        self._manual = True
        sel = self._tracks[self._selected_id]
        self._selected_last_pos = (sel.detection.x, sel.detection.y)
        return self._selected_id

    def select(self, track_id: Optional[int]) -> None:
        """Set the selection explicitly. A non-None id is treated as a MANUAL pick
        (sticky); None clears the selection and the manual flag."""
        self._selected_id = track_id
        self._manual = track_id is not None
        if track_id is not None and track_id in self._tracks:
            sel = self._tracks[track_id]
            self._selected_last_pos = (sel.detection.x, sel.detection.y)
        else:
            self._selected_last_pos = None

    def _nearest_within(self, pos: Optional[Tuple[float, float]],
                        radius: float) -> Optional[int]:
        """Confirmed track id whose centroid is nearest `pos` and within `radius` px."""
        if pos is None:
            return None
        px, py = pos
        best_id, best_d2 = None, radius * radius
        for tid in self._confirmed_ids():
            tgt = self._tracks[tid]
            d2 = (tgt.detection.x - px) ** 2 + (tgt.detection.y - py) ** 2
            if d2 <= best_d2:
                best_id, best_d2 = tid, d2
        return best_id

    # ---- per-frame update ----
    def consume(
        self, image: object, detections: List[Detection], now: float
    ) -> Optional[Target]:
        # Learn the frame bounds (for the off-frame drop) from the image when given.
        shape = getattr(image, "shape", None)
        if shape is not None and len(shape) >= 2:
            self._frame_h, self._frame_w = int(shape[0]), int(shape[1])
        self._associate(detections, now)

        confirmed = self._confirmed_ids()
        # Nothing confirmed to show/select. Only forget the manual lock once the scene
        # is fully empty (so a pick survives a gap while unconfirmed tracks churn).
        if not confirmed:
            if not self._tracks:
                self._selected_id = None
                self._manual = False
                self._selected_last_pos = None
            return None

        # Selected track still present + confirmed (incl. coasting) -> keep it.
        if self._selected_id in confirmed:
            sel = self._tracks[self._selected_id]
            self._selected_last_pos = (sel.detection.x, sel.detection.y)
            return sel

        # Selected track was dropped this frame.
        if self._manual:
            # Sticky operator pick: re-bind to the SAME physical target (nearest
            # confirmed track to its last position). None nearby -> HOLD (None).
            rebind = self._nearest_within(self._selected_last_pos, self._reacquire_radius_px)
            if rebind is not None:
                self._selected_id = rebind
                sel = self._tracks[rebind]
                self._selected_last_pos = (sel.detection.x, sel.detection.y)
                return sel
            return None

        # No manual pick yet: auto-acquire the highest-confidence CONFIRMED track
        # (initial acquisition) while auto_acquire is on (STANDBY). Off -> hold.
        if self.auto_acquire:
            self._selected_id = max(
                confirmed, key=lambda i: self._tracks[i].detection.confidence
            )
            sel = self._tracks[self._selected_id]
            self._selected_last_pos = (sel.detection.x, sel.detection.y)
            return sel
        return None

    def _drop(self, tid: int) -> None:
        """Remove a track and all its side state."""
        self._tracks.pop(tid, None)
        self._vel.pop(tid, None)
        self._hist.pop(tid, None)
        self._confirmed.discard(tid)

    def _record_hit(self, tid: int, matched: bool) -> None:
        """Update a track's match history and latch confirmation once M-of-N is met."""
        h = self._hist.get(tid)
        if h is None:
            h = self._hist[tid] = deque(maxlen=self._confirm_window)
        h.append(matched)
        if tid not in self._confirmed and sum(h) >= self._confirm_hits:
            self._confirmed.add(tid)

    def _off_frame(self, det: Detection) -> bool:
        """True if the detection centre is outside the known frame bounds."""
        if self._frame_w is None or self._frame_h is None:
            return False
        return not (0.0 <= det.x <= self._frame_w and 0.0 <= det.y <= self._frame_h)

    def _associate(self, detections: List[Detection], now: float) -> None:
        dt = (now - self._last_t) if self._last_t is not None else 0.0
        self._last_t = now
        unmatched = list(detections)
        # Existing tracks claim their best unmatched detection, matched against the
        # track's constant-velocity PREDICTION (carries identity through a crossing).
        for tid in sorted(self._tracks):
            cur = self._tracks[tid]
            vx, vy = self._vel.get(tid, (0.0, 0.0))
            pred = replace(cur.detection, x=cur.detection.x + vx * dt,
                           y=cur.detection.y + vy * dt)
            best = best_match(pred, unmatched, self._iou_threshold, self._max_match_dist_px)
            if best is not None:
                unmatched.remove(best)
                if dt > 1e-3:
                    a = self._vel_alpha
                    nvx = a * (best.x - cur.detection.x) / dt + (1 - a) * vx
                    nvy = a * (best.y - cur.detection.y) / dt + (1 - a) * vy
                    speed = (nvx * nvx + nvy * nvy) ** 0.5
                    if speed > self._vel_max_px_s:     # implausible = mostly match noise
                        scale = self._vel_max_px_s / speed
                        nvx, nvy = nvx * scale, nvy * scale
                    self._vel[tid] = (nvx, nvy)
                self._tracks[tid] = Target(detection=best, track_id=tid,
                                           lost_frames=0, timestamp=now)
                self._record_hit(tid, True)
            else:
                lost = cur.lost_frames + 1
                # Drop a coasting track that has left the frame instead of letting it
                # glide off-screen on a (possibly spurious) velocity estimate; also
                # drop once it has coasted past the staleness cap.
                if lost > self._max_lost_frames or self._off_frame(pred):
                    self._drop(tid)
                else:
                    # Coast on the prediction (so a brief miss keeps moving with it),
                    # DECAYING the velocity each unmatched frame: nothing is being
                    # measured, so there is no basis for sustained extrapolation — a
                    # phantom velocity dies out instead of streaking off-screen, and a
                    # real mover re-measures (and refreshes) within a few frames.
                    self._vel[tid] = (vx * self._coast_vel_decay,
                                      vy * self._coast_vel_decay)
                    self._tracks[tid] = Target(detection=pred, track_id=tid,
                                               lost_frames=lost, timestamp=cur.timestamp)
                    self._record_hit(tid, False)
        # Unmatched detections become new tracks (velocity seeds at zero). Off-frame
        # detections are ignored (don't spawn a track that's already gone).
        for d in unmatched:
            if self._off_frame(d):
                continue
            tid = self._next_id
            self._tracks[tid] = Target(detection=d, track_id=tid,
                                       lost_frames=0, timestamp=now)
            self._vel[tid] = (0.0, 0.0)
            self._record_hit(tid, True)
            self._next_id += 1
