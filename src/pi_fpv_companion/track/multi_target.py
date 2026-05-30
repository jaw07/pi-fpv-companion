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
from typing import Dict, List, Optional

from pi_fpv_companion.types import Detection, Target
from pi_fpv_companion.track.iou_associator import _iou


class MultiObjectTracker:
    def __init__(self, iou_threshold: float = 0.3, max_lost_frames: int = 30) -> None:
        self._iou_threshold = iou_threshold
        self._max_lost_frames = max_lost_frames
        self._tracks: Dict[int, Target] = {}
        self._selected_id: Optional[int] = None
        self._next_id: int = 1

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
        # Default / re-acquire: if nothing is selected (first lock, or the selected
        # track was dropped), lock the highest-confidence current track.
        if self._selected_id not in self._tracks and self._tracks:
            self._selected_id = max(
                self._tracks, key=lambda i: self._tracks[i].detection.confidence
            )
        return self._tracks.get(self._selected_id) if self._selected_id is not None else None

    def _associate(self, detections: List[Detection], now: float) -> None:
        unmatched = list(detections)
        # Existing tracks claim their best unmatched detection (highest id first is
        # arbitrary but stable). A matched track resets to lost_frames=0.
        for tid in sorted(self._tracks):
            cur = self._tracks[tid]
            best, best_iou = None, self._iou_threshold
            for d in unmatched:
                iou = _iou(cur.detection, d)
                if iou >= best_iou:
                    best, best_iou = d, iou
            if best is not None:
                unmatched.remove(best)
                self._tracks[tid] = Target(detection=best, track_id=tid,
                                           lost_frames=0, timestamp=now)
            else:
                lost = cur.lost_frames + 1
                if lost > self._max_lost_frames:
                    del self._tracks[tid]
                else:
                    # Coast: keep the last box, advance the lost counter.
                    self._tracks[tid] = Target(detection=cur.detection, track_id=tid,
                                               lost_frames=lost, timestamp=cur.timestamp)
        # Unmatched detections become new tracks.
        for d in unmatched:
            self._tracks[self._next_id] = Target(detection=d, track_id=self._next_id,
                                                 lost_frames=0, timestamp=now)
            self._next_id += 1
