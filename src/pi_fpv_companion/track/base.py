"""Tracker abstraction.

The two implementations differ in cadence:

  KcfTracker      — runs every frame via cv2.legacy.TrackerKCF. Re-initialized
                    by Pipeline on the periodic frames where the detector
                    produces fresh boxes (to refresh scale — KCF can't on its own).

  IouAssociator   — IoU-based association across whatever detections are
                    present each frame. Designed for the IMX500 path where the
                    sensor emits dense per-frame detections.

Both expose the same `consume()` entry point. The tracker decides internally
whether fresh detections mean "lock on" / "re-seed" / "associate" / "carry forward."
"""
from __future__ import annotations
from typing import List, Optional, Protocol

from pi_fpv_companion.types import Detection, Target


class Tracker(Protocol):
    def consume(
        self, image: object, detections: List[Detection], now: float
    ) -> Optional[Target]:
        """Process one frame.

        `detections` is whatever this frame produced — may be empty if the
        camera doesn't do inference and the detector wasn't scheduled this tick,
        or populated by the camera (IMX500) or by Pipeline's periodic detector
        call (PiCam path).

        Returns the current target if locked, None if no lock yet or just lost.
        """
        ...

    def is_locked(self) -> bool: ...

    def reset(self) -> None: ...
