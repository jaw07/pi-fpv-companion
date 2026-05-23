"""Legacy KcfTracker shim. The real implementation moved to `cv2_tracker.py` —
KCF is now one of several `cv2_backend` options on `ClassicalCv2Tracker`, with
MOSSE the measured-best default on Pi Zero 2W.
"""
from __future__ import annotations

from pi_fpv_companion.track.cv2_tracker import ClassicalCv2Tracker


class KcfTracker(ClassicalCv2Tracker):
    """Backwards-compatible name. Equivalent to `ClassicalCv2Tracker(cv2_backend='kcf')`."""

    def __init__(self, max_lost_frames: int = 15) -> None:
        super().__init__(cv2_backend="kcf", max_lost_frames=max_lost_frames)
