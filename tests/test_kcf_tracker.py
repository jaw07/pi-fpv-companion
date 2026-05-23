"""KCF tracker smoke tests via the consume() API."""
import numpy as np

from pi_fpv_companion.track.kcf_tracker import KcfTracker
from pi_fpv_companion.types import Detection


def _frame_with_box(cx: int, cy: int, size: int = 60, shape=(480, 640)) -> np.ndarray:
    img = np.full((shape[0], shape[1], 3), 64, dtype=np.uint8)
    half = size // 2
    img[max(0, cy - half):cy + half, max(0, cx - half):cx + half] = (0, 0, 255)
    return img


def _det(cx, cy, size=60, conf=0.9):
    return Detection(x=cx, y=cy, w=size, h=size, confidence=conf, class_id=0, class_name="t")


def test_starts_unlocked():
    t = KcfTracker()
    assert not t.is_locked()


def test_consume_without_detections_or_lock_returns_none():
    t = KcfTracker()
    assert t.consume(_frame_with_box(200, 200), detections=[], now=0.0) is None


def test_consume_with_detection_acquires_lock():
    t = KcfTracker()
    img = _frame_with_box(200, 200)
    result = t.consume(img, detections=[_det(200, 200)], now=0.0)
    assert result is not None
    assert t.is_locked()


def test_consume_without_detections_after_lock_runs_cv_update():
    t = KcfTracker()
    img0 = _frame_with_box(200, 200)
    t.consume(img0, detections=[_det(200, 200)], now=0.0)
    img1 = _frame_with_box(210, 200)
    out = t.consume(img1, detections=[], now=0.05)
    assert out is not None
    assert out.lost_frames == 0


def test_reseed_picks_detection_closest_to_current_target():
    """KCF's whole point: when the detector returns boxes, pick the one nearest
    the current tracked position (not the highest-confidence one). This refreshes
    scale without losing identity to a different object."""
    t = KcfTracker()
    img = _frame_with_box(200, 200)
    t.consume(img, detections=[_det(200, 200)], now=0.0)
    # Two new detections: one far (high conf), one near (lower conf)
    nearest = _det(205, 198, conf=0.6)
    far = _det(500, 500, conf=0.95)
    out = t.consume(_frame_with_box(205, 198), detections=[nearest, far], now=0.05)
    assert out is not None
    # Should track the near one, not the high-confidence far one
    assert abs(out.detection.x - 205) < 5


def test_tiny_seed_is_rejected_safely():
    """KCF crashes on zero-size bboxes — verify we guard."""
    t = KcfTracker()
    img = _frame_with_box(200, 200)
    out = t.consume(img, detections=[_det(200, 200, size=2)], now=0.0)
    # No crash, no lock (we refused to init from the tiny box)
    assert out is None
    assert not t.is_locked()


def test_reset_drops_lock():
    t = KcfTracker()
    img = _frame_with_box(200, 200)
    t.consume(img, detections=[_det(200, 200)], now=0.0)
    t.reset()
    assert not t.is_locked()
