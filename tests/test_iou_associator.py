from pi_fpv_companion.track.iou_associator import IouAssociator
from pi_fpv_companion.types import Detection


def _det(x, y, w=60, h=60, conf=0.9):
    return Detection(x=x, y=y, w=w, h=h, confidence=conf, class_id=0, class_name="t")


def test_starts_unlocked():
    assoc = IouAssociator()
    assert not assoc.is_locked()
    assert assoc.consume(None, [], 0.0) is None


def test_acquires_lock_on_first_detections():
    assoc = IouAssociator()
    t = assoc.consume(None, [_det(100, 100, conf=0.9), _det(500, 500, conf=0.6)], now=0.0)
    assert t is not None
    assert assoc.is_locked()
    # Highest confidence wins
    assert t.detection.x == 100


def test_associates_to_overlapping_detection():
    assoc = IouAssociator(iou_threshold=0.3)
    assoc.consume(None, [_det(100, 100)], now=0.0)
    t = assoc.consume(None, [_det(105, 102)], now=0.05)
    assert t is not None
    assert t.detection.x == 105
    assert t.lost_frames == 0


def test_picks_highest_iou_when_multiple_present():
    assoc = IouAssociator(iou_threshold=0.3)
    assoc.consume(None, [_det(100, 100)], now=0.0)
    t = assoc.consume(None, [_det(500, 500), _det(101, 101)], now=0.05)
    assert t.detection.x == 101


def test_no_overlap_increments_lost_count():
    assoc = IouAssociator(iou_threshold=0.3, max_lost_frames=3)
    assoc.consume(None, [_det(100, 100)], now=0.0)
    t = assoc.consume(None, [_det(500, 500)], now=0.05)
    assert t is not None
    assert t.lost_frames == 1


def test_drops_target_after_max_lost_frames():
    assoc = IouAssociator(iou_threshold=0.3, max_lost_frames=3)
    assoc.consume(None, [_det(100, 100)], now=0.0)
    for i in range(4):
        assoc.consume(None, [], now=0.05 * (i + 1))
    assert not assoc.is_locked()


def test_empty_detections_after_lock_increments_lost():
    assoc = IouAssociator(iou_threshold=0.3, max_lost_frames=10)
    assoc.consume(None, [_det(100, 100)], now=0.0)
    t = assoc.consume(None, [], now=0.05)
    assert t is not None
    assert t.lost_frames == 1


def test_reset_drops_lock():
    assoc = IouAssociator()
    assoc.consume(None, [_det(100, 100)], now=0.0)
    assoc.reset()
    assert not assoc.is_locked()
