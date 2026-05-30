from pi_fpv_companion.types import Detection
from pi_fpv_companion.track.multi_target import MultiObjectTracker


def _d(x, y, w=40, h=40, conf=0.9, cid=0):
    return Detection(x=x, y=y, w=w, h=h, confidence=conf, class_id=cid)


def test_tracks_all_detections_with_stable_ids():
    t = MultiObjectTracker()
    t.consume(None, [_d(100, 100), _d(400, 300), _d(600, 200)], now=0.0)
    assert len(t.tracks) == 3
    ids = [tr.track_id for tr in t.tracks]
    assert ids == sorted(ids) and len(set(ids)) == 3


def test_association_keeps_identity_as_target_moves():
    t = MultiObjectTracker()
    t.consume(None, [_d(100, 100), _d(400, 300)], 0.0)
    id_a = t.tracks[0].track_id
    # both shift a little → same ids (IoU association)
    t.consume(None, [_d(110, 105), _d(410, 305)], 0.033)
    assert t.tracks[0].track_id == id_a
    assert len(t.tracks) == 2


def test_new_detection_spawns_new_track():
    t = MultiObjectTracker()
    t.consume(None, [_d(100, 100)], 0.0)
    t.consume(None, [_d(100, 100), _d(500, 400)], 0.033)
    assert len(t.tracks) == 2


def test_lost_track_dropped_after_max_lost():
    t = MultiObjectTracker(max_lost_frames=3)
    t.consume(None, [_d(100, 100), _d(500, 400)], 0.0)
    # second target disappears; first persists
    for i in range(5):
        t.consume(None, [_d(100, 100)], (i + 1) * 0.033)
    assert len(t.tracks) == 1
    assert t.tracks[0].detection.x == 100


def test_auto_selects_highest_confidence_then_cycles():
    t = MultiObjectTracker()
    t.consume(None, [_d(100, 100, conf=0.5), _d(400, 300, conf=0.95)], 0.0)
    # highest-confidence auto-locked
    assert t._tracks[t.selected_id].detection.confidence == 0.95
    first = t.selected_id
    nxt = t.cycle()
    assert nxt != first                     # advanced to another track
    wrapped = t.cycle()
    assert wrapped == first                 # wraps back (2 tracks)


def test_selection_persists_across_frames():
    t = MultiObjectTracker()
    t.consume(None, [_d(100, 100), _d(400, 300)], 0.0)
    t.cycle()                                # pick a specific one
    chosen = t.selected_id
    chosen_x = t._tracks[chosen].detection.x
    # several frames with both targets drifting — selection stays on the same id
    for i in range(5):
        dx = (i + 1) * 5
        sel = t.consume(None, [_d(100 + (dx if chosen_x == 100 else 0), 100),
                               _d(400 + (dx if chosen_x == 400 else 0), 300)], (i + 1) * 0.033)
        assert t.selected_id == chosen
        assert sel.track_id == chosen


def test_selected_track_coasts_through_a_brief_miss_then_keeps_lock():
    t = MultiObjectTracker(max_lost_frames=10)
    t.consume(None, [_d(100, 100), _d(400, 300)], 0.0)
    t.cycle()
    chosen = t.selected_id
    chosen_x = t._tracks[chosen].detection.x
    # the chosen target misses one frame (other still present) → lock retained
    other_x = 400 if chosen_x == 100 else 100
    sel = t.consume(None, [_d(other_x, 100 if other_x == 100 else 300)], 0.033)
    assert t.selected_id == chosen
    assert sel is not None and sel.lost_frames >= 1   # coasting, not dropped


def test_dropped_selection_reacquires_highest_confidence():
    t = MultiObjectTracker(max_lost_frames=2)
    t.consume(None, [_d(100, 100, conf=0.6), _d(400, 300, conf=0.9)], 0.0)
    t.select(t.tracks[0].track_id)           # lock the low-confidence one (x=100)
    # it vanishes for good; the other remains
    for i in range(4):
        sel = t.consume(None, [_d(400, 300, conf=0.9)], (i + 1) * 0.033)
    # the dropped selection re-acquires the surviving (highest-conf) track
    assert sel is not None and sel.detection.x == 400


def test_auto_acquire_off_holds_instead_of_swapping_on_drop():
    # With auto_acquire OFF (engaged), a dropped selection does NOT jump to another
    # target — consume returns None (hold), so a committed dive never swaps targets.
    t = MultiObjectTracker(max_lost_frames=2)
    t.consume(None, [_d(100, 100), _d(400, 300)], 0.0)
    t.select(t.tracks[0].track_id)               # lock the left target
    t.auto_acquire = False                       # committed
    out = None
    for i in range(4):                           # left target vanishes for good
        out = t.consume(None, [_d(400, 300)], (i + 1) * 0.033)
    assert out is None                           # held, did NOT swap to the right target


def test_small_boxes_under_motion_keep_their_ids():
    # Distant targets are tiny boxes; under camera motion they shift more than their
    # width (zero IoU). The distance gate must keep their ids instead of spawning a
    # new track every frame (the bug the closed-loop multi-target test surfaced).
    t = MultiObjectTracker(iou_threshold=0.3)
    t.consume(None, [_d(200, 300, w=3, h=3), _d(500, 300, w=3, h=3)], 0.0)
    ids0 = [tr.track_id for tr in t.tracks]
    t.consume(None, [_d(212, 300, w=3, h=3), _d(512, 300, w=3, h=3)], 0.033)  # 12 px shift
    assert [tr.track_id for tr in t.tracks] == ids0   # same ids, no new tracks
    assert len(t.tracks) == 2


def test_no_detections_returns_none():
    t = MultiObjectTracker()
    assert t.consume(None, [], 0.0) is None
    assert t.tracks == []
