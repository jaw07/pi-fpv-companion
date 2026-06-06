import random

from pi_fpv_companion.types import Detection
from pi_fpv_companion.track.multi_target import MultiObjectTracker


def _d(x, y, w=40, h=40, conf=0.9, cid=0):
    return Detection(x=x, y=y, w=w, h=h, confidence=conf, class_id=cid)


class _Frame:
    """Minimal stand-in for a camera frame (the tracker only reads .shape)."""
    def __init__(self, w, h):
        self.shape = (h, w, 3)


def test_default_confirmation_off_shows_tracks_immediately():
    # Back-compat: confirm_hits=1 (default) -> a detection is shown on the first frame.
    t = MultiObjectTracker()
    t.consume(None, [_d(100, 100), _d(400, 300)], 0.0)
    assert len(t.tracks) == 2


def test_one_frame_ghost_never_confirmed_so_never_shown():
    # A spurious detection that appears for a single frame must never reach the HUD
    # (tracks) nor be selectable — the first-flight "spurious detection" fix.
    t = MultiObjectTracker(confirm_hits=3, confirm_window=5)
    t.consume(None, [_d(300, 300)], 0.0)
    assert t.tracks == [] and t.selected_id is None   # 1 hit, not confirmed -> hidden
    for i in range(1, 6):
        t.consume(None, [], i * 0.033)                # never seen again
    assert t.tracks == []


def test_track_confirms_only_after_m_of_n_detections():
    t = MultiObjectTracker(confirm_hits=3, confirm_window=5)
    for i in range(2):
        t.consume(None, [_d(300, 300)], i * 0.033)
        assert t.tracks == []                         # 1-2 hits: not yet confirmed
    t.consume(None, [_d(300, 300)], 2 * 0.033)        # 3rd detection -> confirmed
    assert len(t.tracks) == 1 and t.tracks[0].detection.x == 300


def test_coasting_track_that_leaves_the_frame_is_dropped():
    # A track coasting on its velocity must be dropped when its predicted centre
    # exits the frame, not glide off-screen (the "runs off camera" fix).
    fr = _Frame(720, 576)
    t = MultiObjectTracker(confirm_hits=1, max_lost_frames=100)
    # Move in <=60px steps (within the match distance) so identity holds and a
    # rightward velocity (~500 px/s) builds on a single track.
    for i, x in enumerate((500, 550, 600, 650)):
        t.consume(fr, [_d(x, 300)], i * 0.1)
    assert len(t.tracks) == 1
    dropped = False
    tt = 0.4
    for _ in range(10):                               # withhold detections -> coast right
        t.consume(fr, [], tt)
        tt += 0.1
        if t.tracks == []:
            dropped = True
            break
    assert dropped                                    # gone well before max_lost_frames=100


def test_off_frame_detection_does_not_spawn_a_track():
    fr = _Frame(720, 576)
    t = MultiObjectTracker(confirm_hits=1)
    t.consume(fr, [_d(900, 300)], 0.0)                # centre past the right edge
    assert t.tracks == []


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


def test_manual_pick_that_vanishes_holds_not_swaps_to_highest():
    # A deliberately-selected target that vanishes for good must NOT be replaced by a
    # different (higher-confidence) target — even in STANDBY (auto_acquire on). The
    # operator's pick is sticky; consume holds (None) until they pick again or the
    # same target re-appears. (Before the stickiness fix this snapped to the x=400
    # target — the "it switches back to the original" bug.)
    t = MultiObjectTracker(max_lost_frames=2)
    t.consume(None, [_d(100, 100, conf=0.6), _d(400, 300, conf=0.9)], 0.0)
    t.select(t.tracks[0].track_id)           # manually lock the low-confidence one (x=100)
    sel = "unset"
    for i in range(5):                       # x=100 vanishes for good; only x=400 remains
        sel = t.consume(None, [_d(400, 300, conf=0.9)], (i + 1) * 0.033)
    assert sel is None                       # HELD — did not swap to the x=400 target


def test_manual_pick_rebinds_to_same_target_through_id_churn():
    # The bench failure: cycle to a weaker target whose flickery detection drops and
    # re-appears under a NEW track id. The selection must re-bind to that SAME target
    # (by proximity), never snap back to the stronger "original".
    t = MultiObjectTracker(max_lost_frames=2)
    t.consume(None, [_d(150, 300, conf=0.6), _d(560, 300, conf=0.95)], 0.0)
    t.cycle()                                            # pick a specific target...
    while t._tracks[t.selected_id].detection.x != 150:   # ...force it onto the weak one
        t.cycle()
    # the weak target drops out long enough to lose its id (only the strong one shows)
    gap_sel = "unset"
    for i in range(4):
        gap_sel = t.consume(None, [_d(560, 300, conf=0.95)], (i + 1) * 0.033)
    assert gap_sel is None                               # held through the gap, no swap
    # it re-appears near its last position under a fresh id -> selection re-binds to it
    sel = t.consume(None, [_d(152, 302, conf=0.6), _d(560, 300, conf=0.95)], 0.2)
    assert sel is not None and abs(sel.detection.x - 150) < 90
    assert sel.detection.x != 560                        # did NOT jump to the strong target


def test_auto_acquire_picks_highest_only_before_a_manual_pick():
    # Initial acquisition (no manual pick yet) still auto-locks the highest-confidence
    # target so guidance has a sane default; a manual pick then disables that override.
    t = MultiObjectTracker()
    sel = t.consume(None, [_d(100, 100, conf=0.5), _d(400, 300, conf=0.95)], 0.0)
    assert sel is not None and sel.detection.x == 400   # auto-acquired the strong one
    t.cycle()                                            # operator takes over (manual)
    assert t._manual is True


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


def test_crossing_targets_keep_their_ids_via_velocity_prediction():
    # Two targets passing each other in the image: nearest-neighbour matching would
    # swap their ids at the crossing (and the lock would follow the wrong target).
    # Constant-velocity prediction carries each identity straight through.
    t = MultiObjectTracker(iou_threshold=0.3)
    t.consume(None, [_d(200, 300, w=8, h=8), _d(400, 300, w=8, h=8)], 0.0)
    left_id = min(t.tracks, key=lambda tr: tr.detection.x).track_id   # the left one (→right)
    right_id = max(t.tracks, key=lambda tr: tr.detection.x).track_id  # the right one (→left)
    ax, bx = 200, 400
    for i in range(1, 16):
        ax += 14; bx -= 14                       # A→right, B→left, cross near x=300
        t.consume(None, [_d(ax, 300, w=8, h=8), _d(bx, 300, w=8, h=8)], i * 0.033)
    byid = {tr.track_id: tr.detection.x for tr in t.tracks}
    assert byid[left_id] > byid[right_id]        # left-origin id ended on the right → no swap


def test_crossing_ids_mostly_survive_moderate_noise_and_dropout():
    # A crossing with degraded detections is genuinely ambiguous, but velocity
    # prediction should keep ids through it the large majority of the time under
    # moderate noise (8 px) + dropout (20%). Deterministic (seeded). Guards the
    # robustness from silently regressing.
    def trial(seed):
        rng = random.Random(seed)
        t = MultiObjectTracker(iou_threshold=0.3)
        t.consume(None, [_d(200, 300, w=8, h=8), _d(400, 300, w=8, h=8)], 0.0)
        lid = min(t.tracks, key=lambda tr: tr.detection.x).track_id
        rid = max(t.tracks, key=lambda tr: tr.detection.x).track_id
        ax, bx = 200, 400
        for i in range(1, 18):
            ax += 14; bx -= 14
            dets = [_d(px + rng.gauss(0, 8), 300 + rng.gauss(0, 8), w=8, h=8)
                    for px in (ax, bx) if rng.random() >= 0.2]
            t.consume(None, dets, i * 0.033)
        byid = {tr.track_id: tr.detection.x for tr in t.tracks}
        return lid in byid and rid in byid and byid[lid] > byid[rid]
    preserved = sum(trial(s) for s in range(40))
    assert preserved >= 28           # ≥70% id-preservation through a noisy crossing


def test_no_detections_returns_none():
    t = MultiObjectTracker()
    assert t.consume(None, [], 0.0) is None
    assert t.tracks == []
