from pi_fpv_companion.track.target_filter import AlphaBetaTargetFilter, FilterConfig
from pi_fpv_companion.types import Detection, Target


W, H = 720, 576


def _tgt(x, y, tid=1, conf=0.9, cls=0, w=40, h=40, lost=0):
    return Target(
        detection=Detection(x=x, y=y, w=w, h=h, confidence=conf, class_id=cls),
        track_id=tid, lost_frames=lost, timestamp=0.0,
    )


def test_none_before_first_measurement():
    f = AlphaBetaTargetFilter()
    assert f.update(None, W, H, 0.0) is None
    assert not f.is_active()


def test_seeds_on_first_measurement():
    f = AlphaBetaTargetFilter()
    ft = f.update(_tgt(360, 288, conf=0.8), W, H, 0.0)
    assert ft is not None
    assert ft.detection.x == 360
    assert ft.quality == 0.8
    assert ft.vx_px_s == 0.0


def test_estimates_velocity_on_steady_motion():
    f = AlphaBetaTargetFilter()
    f.update(_tgt(300, 288), W, H, 0.0)
    for i in range(1, 12):
        ft = f.update(_tgt(300 + i * 10, 288), W, H, i * 0.1)  # +100 px/s
    # alpha-beta should converge the velocity estimate toward ~100 px/s
    assert ft.vx_px_s > 50.0
    assert ft.quality > 0.5


def test_rejects_implausible_jump_and_decays_quality():
    f = AlphaBetaTargetFilter()
    f.update(_tgt(360, 288, conf=0.9), W, H, 0.0)
    q0 = f.update(_tgt(365, 288, conf=0.9), W, H, 0.05).quality
    # Teleport to the far corner — physically impossible in one tick.
    ft = f.update(_tgt(20, 560, conf=0.95), W, H, 0.10)
    assert ft is not None
    assert ft.quality < q0                       # penalized, not trusted
    assert ft.detection.x > 200                  # did NOT snap to the teleport


def test_class_flip_degrades_quality_not_position():
    f = AlphaBetaTargetFilter()
    f.update(_tgt(360, 288, cls=0, conf=0.9), W, H, 0.0)
    q0 = f.update(_tgt(362, 288, cls=0, conf=0.9), W, H, 0.05).quality
    # Same place, but the detector now calls it a different class.
    ft = f.update(_tgt(363, 289, cls=14, conf=0.95), W, H, 0.10)
    assert ft.quality < q0


def test_quality_decays_while_coasting_and_eventually_drops():
    f = AlphaBetaTargetFilter()
    f.update(_tgt(360, 288, conf=0.9), W, H, 0.0)
    last = None
    for i in range(1, 40):
        last = f.update(None, W, H, i * 0.1)
        if last is None:
            break
    assert last is None                          # fully dropped after long coast
    assert not f.is_active()


def test_reacquire_on_new_track_id_reseeds():
    f = AlphaBetaTargetFilter()
    f.update(_tgt(100, 100, tid=1, conf=0.9), W, H, 0.0)
    f.update(_tgt(110, 100, tid=1, conf=0.9), W, H, 0.05)
    ft = f.update(_tgt(500, 400, tid=2, conf=0.7), W, H, 0.10)  # new lock
    assert ft.track_id == 2
    assert ft.detection.x == 500                 # reseeded, not gated as a jump
    assert ft.vx_px_s == 0.0
    assert ft.quality == 0.7


def test_persistent_good_track_keeps_high_quality():
    f = AlphaBetaTargetFilter()
    f.update(_tgt(360, 288, conf=0.9), W, H, 0.0)
    ft = None
    for i in range(1, 20):
        ft = f.update(_tgt(360 + i, 288, conf=0.9), W, H, i * 0.05)
    assert ft.quality > 0.8


def test_accepted_measurement_advances_measurement_timestamp():
    f = AlphaBetaTargetFilter()
    f.update(_tgt(360, 288, conf=0.9), W, H, 0.0)
    ft = f.update(_tgt(362, 288, conf=0.9), W, H, 0.05)  # fresh, plausible
    assert ft.measurement_timestamp == 0.05
    assert ft.timestamp == 0.05


def test_coasting_box_does_not_advance_measurement_timestamp():
    # The tracker coasts on a frozen box (lost_frames > 0): `timestamp` keeps
    # advancing (estimate still emitted) but `measurement_timestamp` must freeze
    # at the last real detection, so the safety watchdog can age it out.
    f = AlphaBetaTargetFilter()
    f.update(_tgt(360, 288, conf=0.9), W, H, 0.0)
    last_real = f.update(_tgt(362, 288, conf=0.9), W, H, 0.05)
    assert last_real.measurement_timestamp == 0.05
    ft = None
    for i in range(2, 6):                                # frozen box, lost_frames>0
        ft = f.update(_tgt(362, 288, conf=0.9, lost=i - 1), W, H, i * 0.05)
    assert ft is not None
    assert ft.measurement_timestamp == 0.05              # frozen at last real meas
    assert ft.timestamp == 0.25                          # but the estimate is current
    assert (ft.timestamp - ft.measurement_timestamp) > 0.15  # watchdog can see the age


def test_coasting_box_does_not_recover_quality():
    # A high-confidence frozen box must NOT pull quality back up — it's unconfirmed.
    f = AlphaBetaTargetFilter()
    f.update(_tgt(360, 288, conf=0.9), W, H, 0.0)
    q_locked = f.update(_tgt(360, 288, conf=0.9), W, H, 0.05).quality
    ft = f.update(_tgt(360, 288, conf=0.9, lost=1), W, H, 0.10)
    assert ft.quality < q_locked                         # decayed, not recovered


def test_quality_survives_low_detector_rate_into_fast_pipeline():
    # The core fix: a detector firing at ~5.5 Hz into a 30 fps pipeline coasts ~5
    # frames between real detections. Quality must stay HIGH across those gaps (the
    # track is healthy), where the old per-frame decay would have cratered it.
    f = AlphaBetaTargetFilter()
    t = 0.0
    f.update(_tgt(360, 288, conf=0.85), W, H, t)
    qmin = 1.0
    for _ in range(20):                       # 20 detection cycles
        for k in range(5):                    # 5 coast frames (no fresh detection)
            t += 1 / 30
            ft = f.update(None, W, H, t)
            qmin = min(qmin, ft.quality)
        t += 1 / 30                           # the fresh detection (object barely moved)
        ft = f.update(_tgt(362, 289, conf=0.85), W, H, t)
    assert ft.quality > 0.75, f"healthy track at 5.5Hz must stay high, got {ft.quality:.2f}"
    assert qmin > 0.6, f"between-detection dip must stay well above the gate, got {qmin:.2f}"


def test_quality_still_drops_on_genuine_loss():
    # Time-based staleness must still age out a TRULY lost track (no detection for
    # seconds), so the safety gate mutes it.
    f = AlphaBetaTargetFilter()
    f.update(_tgt(360, 288, conf=0.9), W, H, 0.0)
    last = None
    for i in range(1, 200):
        last = f.update(None, W, H, i * 0.05)
        if last is None:
            break
    assert last is None, "a track with no detection for seconds must drop"
