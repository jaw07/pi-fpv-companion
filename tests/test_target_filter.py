from pi_fpv_companion.track.target_filter import AlphaBetaTargetFilter, FilterConfig
from pi_fpv_companion.types import Detection, Target


W, H = 720, 576


def _tgt(x, y, tid=1, conf=0.9, cls=0, w=40, h=40):
    return Target(
        detection=Detection(x=x, y=y, w=w, h=h, confidence=conf, class_id=cls),
        track_id=tid, lost_frames=0, timestamp=0.0,
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
