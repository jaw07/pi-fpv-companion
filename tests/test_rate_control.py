"""Unit tests for the GUIDED_NOGPS body-rate visual servo (guidance/rate_control.py)."""
from __future__ import annotations
import math

from pi_fpv_companion.types import Detection, FilteredTarget, GuidanceMode
from pi_fpv_companion.guidance.rate_control import (
    PID, RateConfig, RateState, compute_rate_intent)

W, H = 720, 576


def _ft(cx_n, cy_n, h=40, ts=0.0):
    return FilteredTarget(
        detection=Detection(x=cx_n * W, y=cy_n * H, w=h, h=h, confidence=0.9, class_id=0),
        track_id=1, vx_px_s=0.0, vy_px_s=0.0, quality=0.9, timestamp=ts)


def _run(target_seq, *, mode=GuidanceMode.DIVE, pitch=0.0, roll=0.0, gamma=0.0, agl=40.0,
         cfg=None, st=None, n_from=0):
    cfg = cfg or RateConfig(W, H)
    st = st or RateState()
    out = None
    for i, t in enumerate(target_seq):
        out = compute_rate_intent(t, cfg, st, now=(n_from + i) * 0.05, mode=mode,
                                  pitch_rad=pitch, roll_rad=roll, gamma_rad=gamma, agl_m=agl)
    return out, st


def test_pid_proportional_and_clamp():
    p = PID(kp=2.0, out_limit=5.0)
    assert p.update(1.0, 0.1) == 2.0
    assert p.update(100.0, 0.1) == 5.0


def test_dive_low_target_noses_down():
    # DIVE: target below the vert_goal row -> nose DOWN (negative pitch rate, aerospace sign).
    out, _ = _run([_ft(0.5, 0.75, ts=i) for i in range(8)], mode=GuidanceMode.DIVE)
    assert out.pitch_rate < 0.0
    assert out.phase == "DIVE"


def test_dive_terminal_commit_freezes_all_rates():
    # DIVE inside the impact radius (agl < impact) with an off-axis, frame-filling target:
    # normally pitch would nose-over and yaw/roll would slam to chase the frame-edge box.
    # Terminal commit -> all rates frozen (ballistic) so the airframe doesn't whip at impact.
    out, _ = _run([_ft(0.85, 0.10, h=120, ts=i) for i in range(8)], mode=GuidanceMode.DIVE, agl=5.0)
    assert abs(out.pitch_rate) < 1e-6
    assert abs(out.yaw_rate) < 1e-6
    assert abs(out.roll_rate) < 1e-6
    assert out.phase == "DIVE"


def test_track_holds_altitude_at_hover():
    # TRACK follows but does NOT descend: thrust stays at the learned hover (no commit).
    out, _ = _run([_ft(0.5, 0.55, ts=i) for i in range(8)], mode=GuidanceMode.TRACK)
    assert out.phase == "TRACK"
    assert abs(out.thrust - 0.30) < 1e-6          # hover, not descending


def test_track_holds_range_noses_down_when_target_recedes():
    # TRACK captures the engage bbox size, then noses down to CLOSE BACK when the target
    # recedes (gets smaller) — maintaining the engagement range, never committing.
    cfg, st = RateConfig(W, H), RateState()
    seq = [_ft(0.5, 0.55, h=40, ts=0)]            # engage at h=40
    seq += [_ft(0.5, 0.55, h=24, ts=i) for i in range(1, 8)]   # target receded (smaller)
    out = None
    for i, t in enumerate(seq):
        out = compute_rate_intent(t, cfg, st, now=i * 0.05, mode=GuidanceMode.TRACK,
                                  pitch_rad=0.0, roll_rad=0.0, gamma_rad=0.0, agl_m=40.0)
    assert st.engage_h == 40.0
    assert out.pitch_rate < 0.0                   # noses down to re-close the gap


def test_below_horizon_target_descends():
    # Target below frame centre, level airframe, velocity level -> pursuit drives thrust below
    # the learned hover (descend onto it). Above centre -> thrust above hover (climb).
    below, _ = _run([_ft(0.5, 0.80, ts=i) for i in range(8)], pitch=0.0, gamma=0.0)
    above, _ = _run([_ft(0.5, 0.20, ts=i) for i in range(8)], pitch=0.0, gamma=0.0)
    assert below.thrust < 0.30          # hover default
    assert above.thrust > 0.30


def test_centred_target_deadzone_zero_yaw():
    # Centred target (within the horizontal deadzone) -> ZERO yaw (no pan-shake on box noise).
    out, _ = _run([_ft(0.5, 0.45, ts=i) for i in range(8)])
    assert abs(out.yaw_rate) < 1e-6


def test_off_axis_target_yaws_toward_it():
    # Far off to the right -> yaw right (positive yaw rate) to centre it.
    out, _ = _run([_ft(0.92, 0.45, ts=i) for i in range(8)])
    assert out.yaw_rate > 0.05


def test_search_noses_down_at_hover_when_no_target_and_high():
    # No target, still high -> SEARCH: nose down to acquire a below target, hover thrust.
    out, _ = _run([None for _ in range(4)], pitch=0.0, agl=40.0)
    assert out.phase == "SEARCH"
    assert out.pitch_rate < 0.0
    assert abs(out.thrust - 0.30) < 1e-6


def test_impact_latches_stop_near_ground():
    # Had a lock, then target lost near the ground = impact -> STOP (cut throttle) and LATCH:
    # a later target does not re-engage (the persistent ground target must not be re-acquired).
    seq = [_ft(0.5, 0.5, ts=0), _ft(0.5, 0.5, ts=1)] + [None for _ in range(8)]
    out, st = _run(seq, agl=5.0)
    assert out.phase == "STOP"
    assert out.thrust < 0.05             # throttle smoothly cut to ~0
    assert st.impacted is True
    out2, _ = _run([_ft(0.5, 0.5, ts=8)], agl=5.0, st=st, n_from=8)
    assert out2.phase == "STOP"          # stays stopped despite a fresh detection


def test_low_dive_without_prior_lock_does_not_latch():
    # DIVE selected low (agl<impact) with NO target ever acquired -> must NOT false-latch STOP;
    # it searches (noses down) for a target instead. Guards against an instant ground-STOP.
    out, st = _run([None for _ in range(6)], mode=GuidanceMode.DIVE, agl=5.0)
    assert out.phase == "SEARCH"
    assert st.impacted is False


def test_roll_returns_toward_level():
    # Banked right (roll>0), target centred -> roll rate is negative (return to level).
    out, _ = _run([_ft(0.5, 0.45, ts=i) for i in range(8)], roll=0.3)
    assert out.roll_rate < 0.0
