"""Unit tests for the framing controller (GUIDED_NOGPS attitude/thrust visual servo)."""
from __future__ import annotations
import math

from pi_fpv_companion.types import Detection, FilteredTarget, GuidanceMode
from pi_fpv_companion.guidance.framing_control import (
    PID, FramingConfig, FramingState, compute_framing_intent, HOVER_THRUST)

W, H = 720, 576


def _ft(x, y, vx=0.0, vy=0.0, h=40, ts=0.0):
    return FilteredTarget(
        detection=Detection(x=x, y=y, w=h, h=h, confidence=0.9, class_id=0),
        track_id=1, vx_px_s=vx, vy_px_s=vy, quality=0.9, timestamp=ts)


def _run(target_seq, cfg=None, pitch_meas=0.0):
    cfg = cfg or FramingConfig(W, H)
    st = FramingState()
    out = None
    for i, t in enumerate(target_seq):
        out = compute_framing_intent(t, cfg, st, now=i * 0.05, pitch_deg_measured=pitch_meas)
    return out


def test_pid_proportional_and_clamp():
    p = PID(kp=2.0, out_limit=5.0)
    assert p.update(1.0, 0.1) == 2.0
    assert p.update(100.0, 0.1) == 5.0          # clamped


def test_pid_integral_accumulates_and_clamps():
    p = PID(kp=0.0, ki=1.0, i_limit=0.5)
    for _ in range(20):
        out = p.update(1.0, 0.1)
    assert abs(out - 0.5) < 1e-9                # integral clamped at i_limit


def test_low_target_pitches_nose_down_to_frame_it():
    # A target BELOW the vert_goal (low in frame) -> nose DOWN (camera looks down, the
    # low target rises toward the top goal — and the nose-down carries the quad forward).
    cy_low = int(0.8 * H)
    out = _run([_ft(W / 2, cy_low, ts=i) for i in range(6)])
    assert out.pitch_deg < -1.0


def test_below_horizon_target_commands_descent():
    # Target below frame centre AND nose level -> truly below the horizon -> thrust < hover
    # (descend onto it). Above centre + level -> thrust > hover (climb).
    below = _run([_ft(W / 2, int(0.85 * H), ts=i) for i in range(6)], pitch_meas=0.0)
    above = _run([_ft(W / 2, int(0.15 * H), ts=i) for i in range(6)], pitch_meas=0.0)
    assert below.thrust < HOVER_THRUST
    assert above.thrust > HOVER_THRUST


def test_pitch_fold_into_descent_uses_measured_pitch():
    # A target at frame CENTRE while the nose is pitched DOWN is truly below the horizon
    # -> descend (the measured pitch folds into the angle-to-target).
    level = _run([_ft(W / 2, H // 2, ts=i) for i in range(6)], pitch_meas=0.0)
    nosedown = _run([_ft(W / 2, H // 2, ts=i) for i in range(6)], pitch_meas=-25.0)
    assert abs(level.thrust - HOVER_THRUST) < 1e-6      # centred + level -> hover
    assert nosedown.thrust < HOVER_THRUST - 0.05        # centred + nose-down -> descend


def test_right_target_yaws_and_banks_right():
    # Far off to the right -> yaw right (toward it); the bank fades in as it nears centre.
    far_right = _run([_ft(int(0.95 * W), H // 2, ts=i) for i in range(4)])
    assert far_right.yaw_rate_dps > 1.0                 # turns toward it
    near_right = _run([_ft(int(0.55 * W), H // 2, ts=i) for i in range(4)])
    assert near_right.roll_deg > 0.5                    # banks right to slide on near centre


def test_centred_level_target_is_hover_and_neutral():
    out = _run([_ft(W / 2, H // 2, ts=i) for i in range(6)], pitch_meas=0.0)
    assert abs(out.thrust - HOVER_THRUST) < 1e-6
    assert abs(out.yaw_rate_dps) < 1e-6
    assert out.vertical_rate_mps is None                # descent is the thrust, not a rate


def test_framing_controller_dives_onto_a_steep_target_from_altitude():
    # Closed loop through the sim airframe/camera (pitch->forward, thrust->vertical): from
    # altitude over a target below (a steep ~45deg engagement), the framing pitch + descent
    # thrust fly the quad down onto it. (Drone starts pointed at the target, as after TRACK.)
    from pi_fpv_companion.types import Target
    from pi_fpv_companion.track.target_filter import AlphaBetaTargetFilter
    from tests.closed_loop_sim import Airframe, CameraModel
    for alt, horiz in ((20, 22), (30, 33), (40, 44)):
        cam = CameraModel(W, H)
        af = Airframe(pos=(0.0, 0.0, float(alt)), phi=-math.atan2(alt, horiz))
        flt = AlphaBetaTargetFilter(); cfg = FramingConfig(W, H); st = FramingState()
        tpos = [float(horiz), 0.0, 0.0]; minr = 9e9; dt = 1 / 30.0
        for i in range(int(40 / dt)):
            t = i * dt
            d = (tpos[0] - af.pos[0], tpos[1] - af.pos[1], tpos[2] - af.pos[2])
            minr = min(minr, math.hypot(math.hypot(d[0], d[1]), d[2]))
            det, depth, inf = cam.project(d, af.psi, af.phi, af.roll)
            raw = Target(detection=det, track_id=1, lost_frames=0, timestamp=t) if (det and inf) else None
            ft = flt.update(raw, W, H, t)
            if ft is None:
                break
            af.step(compute_framing_intent(ft, cfg, st, t, pitch_deg_measured=math.degrees(af.phi)), dt)
        assert minr < 4.0, f"alt={alt}: min_range={minr:.1f} (framing dive did not reach)"
