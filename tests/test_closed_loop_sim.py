"""Closed-loop TRACK / DIVE behaviour through the FULL guidance chain.

Where test_visual_servo.py checks a single `compute_intent` call, these tests
close the loop (camera projection → filter → servo → safety → airframe → repeat)
so they can answer the question single-frame tests can't:

    The camera is bolted to the airframe, so every command rotates the field of
    view. Does the closed loop KEEP THE TARGET IN FRAME, or does the guidance
    steer the FOV off the target and lose the lock?

Driven by the shipping config gains (config/imx500.yaml — see closed_loop_sim
.imx500_servo / .imx500_safety). The airframe/camera model and its assumptions
are documented in closed_loop_sim.py.
"""
from __future__ import annotations
import math

import pytest

from pi_fpv_companion.types import GuidanceMode
from tests.closed_loop_sim import (
    Airframe, CameraModel, SimWorld, imx500_servo, imx500_safety,
)

W, H = 720, 576


def _world(target_pos, target_vel=(0.0, 0.0, 0.0), alt=50.0, **servo):
    cam = CameraModel(W, H)   # IMX500 defaults: HFoV 66.3°, VFoV 52.3°
    return SimWorld(
        camera=cam, servo=imx500_servo(**servo), safety=imx500_safety(),
        airframe=Airframe(pos=(0.0, 0.0, alt)),
        target_pos=target_pos, target_vel=target_vel,
    )


# --------------------------------------------------------------------------
# Camera projection sign conventions (the fixed-camera geometry the loop relies on)
# --------------------------------------------------------------------------

def test_projection_centre_right_above_and_nosedown_rise():
    cam = CameraModel(W, H)
    ahead, _, inf = cam.project((20.0, 0.0, 0.0), psi=0.0, phi=0.0)
    assert inf and abs(ahead.x - W / 2) < 1 and abs(ahead.y - H / 2) < 1
    right, _, _ = cam.project((20.0, -5.0, 0.0), 0.0, 0.0)
    assert right.x > W / 2                                   # starboard → right of centre
    above, _, _ = cam.project((20.0, 0.0, 5.0), 0.0, 0.0)
    assert above.y < H / 2                                   # higher world → higher in frame
    # Nose-down (forward lean) depresses the boresight → a level target RISES.
    nd, _, _ = cam.project((20.0, 0.0, 0.0), 0.0, math.radians(-10))
    assert nd.y < H / 2


def test_projection_behind_camera_is_not_in_frame():
    cam = CameraModel(W, H)
    det, depth, inf = cam.project((-20.0, 0.0, 0.0), 0.0, 0.0)
    assert det is None and depth <= 0 and not inf


# --------------------------------------------------------------------------
# TRACK — the follow-and-hold mode
# --------------------------------------------------------------------------

def test_track_offaxis_target_is_centred_and_held_in_frame():
    w = _world(target_pos=(30.0, -8.0, 50.0))      # ahead, to the right, level
    tr = w.run(GuidanceMode.TRACK, duration_s=15.0)
    assert not tr.ever_left_frame                  # the whole point: never lose it
    assert tr.muted_ticks == 0                     # quality stays healthy throughout
    last = tr.ticks[-1]
    assert abs(last.px - W / 2) < 60               # yaw drove it back to centre
    assert tr.peak_excursion(W, H) < 1.0           # never pinned to an edge


def test_track_converges_to_a_stable_hold_range():
    w = _world(target_pos=(40.0, 0.0, 50.0))       # straight ahead, far
    tr = w.run(GuidanceMode.TRACK, duration_s=20.0)
    hold = w.camera.hold_range(w.servo.desired_bbox_frac)
    # Closes from 40 m and settles near the hold band (the TRACK vcenter term
    # shares the pitch budget with closure, so the equilibrium sits a bit beyond
    # the pure-closure hold range — see closed_loop_sim notes).
    assert tr.final_range < 30.0                   # it really did close
    assert hold <= tr.final_range < 3.0 * hold     # ...to a sane, bounded hold
    # Range stops collapsing — it holds station, not a fly-through.
    tail = [tk.range_m for tk in tr.ticks[-30:]]
    assert max(tail) - min(tail) < 2.0


def test_track_keeps_a_crossing_target_within_yaw_authority():
    # Target crossing laterally slowly enough that the required LOS rate stays
    # under max_yaw_rate at the hold range → yaw keeps up, target stays framed.
    w = _world(target_pos=(12.0, 0.0, 50.0), target_vel=(0.0, 2.0, 0.0))
    tr = w.run(GuidanceMode.TRACK, duration_s=12.0)
    assert not tr.ever_left_frame
    assert tr.muted_ticks == 0


def test_track_loses_a_target_crossing_faster_than_yaw_authority():
    # A near, fast crossing target's angular rate exceeds max_yaw_rate_dps → the
    # FOV cannot keep up and the target exits. Proves (a) there IS a yaw-rate /
    # crossing-speed envelope and (b) the loop detects the loss rather than
    # silently flying blind.
    w = _world(target_pos=(6.0, 0.0, 50.0), target_vel=(0.0, 25.0, 0.0))
    tr = w.run(GuidanceMode.TRACK, duration_s=6.0)
    assert tr.ever_left_frame
    assert tr.first_exit_t is not None


# --------------------------------------------------------------------------
# DIVE — the commit-and-descend mode
# --------------------------------------------------------------------------

def test_dive_descends_and_keeps_yaw_centred_on_an_in_fov_target():
    # Target within the initial FOV (15 m below at 60 m → ~14° depression < half-VFOV).
    w = _world(target_pos=(60.0, 0.0, 35.0))
    tr = w.run(GuidanceMode.DIVE, duration_s=15.0)
    assert tr.altitude_lost > 5.0                  # the gravity dive actually descended
    assert tr.muted_ticks == 0
    assert all(abs(tk.px - W / 2) < 40 for tk in tr.ticks if tk.in_frame)  # yaw stays centred


def test_dive_does_not_descend_when_target_is_off_axis():
    # Horizontal aim outside dive_center_frac → descent muted (re-aim first). The
    # target sits off to the side but still in frame.
    w = _world(target_pos=(40.0, -14.0, 50.0))
    tr = w.run(GuidanceMode.DIVE, duration_s=2.0)
    # First tick, before yaw re-centres: off-axis → no descent commanded.
    assert tr.ticks[0].thrust == pytest.approx(0.5)


def test_dive_holds_when_target_starts_outside_the_fov():
    # Target far below the frame (steep depression > half-VFOV) → no detection →
    # safety mutes → ZERO_INTENT (neutral thrust, no descent). The aircraft must
    # NOT blindly dive at something it cannot see.
    w = _world(target_pos=(40.0, 0.0, 0.0))        # 50 m below at 40 m → ~51° depression
    tr = w.run(GuidanceMode.DIVE, duration_s=3.0)
    assert all(tk.muted for tk in tr.ticks)
    assert tr.altitude_lost == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------
# Agnostic DIVE convergence: the bias keeps the target framed AND closes, for a
# target below, level, or above — the whole point of the LOS-elevation framing.
# --------------------------------------------------------------------------

def _converges(tr, max_range=5.0):
    return tr.min_range < max_range and not tr.ever_left_frame


def test_dive_converges_on_a_far_ground_target_in_frame():
    # The case the legacy centred dive could NOT do: a ground target acquirable
    # only at a shallow depression (far ahead). The agnostic vertical bias keeps
    # it framed and closes onto it instead of pancaking short.
    w = _world(target_pos=(110.0, 0.0, 0.0))       # ground target, ~24° depression at start
    tr = w.run(GuidanceMode.DIVE, duration_s=60.0)
    assert _converges(tr)
    assert tr.altitude_lost > 30.0                 # it really descended onto it


def test_dive_converges_on_a_level_target_without_diving_below_it():
    # A level/front target must be PURSUED, not dived under (the failure mode of a
    # constant top-bias). Altitude is held; it closes horizontally.
    w = _world(target_pos=(35.0, 0.0, 50.0))
    tr = w.run(GuidanceMode.DIVE, duration_s=30.0)
    assert _converges(tr)
    assert abs(tr.altitude_lost) < 5.0             # essentially level


def test_dive_converges_on_an_above_target_by_climbing():
    # Target above the aircraft → climb toward it (throttle), keep it framed,
    # close. Altitude is GAINED, not lost. Closure is intentionally gentle once
    # co-altitude (a fixed forward camera cannot lean hard at a level/above
    # target without it rising out the top), so a far above target takes longer.
    w = _world(target_pos=(70.0, 0.0, 70.0))       # 20 m above, 70 m ahead
    tr = w.run(GuidanceMode.DIVE, duration_s=90.0)
    assert _converges(tr)
    assert tr.altitude_lost < -10.0                # climbed (lost < 0 == gained)


def test_dive_ground_target_with_lateral_offset_recenters_then_converges():
    w = _world(target_pos=(110.0, -18.0, 0.0))     # ground, off to the side
    tr = w.run(GuidanceMode.DIVE, duration_s=60.0)
    assert _converges(tr)


def test_dive_blind_when_target_depression_exceeds_half_vfov():
    # Acquisition limit (NOT a guidance bug): a ground target whose start
    # depression exceeds half the vertical FOV is below the frame and is never
    # seen → the aircraft holds rather than diving blind. ~33° at 75 m / 50 m alt
    # vs ~27.5° half-VFOV for the 66° lens.
    w = _world(target_pos=(75.0, 0.0, 0.0))
    tr = w.run(GuidanceMode.DIVE, duration_s=5.0)
    assert all(tk.muted for tk in tr.ticks)
    assert tr.altitude_lost == pytest.approx(0.0, abs=1e-6)
