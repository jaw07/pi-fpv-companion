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

from pi_fpv_companion.types import Detection, GuidanceMode
from tests.closed_loop_sim import (
    Airframe, CameraModel, SimWorld, imx500_servo, imx500_safety,
)

W, H = 720, 576

# SimWorld-level knobs (vs servo gains) so _world can route kwargs correctly.
_WORLD_KW = {"target_vel", "target_accel", "detection_noise_px",
             "detection_dropout_prob", "detect_latency_frames", "seed", "glitch"}


def _world(target_pos, alt=50.0, **kw):
    cam = CameraModel(W, H)   # IMX500 defaults: HFoV 66.3°, VFoV 52.3°
    world_kw = {k: kw.pop(k) for k in list(kw) if k in _WORLD_KW}
    return SimWorld(
        camera=cam, servo=imx500_servo(**kw), safety=imx500_safety(),
        airframe=Airframe(pos=(0.0, 0.0, alt)),
        target_pos=target_pos, **world_kw,
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
    w = _world(target_pos=(25.0, 0.0, 50.0))       # straight ahead
    tr = w.run(GuidanceMode.TRACK, duration_s=40.0)
    hold = w.camera.hold_range(w.servo.desired_bbox_frac)
    # Closes from 40 m and settles at the closure-regulated hold range.
    assert tr.final_range < 30.0                   # it really did close
    assert hold * 0.7 <= tr.final_range < 2.0 * hold   # ...to the hold band
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


def test_dive_does_not_command_vertical_when_target_is_off_axis():
    # Horizontal aim outside dive_center_frac → vertical commit gated off (re-aim
    # yaw first). The target sits off to the side but still in frame.
    w = _world(target_pos=(40.0, -16.0, 0.0), alt=35.0)
    tr = w.run(GuidanceMode.DIVE, duration_s=2.0)
    # First tick, before yaw re-centres: off-axis → no vertical rate commanded.
    assert tr.ticks[0].vrate_cmd == pytest.approx(0.0, abs=1e-6)


def test_dive_holds_when_target_starts_outside_the_fov():
    # Target far below the frame (steep depression > half-VFOV) → no detection →
    # safety mutes → ZERO_INTENT (neutral thrust, no descent). The aircraft must
    # NOT blindly dive at something it cannot see.
    w = _world(target_pos=(40.0, 0.0, 0.0))        # 50 m below at 40 m → ~51° depression
    tr = w.run(GuidanceMode.DIVE, duration_s=3.0)
    assert all(tk.muted for tk in tr.ticks)
    assert tr.altitude_lost == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------
# Closed-loop DIVE convergence (constant-bearing homing): the commanded vertical
# RATE holds the target's framing, so the flight path follows the LOS — closing
# onto a target below, level, OR above. A terminal frame-exit inside the impact
# radius is the target passing the camera, not a tracking loss.
# --------------------------------------------------------------------------

def _converges(tr, max_range=5.0):
    return tr.min_range < max_range and not tr.lost_before_impact(max_range)


def test_dive_converges_on_a_ground_target_in_frame():
    # The case the open-loop centred dive could NOT do: a ground target acquirable
    # only at a shallow depression (far ahead). The closed-loop descent follows the
    # LOS and closes onto it instead of pancaking short. Engagement alt 35 m.
    w = _world(target_pos=(75.0, 0.0, 0.0), alt=35.0)
    tr = w.run(GuidanceMode.DIVE, duration_s=90.0)
    assert _converges(tr)
    assert tr.altitude_lost > 25.0                 # it really descended onto it


def test_dive_converges_on_a_level_target_without_diving_below_it():
    # A level/front target must be PURSUED, not dived under. Altitude is held; it
    # closes horizontally.
    w = _world(target_pos=(35.0, 0.0, 35.0), alt=35.0)
    tr = w.run(GuidanceMode.DIVE, duration_s=90.0)
    assert _converges(tr)
    assert abs(tr.altitude_lost) < 5.0             # essentially level


def test_dive_converges_on_an_above_target_by_climbing():
    # Closed-loop homing closes on an ABOVE target too: the framing loop commands a
    # climb so the flight path follows the LOS up to it. Altitude is GAINED.
    w = _world(target_pos=(55.0, 0.0, 50.0), alt=35.0)   # 15 m above, 55 m ahead
    tr = w.run(GuidanceMode.DIVE, duration_s=110.0)
    assert _converges(tr)
    assert tr.altitude_lost < -8.0                 # climbed to it (lost < 0 == gained)


def test_dive_ground_target_with_lateral_offset_recenters_then_converges():
    w = _world(target_pos=(85.0, -15.0, 0.0), alt=35.0)  # ground, off to the side
    tr = w.run(GuidanceMode.DIVE, duration_s=90.0)
    assert _converges(tr)


def test_track_then_dive_handoff_follows_then_commits():
    # The real operational flow: FOLLOW the target in TRACK (held framed, no loss),
    # then the operator commits to DIVE and it closes. Exercises filter/tracker
    # continuity across the mode switch.
    w = _world(target_pos=(90.0, -10.0, 0.0), alt=35.0)   # ground target, off to the side
    tr = w.run(GuidanceMode.DIVE, duration_s=100.0, dive_after_s=4.0)
    # During the TRACK phase the target is followed and framed (no early loss)...
    track_phase = [tk for tk in tr.ticks if tk.t < 4.0]
    assert all(tk.in_frame for tk in track_phase)
    assert not any(tk.muted for tk in track_phase)
    # ...then DIVE closes onto it.
    assert _converges(tr)
    assert tr.altitude_lost > 25.0                  # descended onto the ground target


# --------------------------------------------------------------------------
# Perception realism: the closed loop must ride out noisy / dropped / stale
# detections (the filter smooths + coasts) and MUTE on a misdetection (the
# filter's innovation + class gating, the project's dominant-hazard defence).
# --------------------------------------------------------------------------

def test_dive_rides_out_detection_noise():
    w = _world(target_pos=(110.0, 0.0, 0.0), detection_noise_px=12.0)
    tr = w.run(GuidanceMode.DIVE, duration_s=120.0)
    assert _converges(tr)                          # alpha-beta filter smooths the jitter


def test_dive_rides_out_detection_dropout():
    w = _world(target_pos=(110.0, 0.0, 0.0), detection_dropout_prob=0.3)
    tr = w.run(GuidanceMode.DIVE, duration_s=120.0)
    assert _converges(tr)                          # coasts on the motion model through gaps


def test_dive_rides_out_detector_latency():
    w = _world(target_pos=(110.0, 0.0, 0.0), detect_latency_frames=5)
    tr = w.run(GuidanceMode.DIVE, duration_s=120.0)
    assert _converges(tr)


def test_track_holds_a_maneuvering_target():
    # A laterally-accelerating crosser — the feed-forward + closed loop keep it
    # framed (TRACK's job is FOV retention, which the filter velocity estimate aids).
    w = _world(target_pos=(20.0, 0.0, 50.0), target_vel=(0.0, 6.0, 0.0),
               target_accel=(0.0, 1.5, 0.0))
    tr = w.run(GuidanceMode.TRACK, duration_s=12.0)
    assert not tr.ever_left_frame
    assert tr.muted_ticks == 0


def test_misdetection_teleport_is_gated_and_mutes_then_recovers():
    # A misdetection (centroid teleports to a frame corner for ~0.5 s) must NOT be
    # acted on: the filter's innovation gate rejects it, quality collapses, and the
    # safety gate MUTES (the aircraft holds, doesn't chase the corner). Good
    # detections resume → quality recovers → the engagement still converges.
    def teleport(i, det):
        if det is None:
            return None
        if 120 <= i < 140:                         # ~t 4.0–4.7 s, mid-dive
            return Detection(x=20.0, y=540.0, w=det.w, h=det.h, confidence=0.9, class_id=0)
        return det
    w = _world(target_pos=(110.0, 0.0, 0.0), glitch=teleport)
    tr = w.run(GuidanceMode.DIVE, duration_s=120.0)
    glitch_ticks = [tk for tk in tr.ticks if 120 <= round(tk.t * 30) - 1 < 140]
    assert any(tk.muted for tk in glitch_ticks)    # held during the misdetection
    assert _converges(tr)                          # and recovered onto the real target


def test_occluded_target_mutes_then_reacquires_and_converges():
    # Target occluded (detector returns nothing — e.g. behind cover) for ~2 s
    # mid-dive: the filter coasts, quality decays, and the safety gate MUTES (the
    # aircraft holds, does not fly blind). When it reappears the filter re-acquires
    # and the dive resumes onto it.
    def occlude(i, det):
        return None if 90 <= i < 150 else det          # ~t 3.0–5.0 s
    w = _world(target_pos=(120.0, 0.0, 0.0), glitch=occlude)
    tr = w.run(GuidanceMode.DIVE, duration_s=130.0)
    occ = [tk for tk in tr.ticks if 90 <= round(tk.t * 30) - 1 < 150]
    assert sum(tk.muted for tk in occ) >= 0.7 * len(occ)   # held for most of the gap
    post = [tk for tk in tr.ticks if round(tk.t * 30) - 1 >= 165]
    assert any(not tk.muted for tk in post)            # re-acquired after reappearance
    assert _converges(tr)                              # and closed onto it


def test_monte_carlo_hit_rate_over_noisy_engagements():
    # Headline robustness: randomized engagement altitude, acquirable depression,
    # lateral offset, detection noise and dropout → the closed loop should hit a
    # large majority. Seeded for determinism; small N to stay fast (the full sweep
    # is scripts/sim_track_dive.py). Guards against a regression in the loop.
    import math
    import random
    m = random.Random(0)
    hits = engaged = 0
    for k in range(30):
        alt = m.uniform(30.0, 50.0)
        dep = m.uniform(14.0, 25.0)
        hr = alt / math.tan(math.radians(dep))
        off = m.uniform(-0.25, 0.25) * hr
        tr = _world((hr, off, 0.0), alt=alt, detection_noise_px=m.uniform(0.0, 10.0),
                    detection_dropout_prob=m.uniform(0.0, 0.3), seed=k).run(
            GuidanceMode.DIVE, duration_s=130.0)
        if not any(tk.in_frame for tk in tr.ticks):
            continue                                # blind (not acquirable) — not counted
        engaged += 1
        if tr.min_range < 5.0 and not tr.lost_before_impact(5.0):
            hits += 1
    assert engaged >= 25
    assert hits / engaged >= 0.8                    # ≥80% of acquirable engagements hit


def test_class_flip_is_gated_and_mutes():
    # The tracker hands over a different class mid-engagement (re-locked the wrong
    # object) → class-consistency gating collapses quality → safety mutes.
    def flip(i, det):
        if det is None:
            return None
        if 120 <= i < 160:
            return Detection(x=det.x, y=det.y, w=det.w, h=det.h, confidence=0.9, class_id=7)
        return det
    w = _world(target_pos=(110.0, 0.0, 0.0), glitch=flip)
    tr = w.run(GuidanceMode.DIVE, duration_s=120.0)
    assert any(tk.muted for tk in tr.ticks)


def test_dive_blind_when_target_depression_exceeds_half_vfov():
    # Acquisition limit (NOT a guidance bug): a ground target whose start
    # depression exceeds half the vertical FOV is below the frame and is never
    # seen → the aircraft holds rather than diving blind. ~35° at 50 m / 35 m alt
    # vs ~26.1° half-VFOV for the IMX500's 52.3° vertical FoV.
    w = _world(target_pos=(50.0, 0.0, 0.0), alt=35.0)
    tr = w.run(GuidanceMode.DIVE, duration_s=5.0)
    assert all(tk.muted for tk in tr.ticks)
    assert tr.altitude_lost == pytest.approx(0.0, abs=1e-6)
