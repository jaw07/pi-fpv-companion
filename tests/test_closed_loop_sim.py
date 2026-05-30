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


def test_track_holds_the_engage_distance_not_a_fixed_standoff():
    # TRACK captures the distance at engagement and holds it; it must NOT fly in to
    # the nominal desired_bbox_frac. Locked on a stationary target 25 m ahead (well
    # beyond the ~11.5 m nominal), it should stay ~25 m — maintain, not close.
    w = _world(target_pos=(25.0, 0.0, 50.0))       # straight ahead, stationary
    tr = w.run(GuidanceMode.TRACK, duration_s=40.0)
    nominal = w.camera.hold_range(w.servo.desired_bbox_frac)   # ~11.5 m
    assert abs(tr.final_range - 25.0) < 3.0        # held the engage distance...
    assert tr.final_range > nominal + 5.0          # ...clearly did NOT fly in to the nominal
    # Holds station — no drift.
    tail = [tk.range_m for tk in tr.ticks[-30:]]
    assert max(tail) - min(tail) < 2.0


def test_track_pi_holds_the_engage_distance_on_a_receding_target():
    # The "maintain distance" requirement: engaged at 20 m on a target that then
    # moves away at 1 m/s, TRACK keeps the ~20 m gap (matches its motion) rather
    # than lagging farther and farther back. The PI integral holds it exactly;
    # pure-P (closure_i_gain=0) settles meaningfully farther back (a residual size
    # error is needed to sustain the chase lean). 20 m is the ENGAGE distance, not
    # the nominal desired_bbox_frac — proving it holds the gap, not a fixed standoff.
    start = 20.0
    pi = _world(target_pos=(start, 0.0, 50.0), target_vel=(1.0, 0.0, 0.0)) \
        .run(GuidanceMode.TRACK, duration_s=70.0)
    pure_p = _world(target_pos=(start, 0.0, 50.0), target_vel=(1.0, 0.0, 0.0),
                    closure_i_gain=0.0).run(GuidanceMode.TRACK, duration_s=70.0)
    assert abs(pi.final_range - start) < 2.0          # PI holds the 20 m engage gap
    assert pure_p.final_range > pi.final_range + 2.0  # pure-P lags farther back
    assert not pi.lost_before_impact(2.0)             # stays framed throughout
    tail = [tk.range_m for tk in pi.ticks[-60:]]
    assert max(tail) - min(tail) < 2.0                # settled, no limit cycle


def test_track_does_not_nod_on_a_below_ground_target():
    # A far-below (ground) target must NOT send TRACK's pitch into a limit cycle. The
    # vertical re-centring is a gentle nudge, not an attempt to fully centre a below
    # target (that fights range-hold closure → a sustained nose nod). At the old gain
    # 0.10 this swung the pitch ~26° with ~17 reversals and lost the target.
    w = _world(target_pos=(50.0, 0.0, 0.0), alt=15.0)
    tr = w.run(GuidanceMode.TRACK, duration_s=25.0)
    assert not tr.ever_left_frame                       # stays framed (didn't oscillate out)
    pc = [tk.pitch_cmd for tk in tr.ticks if tk.in_frame]
    d = [pc[i] - pc[i - 1] for i in range(1, len(pc))]
    reversals = sum(1 for i in range(1, len(d)) if d[i] * d[i - 1] < 0 and abs(d[i]) > 0.05)
    assert reversals < 6                                # no sustained nod
    assert max(pc) - min(pc) < 18.0                     # bounded pitch travel


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


def _two_target_dive(n_cycles):
    """Closed loop with TWO acquirable ground targets (A right / y<0, B left / y>0;
    B higher-confidence so it auto-locks). Cycle `n_cycles` times in STANDBY, then
    DIVE. Returns (engaged 'A'|'B', min_range_A, min_range_B)."""
    from dataclasses import replace
    from tests.closed_loop_sim import _norm
    from pi_fpv_companion.track.multi_target import MultiObjectTracker
    from pi_fpv_companion.track.target_filter import AlphaBetaTargetFilter
    from pi_fpv_companion.guidance.visual_servo import compute_intent
    from pi_fpv_companion.guidance.safety import gate
    from pi_fpv_companion.types import SwitchState, ZERO_INTENT

    cam = CameraModel(W, H)
    af = Airframe(pos=(0.0, 0.0, 50.0))
    servo, safety = imx500_servo(), imx500_safety()
    flt = AlphaBetaTargetFilter()
    trk = MultiObjectTracker(iou_threshold=0.2)
    A, B = (115.0, -22.0, 0.0), (115.0, 22.0, 0.0)   # A→right in frame, B→left
    dt = 1.0 / 30.0
    min_a = min_b = float("inf")
    engaged = None
    cycled = 0
    for i in range(int(130.0 / dt)):
        t = i * dt
        mode = GuidanceMode.STANDBY if t < 2.0 else GuidanceMode.DIVE
        dets = []
        for pos, conf in ((A, 0.6), (B, 0.9)):
            d = (pos[0] - af.pos[0], pos[1] - af.pos[1], pos[2] - af.pos[2])
            det, _, inf = cam.project(d, af.psi, af.phi)
            if det is not None and inf:
                dets.append(replace(det, confidence=conf))
        min_a = min(min_a, _norm((A[0] - af.pos[0], A[1] - af.pos[1], A[2] - af.pos[2])))
        min_b = min(min_b, _norm((B[0] - af.pos[0], B[1] - af.pos[1], B[2] - af.pos[2])))
        trk.auto_acquire = mode is GuidanceMode.STANDBY
        if mode is GuidanceMode.STANDBY and t > 1.0 and cycled < n_cycles:
            trk.cycle(); cycled += 1
        raw = trk.consume(None, dets, t)
        filtered = flt.update(raw, W, H, t)
        switch = SwitchState(active=mode is not GuidanceMode.STANDBY, pwm_us=1800,
                             timestamp=t, mode=mode)
        if filtered is None:
            intent = ZERO_INTENT
        else:
            intent = gate(compute_intent(filtered, servo, mode), filtered,
                          switch, True, t, safety).intent
        if engaged is None and mode is GuidanceMode.DIVE and raw is not None:
            engaged = "A" if raw.detection.x > W / 2 else "B"
        af.step(intent, dt)
        if min(min_a, min_b) <= 1.5:
            break
    return engaged, min_a, min_b


def test_multi_target_selection_determines_which_target_is_hit():
    # The whole point of selection: which target the aircraft actually dives onto
    # follows the operator's choice. No cycle → auto-locked B (higher conf) → hits
    # B, leaves A. One cycle in STANDBY → A → hits A, leaves B.
    eng0, a0, b0 = _two_target_dive(0)
    assert eng0 == "B" and b0 < 5.0 and a0 > 20.0
    eng1, a1, b1 = _two_target_dive(1)
    assert eng1 == "A" and a1 < 5.0 and b1 > 20.0


def test_ground_dive_through_real_iou_associator_holds_lock():
    # Route the closed loop through the production single-target IouAssociator (not
    # an injected Target): a distant ground target is a tiny box, so this exercises
    # the distance-gated association under the moving FOV through a full dive. The
    # lock must hold (no coast-to-drop) and the dive converge.
    from pi_fpv_companion.track.iou_associator import IouAssociator
    w = _world(target_pos=(75.0, 0.0, 0.0), alt=35.0)
    w.tracker = IouAssociator(iou_threshold=0.3, max_lost_frames=15)
    tr = w.run(GuidanceMode.DIVE, duration_s=60.0)
    assert _converges(tr)
    # The association holds the (tiny, distant) box through the whole dive with NO
    # muting — the lean soft-start ramps the steep lean in so the target never slews
    # faster than the tracker/filter can follow at commit. (Without the soft-start
    # this showed a ~0.3 s onset transient.)
    assert tr.muted_ticks == 0


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
