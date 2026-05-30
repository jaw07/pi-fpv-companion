import pytest

from pi_fpv_companion.types import Detection, FilteredTarget, GuidanceMode
from pi_fpv_companion.guidance.visual_servo import ClosureState, ServoConfig, compute_intent


def _cfg(**kw):
    base = dict(
        frame_width=720, frame_height=576,
        max_yaw_rate_dps=60.0, max_pitch_deg=15.0,
        pixel_deadzone_px=20.0, yaw_p_gain=0.15, yaw_ff_gain=0.0,
        desired_bbox_frac=0.30, closure_p_gain=4.0,   # range-linear (inverse-size) scale
    )
    base.update(kw)
    return ServoConfig(**base)


def _target(x, y, w=40, h=None, ts=1.0, vx=0.0, vy=0.0, quality=0.9):
    # h drives closure: size_frac = h / frame_height. Default ~ small/far.
    if h is None:
        h = 40
    return FilteredTarget(
        detection=Detection(x=x, y=y, w=w, h=h, confidence=quality, class_id=0),
        track_id=1, vx_px_s=vx, vy_px_s=vy, quality=quality, timestamp=ts,
    )


def test_centered_far_target_yields_no_yaw_but_forward_lean():
    cfg = _cfg()
    # Small bbox (h=40 / 576 = 0.069 << desired 0.30) -> far -> forward lean.
    out = compute_intent(_target(cfg.frame_width / 2, cfg.frame_height / 2, h=40), cfg)
    assert out.yaw_rate_dps == 0.0
    assert out.roll_deg == 0.0                        # pure pursuit, never roll
    assert out.pitch_deg < 0.0                        # nose-down = approaching
    assert out.thrust == 0.5                          # FC holds altitude in v1


def test_target_right_of_center_commands_positive_yaw():
    cfg = _cfg()
    out = compute_intent(_target(cfg.frame_width / 2 + 100, cfg.frame_height / 2), cfg)
    assert out.yaw_rate_dps > 0


def test_target_left_of_center_commands_negative_yaw():
    cfg = _cfg()
    out = compute_intent(_target(cfg.frame_width / 2 - 100, cfg.frame_height / 2), cfg)
    assert out.yaw_rate_dps < 0


def test_yaw_rate_is_clamped_to_max():
    cfg = _cfg(yaw_p_gain=10.0)
    out = compute_intent(_target(cfg.frame_width / 2 + 100, cfg.frame_height / 2), cfg)
    assert out.yaw_rate_dps == cfg.max_yaw_rate_dps


def test_default_gain_does_not_saturate_yaw_clamp_at_frame_edge():
    cfg = _cfg()
    out = compute_intent(_target(cfg.frame_width, cfg.frame_height / 2), cfg)
    assert out.yaw_rate_dps < cfg.max_yaw_rate_dps


def test_deadzone_suppresses_small_offset():
    cfg = _cfg()
    out = compute_intent(_target(cfg.frame_width / 2 + 10, cfg.frame_height / 2), cfg)
    assert out.yaw_rate_dps == 0.0


def test_far_target_leans_forward_close_target_backs_off():
    cfg = _cfg()
    far = compute_intent(_target(360, 288, h=40), cfg)            # small = far
    hold_h = int(cfg.desired_bbox_frac * cfg.frame_height)
    at = compute_intent(_target(360, 288, h=hold_h), cfg)         # at hold dist
    close = compute_intent(_target(360, 288, h=int(0.55 * cfg.frame_height)), cfg)
    assert far.pitch_deg < 0.0          # nose down -> accelerate toward it
    assert abs(at.pitch_deg) < 1.0      # arrived -> ~zero (hold station)
    assert close.pitch_deg > 0.0        # too close -> nose up -> back off (collision guard)


def test_closing_eases_the_forward_lean_monotonically():
    cfg = _cfg()
    p_far = compute_intent(_target(360, 288, h=40), cfg).pitch_deg
    p_mid = compute_intent(_target(360, 288, h=120), cfg).pitch_deg
    assert p_mid > p_far                # less negative as the target grows/nears


def test_pitch_clamped_both_directions():
    cfg = _cfg(closure_p_gain=1000.0)   # huge gain -> always saturates
    far = compute_intent(_target(360, 288, h=1), cfg)
    close = compute_intent(_target(360, 288, h=cfg.frame_height), cfg)
    assert far.pitch_deg == -cfg.max_pitch_deg
    assert close.pitch_deg == cfg.max_pitch_deg


# ---- PI closure (ClosureState): exact standoff on a mover ----

def _ft(track_id, h, ts):
    return FilteredTarget(
        detection=Detection(x=360, y=288, w=40, h=h, confidence=0.9, class_id=0),
        track_id=track_id, vx_px_s=0.0, vy_px_s=0.0, quality=0.9, timestamp=ts,
    )


def test_closure_captures_engage_distance_and_winds_in_to_hold_it():
    # TRACK captures the apparent size at engage as the hold setpoint. When the
    # target then drifts FARTHER (smaller box) the integral winds in more nose-down
    # to restore the engage gap — and a target held AT the engage size produces no
    # lean (nothing to correct), proving it holds distance rather than closing.
    cfg = _cfg(closure_p_gain=1.0, closure_i_gain=2.0)
    cs = ClosureState()
    at_engage = compute_intent(_ft(1, h=200, ts=0.0), cfg, closure=cs)  # capture @ h=200
    assert abs(at_engage.pitch_deg) < 1e-6      # at the engage distance -> no lean
    p_early = compute_intent(_ft(1, h=120, ts=0.1), cfg, closure=cs).pitch_deg  # drifted far
    out = None
    for i in range(2, 30):
        out = compute_intent(_ft(1, h=120, ts=i * 0.1), cfg, closure=cs)
    assert out.pitch_deg < p_early - 1.0        # integral winds in to restore the gap
    assert out.pitch_deg >= -cfg.max_pitch_deg


def test_closure_state_resets_and_recaptures_on_a_new_lock():
    cfg = _cfg(closure_p_gain=1.0, closure_i_gain=2.0)
    cs = ClosureState()
    compute_intent(_ft(1, h=200, ts=0.0), cfg, closure=cs)       # engage track 1 @ h=200
    for i in range(1, 20):
        compute_intent(_ft(1, h=120, ts=i * 0.1), cfg, closure=cs)   # drifted far -> winds
    assert cs.integral != 0.0 and cs.setpoint_inv is not None
    compute_intent(_ft(2, h=80, ts=2.1), cfg, closure=cs)        # operator switched target
    assert cs.integral == 0.0 and cs.track_id == 2               # windup cleared
    assert cs.setpoint_inv == 1.0 / (80 / 576)                   # new engage distance captured


def test_closure_anti_windup_bounds_the_integral_and_unwinds():
    # Drive hard into the clamp: back-calculation keeps the integral's authority
    # within the pitch budget (no runaway windup), and it unwinds once the target
    # becomes too close (error reverses) so the loop backs off instead of staying
    # pinned nose-down.
    cfg = _cfg(closure_p_gain=1.0, closure_i_gain=5.0)
    cs = ClosureState()
    compute_intent(_ft(1, h=300, ts=0.0), cfg, closure=cs)          # engage near (big box)
    for i in range(1, 200):
        compute_intent(_ft(1, h=50, ts=i * 0.1), cfg, closure=cs)   # drifted far -> saturate
    assert abs(cfg.closure_i_gain * cs.integral) <= cfg.max_pitch_deg + 1e-6
    out = None
    for i in range(80):
        out = compute_intent(_ft(1, h=500, ts=20.0 + i * 0.1), cfg, closure=cs)  # now too CLOSE
    assert out.pitch_deg > 0.0                # integral unwound -> backs off (no longer pinned down)


def test_closure_integral_inert_without_state_or_gain():
    # No ClosureState, or zero gain, -> pure-P (no accumulation): the system stays
    # backward compatible and STANDBY/DIVE (which pass closure=None) are unaffected.
    cfg = _cfg(closure_p_gain=2.0, closure_i_gain=5.0)
    a = compute_intent(_ft(1, h=80, ts=1.0), cfg).pitch_deg           # closure=None
    b = compute_intent(_ft(1, h=80, ts=2.0), cfg).pitch_deg
    assert a == b
    cs = ClosureState()
    cfg0 = _cfg(closure_p_gain=2.0, closure_i_gain=0.0)
    compute_intent(_ft(1, h=80, ts=1.0), cfg0, closure=cs)
    assert cs.integral == 0.0                  # gain 0 -> integrator never engages


def test_velocity_feedforward_adds_yaw_for_a_moving_centred_target():
    # Target dead-centre (zero P error) but moving right at 200 px/s. Pure-P
    # would command zero yaw and lag behind; feedforward pre-empts the motion.
    cfg = _cfg(yaw_ff_gain=0.1)
    centred_moving = _target(cfg.frame_width / 2, cfg.frame_height / 2, vx=200.0)
    out = compute_intent(centred_moving, cfg)
    assert out.yaw_rate_dps == 0.1 * 200.0       # purely feedforward (P term is 0)

    # And it still clamps.
    fast = _target(cfg.frame_width / 2, cfg.frame_height / 2, vx=100000.0)
    assert compute_intent(fast, cfg).yaw_rate_dps == cfg.max_yaw_rate_dps


def test_lead_pursuit_aims_ahead_of_a_crossing_target():
    # With lead, a target moving right is aimed at where it WILL be → more yaw-right
    # than pure pursuit on the same instantaneous position.
    cx, cy = 360.0, 288.0
    moving = _target(cx + 30, cy, vx=300.0)            # right of centre, moving right
    pure = compute_intent(moving, _cfg(yaw_ff_gain=0.0, lead_time_s=0.0)).yaw_rate_dps
    lead = compute_intent(moving, _cfg(yaw_ff_gain=0.0, lead_time_s=0.3)).yaw_rate_dps
    assert lead > pure > 0
    # A centred target moving right → pure pursuit sees zero error; lead aims right.
    centred = _target(cx, cy, vx=300.0)
    assert compute_intent(centred, _cfg(yaw_ff_gain=0.0, lead_time_s=0.3)).yaw_rate_dps > 0
    assert compute_intent(centred, _cfg(yaw_ff_gain=0.0, lead_time_s=0.0)).yaw_rate_dps == 0


def test_intent_timestamp_matches_target_timestamp():
    cfg = _cfg()
    out = compute_intent(_target(cfg.frame_width / 2, cfg.frame_height / 2, ts=42.0), cfg)
    assert out.timestamp == 42.0


def test_yaw_sign_inversion_flips_command_direction():
    # Audit §6: a mirrored camera inverts the sign. yaw_sign=-1 must flip it.
    base = _cfg()
    inv = _cfg(yaw_sign=-1.0)
    t = _target(base.frame_width / 2 + 100, base.frame_height / 2)
    assert compute_intent(t, base).yaw_rate_dps > 0
    assert compute_intent(t, inv).yaw_rate_dps < 0
    # Exact mirror
    assert compute_intent(t, inv).yaw_rate_dps == -compute_intent(t, base).yaw_rate_dps


def test_pitch_sign_inversion_flips_closure_direction():
    base = _cfg()
    inv = _cfg(pitch_sign=-1.0)
    far = _target(360, 288, h=40)                       # far -> forward (neg) normally
    assert compute_intent(far, base).pitch_deg < 0
    assert compute_intent(far, inv).pitch_deg > 0       # inverted


# ---- DIVE vs TRACK modes ----

def _dcfg(**kw):
    # Closed-loop DIVE config: adaptive lean + vertical-rate homing enabled.
    base = dict(dive_forward_deg=25.0, dive_climb_forward_deg=6.0, dive_max_pitch_deg=30.0,
                dive_center_frac=0.30, dive_vrate_gain=17.0,
                dive_max_descent_mps=8.0, dive_max_climb_mps=4.0)
    base.update(kw)
    return _cfg(**base)


def test_dive_lean_soft_starts_then_reaches_full_steep():
    # At commit the steep lean ramps in over dive_lean_ramp_s (so the target doesn't
    # slew faster than the filter tracks); by the ramp time it is full steep.
    cfg = _dcfg(dive_lean_ramp_s=0.5)
    cx, cy = cfg.frame_width / 2, cfg.frame_height / 2
    below = _target(cx, cy + 200)                       # descending → steep at full ramp
    at_commit = compute_intent(below, cfg, GuidanceMode.DIVE, dive_elapsed_s=0.0).pitch_deg
    mid = compute_intent(below, cfg, GuidanceMode.DIVE, dive_elapsed_s=0.25).pitch_deg
    full = compute_intent(below, cfg, GuidanceMode.DIVE, dive_elapsed_s=1.0).pitch_deg
    assert at_commit == pytest.approx(-cfg.dive_climb_forward_deg)   # starts gentle
    assert at_commit < 0 and mid < at_commit and full < mid          # ramps steeper
    assert full == pytest.approx(-cfg.dive_forward_deg)              # full steep by ramp time


def test_dive_lean_is_steep_descending_gentle_climbing():
    # Adaptive forward lean: STEEP when descending onto a below target (target low
    # in frame → commit descent), GENTLE when level/climbing toward an above one
    # (target high in frame). Always nose-down (≤0), independent of bbox size.
    cfg = _dcfg()
    cx, cy = cfg.frame_width / 2, cfg.frame_height / 2
    below = compute_intent(_target(cx, cy + 200), cfg, GuidanceMode.DIVE).pitch_deg
    above = compute_intent(_target(cx, cy - 200), cfg, GuidanceMode.DIVE).pitch_deg
    assert below == pytest.approx(-cfg.dive_forward_deg)       # full steep lean
    assert above == pytest.approx(-cfg.dive_climb_forward_deg)  # gentle lean
    assert below < above <= 0.0                                # steeper diving than climbing
    # independent of bbox size (range): a far vs near below target → same steep lean
    assert (compute_intent(_target(cx, cy + 200, h=40), cfg, GuidanceMode.DIVE).pitch_deg
            == compute_intent(_target(cx, cy + 200, h=300), cfg, GuidanceMode.DIVE).pitch_deg)


def test_dive_never_pitches_up():
    # DIVE is commit: pitch is clamped nose-down (<= 0), even for a target high in
    # frame (pitching up would fly backward — the climb is the throttle's job).
    cfg = _dcfg(dive_forward_deg=0.0)
    cx, cy = cfg.frame_width / 2, cfg.frame_height / 2
    assert compute_intent(_target(cx, cy - 200), cfg, GuidanceMode.DIVE).pitch_deg <= 0.0


def test_dive_still_centers_yaw():
    cfg = _dcfg()
    t = _target(cfg.frame_width / 2 + 100, cfg.frame_height / 2)
    assert compute_intent(t, cfg, GuidanceMode.DIVE).yaw_rate_dps > 0


def test_dive_vertical_rate_descends_below_climbs_above_holds_level():
    # Constant-bearing homing: the commanded vertical RATE drives the target's
    # vertical frame error to zero — below centre (low) -> descend (-), above
    # centre (high) -> climb (+), centred -> ~0. Gated on horizontal aim.
    cfg = _dcfg()
    cx, cy = cfg.frame_width / 2, cfg.frame_height / 2
    assert compute_intent(_target(cx, cy + 150), cfg, GuidanceMode.DIVE).vertical_rate_mps < 0
    assert compute_intent(_target(cx, cy - 150), cfg, GuidanceMode.DIVE).vertical_rate_mps > 0
    assert compute_intent(_target(cx, cy), cfg, GuidanceMode.DIVE).vertical_rate_mps == pytest.approx(0.0)
    # horizontally off-centre -> vertical commit gated off (re-aim yaw first)
    assert compute_intent(_target(cx + 0.6 * cx, cy + 150), cfg, GuidanceMode.DIVE).vertical_rate_mps == pytest.approx(0.0)
    # TRACK never commands a vertical rate
    assert compute_intent(_target(cx, cy + 150), cfg, GuidanceMode.TRACK).vertical_rate_mps is None


def test_dive_vertical_rate_grows_with_frame_error_and_clamps():
    cfg = _dcfg(dive_max_descent_mps=6.0)
    cx, cy = cfg.frame_width / 2, cfg.frame_height / 2
    near = compute_intent(_target(cx, cy + 60), cfg, GuidanceMode.DIVE).vertical_rate_mps
    far = compute_intent(_target(cx, cy + 200), cfg, GuidanceMode.DIVE).vertical_rate_mps
    assert far < near < 0                          # larger error -> stronger descent
    # saturates at the descent clamp for a target at the bottom edge
    floor = compute_intent(_target(cx, cy + cy), cfg, GuidanceMode.DIVE).vertical_rate_mps
    assert floor == pytest.approx(-6.0)


def test_dive_vertical_damping_eases_command_when_target_moves_toward_centre():
    # PD: the derivative term opposes the rate of change of the vertical error so the
    # vertical homing doesn't oscillate (the Gazebo dive wiggle). A below-centre target
    # ALREADY rising toward centre (vy<0) should get a GENTLER descent than a still one.
    cfg = _dcfg(dive_vrate_gain=17.0, dive_vrate_damp=4.0)
    cx, cy = cfg.frame_width / 2, cfg.frame_height / 2
    still = compute_intent(_target(cx, cy + 60, vy=0.0), cfg, GuidanceMode.DIVE).vertical_rate_mps
    rising = compute_intent(_target(cx, cy + 60, vy=-120.0), cfg, GuidanceMode.DIVE).vertical_rate_mps
    assert still < 0 and rising < 0                 # below centre -> both descend
    assert rising > still                           # damped: eased descent (smaller magnitude)
    # with no damping the two are identical (pure-P ignores vy)
    cfg0 = _dcfg(dive_vrate_gain=17.0, dive_vrate_damp=0.0)
    a = compute_intent(_target(cx, cy + 60, vy=0.0), cfg0, GuidanceMode.DIVE).vertical_rate_mps
    b = compute_intent(_target(cx, cy + 60, vy=-120.0), cfg0, GuidanceMode.DIVE).vertical_rate_mps
    assert a == b


def test_dive_state_low_passes_the_lean():
    from pi_fpv_companion.guidance.visual_servo import DiveState
    d = DiveState()
    assert d.smooth(12.0, 0.0, 1.0) == 12.0        # first dive frame snaps (no history)
    a = d.smooth(0.0, 0.1, 1.0)                     # lean target drops -> eased toward it
    assert 0.0 < a < 12.0                            # partial move, not a jump (anti-nod)
    assert d.smooth(0.0, 0.2, 1.0) < a               # keeps converging toward the new target
    d0 = DiveState(); d0.smooth(12.0, 0.0, 0.0)
    assert d0.smooth(0.0, 0.1, 0.0) == 0.0          # tau=0 disables smoothing (snap)


def test_dive_lean_smoothing_steadies_the_pitch():
    # With the lean low-pass, a sudden change in the would-be lean (the adaptive lean
    # flipping) only nudges the pitch — it doesn't snap. That's what kills the nod.
    from pi_fpv_companion.guidance.visual_servo import DiveState
    cfg = _dcfg(dive_lean_tau_s=1.0)
    cx, cy = cfg.frame_width / 2, cfg.frame_height / 2
    ds = DiveState()
    p1 = compute_intent(_target(cx, cy + 200, ts=0.0), cfg, GuidanceMode.DIVE, dive=ds).pitch_deg
    p2 = compute_intent(_target(cx, cy + 5, ts=0.05), cfg, GuidanceMode.DIVE, dive=ds).pitch_deg
    p2_raw = compute_intent(_target(cx, cy + 5, ts=0.05), cfg, GuidanceMode.DIVE).pitch_deg
    assert abs(p2 - p1) < abs(p2_raw - p1)          # smoothed pitch changes far less


def test_dive_vertical_disabled_when_gain_zero():
    cfg = _dcfg(dive_vrate_gain=0.0)               # vertical homing off -> DIVE just leans
    cx, cy = cfg.frame_width / 2, cfg.frame_height / 2
    assert compute_intent(_target(cx, cy + 150), cfg, GuidanceMode.DIVE).vertical_rate_mps is None
    assert compute_intent(_target(cx, cy + 150), cfg, GuidanceMode.DIVE).thrust == 0.5


def test_track_is_the_default_mode():
    cfg = _cfg()
    t = _target(360, 288, h=40)
    assert compute_intent(t, cfg) == compute_intent(t, cfg, GuidanceMode.TRACK)


# ---- TRACK vertical re-centering (accommodates camera tilt from forward lean) ----

def _hold_h(cfg):
    return cfg.desired_bbox_frac * cfg.frame_height   # size_err == 0 -> range term 0


def test_track_high_target_pitches_up_to_recenter():
    cfg = _cfg()
    cx, cy = cfg.frame_width / 2, cfg.frame_height / 2
    # at hold distance, target well ABOVE centre -> nose UP (re-centre / limit lean)
    out = compute_intent(_target(cx, cy - 150, h=_hold_h(cfg)), cfg, GuidanceMode.TRACK)
    assert out.pitch_deg > 0


def test_track_low_target_pitches_down_to_recenter():
    cfg = _cfg()
    cx, cy = cfg.frame_width / 2, cfg.frame_height / 2
    out = compute_intent(_target(cx, cy + 150, h=_hold_h(cfg)), cfg, GuidanceMode.TRACK)
    assert out.pitch_deg < 0


def test_track_vcenter_gain_zero_disables_recenter():
    cfg = _cfg(track_vcenter_gain=0.0)
    cx, cy = cfg.frame_width / 2, cfg.frame_height / 2
    # with re-centring off + at hold distance, vertical position has no pitch effect
    out = compute_intent(_target(cx, cy - 150, h=_hold_h(cfg)), cfg, GuidanceMode.TRACK)
    assert abs(out.pitch_deg) < 1e-6
