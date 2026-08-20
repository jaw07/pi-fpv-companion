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


# ---- output conditioning: incremental + stable -----------------------------------

def _steps(cfg, st, hz, seconds, cx_n=0.95, mode=GuidanceMode.DIVE):
    """Drive a hard off-centre target at `hz` for `seconds`; return the yaw-rate trace."""
    dt = 1.0 / hz
    trace = []
    n = int(seconds * hz)
    for i in range(n):
        out = compute_rate_intent(_ft(cx_n, 0.5), cfg, st, now=i * dt, mode=mode,
                                  pitch_rad=0.0, roll_rad=0.0, gamma_rad=0.0, agl_m=40.0)
        trace.append(out.yaw_rate)
    return trace


def test_smoothing_is_independent_of_control_loop_rate():
    """The smoothing must be a function of TIME, not of tick count.

    This is a real regression guard: the old per-tick EMA coefficients meant the
    5Hz -> 22Hz control-loop fix silently cut the yaw smoothing time constant by ~4x.
    Same wall-clock elapsed, same commanded rate, whatever the loop rate."""
    slow = _steps(RateConfig(W, H), RateState(), hz=5.0, seconds=1.0)
    fast = _steps(RateConfig(W, H), RateState(), hz=22.0, seconds=1.0)
    # compare at the same wall-clock instant (end of a 1s run)
    assert abs(slow[-1] - fast[-1]) < 0.05 * max(1e-6, abs(fast[-1])) + 0.02, (
        f"rate-dependent smoothing: 5Hz ended at {slow[-1]:.4f}, 22Hz at {fast[-1]:.4f}")


def test_commanded_rates_move_incrementally_on_a_step_input():
    """A fresh lock hard off-centre is a step into the controller. The commanded body
    rate must WALK toward the demand at no more than the slew limit — never jump."""
    cfg = RateConfig(W, H)
    st = RateState()
    hz = 22.0
    dt = 1.0 / hz
    prev = 0.0
    worst = 0.0
    for i in range(60):
        out = compute_rate_intent(_ft(0.98, 0.5), cfg, st, now=i * dt,
                                  mode=GuidanceMode.DIVE, pitch_rad=0.0, roll_rad=0.0,
                                  gamma_rad=0.0, agl_m=40.0)
        d = abs(out.yaw_rate - prev) / dt
        worst = max(worst, d)
        prev = out.yaw_rate
    assert worst <= cfg.slew_yaw + 1e-6, f"yaw slewed at {worst:.2f} rad/s^2 > {cfg.slew_yaw}"


def test_first_tick_commands_nothing():
    """dt is 0 on the very first tick of an engagement. The output must stay at zero
    rather than jumping to the controller's demand — engaging must not kick."""
    out = compute_rate_intent(_ft(0.98, 0.9), RateConfig(W, H), RateState(), now=0.0,
                              mode=GuidanceMode.DIVE, pitch_rad=0.0, roll_rad=0.0,
                              gamma_rad=0.0, agl_m=40.0)
    assert out.yaw_rate == 0.0 and out.pitch_rate == 0.0 and out.roll_rate == 0.0


def test_commanded_rates_are_clamped():
    """Whatever the controller asks for, the commanded body rates stay inside the
    configured envelope."""
    cfg = RateConfig(W, H, max_yaw_rate=0.30, max_pitch_rate=0.25, max_roll_rate=0.20)
    st = RateState()
    dt = 1.0 / 22.0
    for i in range(200):
        out = compute_rate_intent(_ft(0.99, 0.99), cfg, st, now=i * dt,
                                  mode=GuidanceMode.DIVE, pitch_rad=0.0, roll_rad=0.0,
                                  gamma_rad=0.0, agl_m=40.0)
        assert abs(out.yaw_rate) <= cfg.max_yaw_rate + 1e-9
        assert abs(out.pitch_rate) <= cfg.max_pitch_rate + 1e-9
        assert abs(out.roll_rate) <= cfg.max_roll_rate + 1e-9


def test_track_mode_output_is_also_conditioned():
    """TRACK uses a different pitch law but the same output conditioning."""
    cfg = RateConfig(W, H)
    st = RateState()
    dt = 1.0 / 22.0
    prev = 0.0
    worst = 0.0
    for i in range(60):
        out = compute_rate_intent(_ft(0.95, 0.5, h=20), cfg, st, now=i * dt,
                                  mode=GuidanceMode.TRACK, pitch_rad=0.0, roll_rad=0.0,
                                  gamma_rad=0.0, agl_m=40.0)
        worst = max(worst, abs(out.pitch_rate - prev) / dt)
        prev = out.pitch_rate
    assert worst <= cfg.slew_pitch + 1e-6


def test_throttle_never_commands_idle_in_a_dive():
    """MEASURED REGRESSION (2026-08-18): thrust_ilim == thrust_out meant the pursuit
    integral alone saturated the loop, so a sustained dive error drove throttle to
    EXACTLY 0.0 and held it — motors at idle, freefall, and on a quad no headroom left
    to differential-thrust against, so attitude authority goes with it."""
    cfg = RateConfig(W, H)
    st = RateState(); st.hover = 0.30
    dt = 1.0 / 22.0
    out = None
    for i in range(200):
        out = compute_rate_intent(_ft(0.5, 0.85), cfg, st, now=i * dt,
                                  mode=GuidanceMode.DIVE, pitch_rad=-0.2, roll_rad=0.0,
                                  gamma_rad=0.0, agl_m=40.0)
    floor = cfg.min_thrust_frac * st.hover
    assert out.thrust >= floor - 1e-9, f"throttle sank to {out.thrust:.4f}, floor {floor:.4f}"
    assert out.thrust > 0.0


def test_throttle_moves_incrementally():
    """Throttle was the one channel with no slew limit; it fell at 1.71/s (hover to zero
    in well under a second). That is the 'overcompensating' feel."""
    cfg = RateConfig(W, H)
    st = RateState(); st.hover = 0.30
    dt = 1.0 / 22.0
    prev = st.hover
    worst = 0.0
    for i in range(200):
        out = compute_rate_intent(_ft(0.5, 0.85), cfg, st, now=i * dt,
                                  mode=GuidanceMode.DIVE, pitch_rad=-0.2, roll_rad=0.0,
                                  gamma_rad=0.0, agl_m=40.0)
        worst = max(worst, abs(out.thrust - prev) / dt)
        prev = out.thrust
    assert worst <= cfg.slew_thrust + 1e-6, f"throttle slewed at {worst:.2f}/s"


def test_throttle_floor_is_the_higher_of_absolute_and_hover_relative():
    """hover is learned in flight and varies with airframe/battery, so the floor scales
    with it — BUT the hover-relative floor is only as trustworthy as hover itself. A
    runaway learner drags the floor down with it (measured: hover collapsed 0.45 -> 0.056,
    floor 0.180 -> 0.023). The absolute floor is what a bad hover cannot remove."""
    cfg = RateConfig(W, H)
    for hover in (0.20, 0.45):
        st = RateState(); st.hover = hover
        dt = 1.0 / 22.0
        out = None
        for i in range(200):
            out = compute_rate_intent(_ft(0.5, 0.85), cfg, st, now=i * dt,
                                      mode=GuidanceMode.DIVE, pitch_rad=-0.2,
                                      roll_rad=0.0, gamma_rad=0.0, agl_m=40.0)
        expected = max(cfg.min_thrust_abs, cfg.min_thrust_frac * hover)
        assert abs(out.thrust - expected) < 1e-6
    # a collapsed hover must not be able to take the floor with it
    st = RateState(); st.hover = 0.056
    for i in range(200):
        out = compute_rate_intent(_ft(0.5, 0.85), cfg, st, now=i * dt,
                                  mode=GuidanceMode.DIVE, pitch_rad=-0.2,
                                  roll_rad=0.0, gamma_rad=0.0, agl_m=40.0)
    assert out.thrust >= cfg.min_thrust_abs - 1e-9


def test_hover_trim_is_time_based_not_per_tick():
    """MEASURED REGRESSION: the trim was `hover -= 0.01 * climb` applied per TICK, so its
    rate was a function of the control-loop rate. At 22Hz with a 12m/s climb that is
    -2.6/s and hover slams to its clamp in ~0.15s, collapsing thrust and dropping the
    aircraft. Same wall-clock, same result, whatever the loop rate."""
    from pi_fpv_companion.guidance.rate_control import trim_hover
    cfg = RateConfig(W, H)
    def run(hz, seconds=1.0, climb=1.0):
        h, dt = 0.45, 1.0 / hz
        for _ in range(int(seconds * hz)):
            h = trim_hover(h, climb, dt, cfg)
        return h
    assert abs(run(5.0) - run(22.0)) < 0.01
    assert abs(run(22.0) - run(50.0)) < 0.01


def test_hover_trim_ignores_large_climb_rates():
    """Engaging TRACK while still climbing hard is the normal case (you climb to altitude
    then engage). A big climb says the craft is nowhere near hover, so it carries no
    information about what hover is — learning from it is what caused the collapse."""
    from pi_fpv_companion.guidance.rate_control import trim_hover
    cfg = RateConfig(W, H)
    assert trim_hover(0.45, 12.0, 0.045, cfg) == 0.45     # ignored
    assert trim_hover(0.45, 1.0, 0.045, cfg) < 0.45       # gentle trim down


def test_hover_trim_rate_is_bounded():
    from pi_fpv_companion.guidance.rate_control import trim_hover
    cfg = RateConfig(W, H)
    h0 = 0.45
    h1 = trim_hover(h0, cfg.hover_learn_max_climb, 1.0, cfg)   # a full second
    assert abs(h1 - h0) <= cfg.hover_learn_max_per_s + 1e-9


def test_impact_stop_may_still_cut_throttle():
    """The floor protects the dive; the impact latch is a deliberate cut and must not be
    floored, or 'STOP' would keep the props driving after ground contact."""
    cfg = RateConfig(W, H)
    st = RateState(); st.hover = 0.30
    st.had_lock = True
    dt = 1.0 / 22.0
    out = None
    for i in range(40):
        out = compute_rate_intent(None, cfg, st, now=i * dt, mode=GuidanceMode.DIVE,
                                  pitch_rad=0.0, roll_rad=0.0, gamma_rad=0.0,
                                  agl_m=cfg.impact_agl_m - 1.0)
    assert out.phase == "STOP"
    assert out.thrust < cfg.min_thrust_frac * st.hover
