"""Integration tests for ArduPilotBackend using a UDP loopback fake ArduCopter.

These exercise the real MAVLink wire protocol — pymavlink encoding,
HEARTBEAT-driven armed state, RC_CHANNELS-driven switch state, and the outbound
RC_CHANNELS_OVERRIDE AETR control path (ALT_HOLD, GPS-denied). No SITL or
hardware required; SITL (scripts/validate_sitl.py) is the ground truth for the
real ArduCopter response + stick signs.
"""
from __future__ import annotations
import socket
import time

import pytest

from pi_fpv_companion.fc.ardupilot import (
    ArduPilotBackend, ArduCopterRcMapping, _STREAM_REREQUEST_S)
from pi_fpv_companion.types import GuidanceIntent, GuidanceMode
from tests.fakes.fake_ardupilot import FakeArduCopter


def _free_udp_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def ap_pair():
    port = _free_udp_port()
    backend = ArduPilotBackend(
        device=f"udpin:127.0.0.1:{port}",
        baud=0,
        switch_channel=7,
        track_threshold_us=1300,
        dive_threshold_us=1700,
    )
    backend.open()                  # bind first so packets aren't dropped
    fake = FakeArduCopter(target_port=port)
    fake.start()
    backend.wait_ready(timeout=3.0)  # blocks until first heartbeat lands
    yield backend, fake
    backend.close()
    fake.stop()


def test_backend_reflects_fc_armed_state(ap_pair):
    backend, fake = ap_pair

    fake.armed = True
    time.sleep(0.25)
    assert backend.is_armed()

    fake.armed = False
    time.sleep(0.25)
    assert not backend.is_armed()


def test_agl_captures_ground_home_while_disarmed_and_freezes_at_arm(ap_pair):
    # AGL = alt - ground-home. Home must be captured while DISARMED (and refreshed to the
    # ground), then frozen at arming, so AGL reads true height above the takeoff point.
    backend, fake = ap_pair
    fake.armed = False
    fake.alt = 100.0                         # on the ground at 100 m AMSL
    time.sleep(0.3)
    backend.is_armed()                       # pump the drain (as the pipeline does each tick)
    assert abs(backend.agl_m()) < 1.0        # ~0 AGL on the ground

    fake.armed = True                        # take off...
    fake.alt = 140.0                         # ...climb 40 m; home stays frozen at 100
    time.sleep(0.3)
    backend.is_armed()
    assert abs(backend.agl_m() - 40.0) < 1.0


def test_agl_never_homes_when_started_mid_flight_armed(ap_pair):
    # Mid-flight process restart: the backend only ever sees the craft ARMED, so it must NOT
    # capture a ground home at altitude (which would make AGL ~0 and false-latch the impact
    # STOP). agl_m() stays large until a real ground reference exists.
    backend, fake = ap_pair
    fake.armed = True
    fake.alt = 140.0
    time.sleep(0.3)
    backend.is_armed()                       # pump the drain
    assert backend.agl_m() > 1e6


def test_backend_reads_switch_channel_pwm(ap_pair):
    backend, fake = ap_pair

    fake.rc_channels[6] = 1800       # ch7 (0-indexed 6), >= dive threshold
    time.sleep(0.6)                  # escalation debounce is now 5 samples / 350ms
    s = backend.read_switch()
    assert s.pwm_us == 1800
    assert s.active is True
    assert s.mode is GuidanceMode.DIVE

    fake.rc_channels[6] = 1500       # between track and dive -> TRACK
    time.sleep(0.25)
    s = backend.read_switch()
    assert s.mode is GuidanceMode.TRACK
    assert s.active is True

    fake.rc_channels[6] = 1200       # below track -> STANDBY
    time.sleep(0.25)
    s = backend.read_switch()
    assert s.pwm_us == 1200
    assert s.active is False
    assert s.mode is GuidanceMode.STANDBY


def test_ensure_params_writes_mismatched_and_leaves_correct():
    # Startup FC validation: a param at the wrong value is written + verified; one
    # already correct is left alone.
    port = _free_udp_port()
    backend = ArduPilotBackend(device=f"udpin:127.0.0.1:{port}", baud=0, switch_channel=7,
                               track_threshold_us=1300, dive_threshold_us=1700)
    backend.open()
    fake = FakeArduCopter(target_port=port)
    fake.params = {"ANGLE_MAX": 3000.0, "RC7_OPTION": 0.0}   # ANGLE_MAX wrong, RC7 right
    fake.start()
    try:
        backend.wait_ready(timeout=3.0)
        status = backend.ensure_params({"ANGLE_MAX": 4500.0, "RC7_OPTION": 0.0})
        assert status["ANGLE_MAX"] == "set"        # corrected
        assert status["RC7_OPTION"] == "ok"        # already right, untouched
        assert fake.params["ANGLE_MAX"] == 4500.0  # write landed on the FC
    finally:
        backend.close()
        fake.stop()


def test_ensure_params_aborts_after_first_unresponsive_read():
    # A dead/wrong-baud FC must not stall boot: the first no-response read aborts
    # the whole pass (no timeout × N params). read_param is stubbed to return None.
    b = _stab_backend()
    calls = []
    b.read_param = lambda name, timeout=2.0: calls.append(name) or None
    status = b.ensure_params({"ANGLE_MAX": 4500.0, "RC7_OPTION": 0.0, "RC9_OPTION": 0.0})
    assert len(calls) == 1                      # bailed after the first failed read
    assert all(v == "read-fail" for v in status.values())


def test_release_resets_vertical_rate_integral():
    b = _stab_backend()
    b._vrate_i = 123.0
    b.release()                                  # STANDBY handback
    assert b._vrate_i == 0.0                      # no stale bias carried into next DIVE


def test_backend_reads_select_channel_over_the_wire():
    # The target-select channel (ch9 — the first free CRSF channel on the Tango 2)
    # must reach select_pwm() so the pipeline can edge-detect a cycle.
    port = _free_udp_port()
    backend = ArduPilotBackend(device=f"udpin:127.0.0.1:{port}", baud=0, switch_channel=7,
                               select_channel=9, track_threshold_us=1300, dive_threshold_us=1700)
    backend.open()
    fake = FakeArduCopter(target_port=port)
    fake.start()
    try:
        backend.wait_ready(timeout=3.0)
        fake.rc_channels[8] = 1900            # ch9 (0-indexed 8) high
        time.sleep(0.25)
        backend.read_switch()                 # drains RC_CHANNELS
        assert backend.select_pwm() == 1900
    finally:
        backend.close()
        fake.stop()


def test_backend_reads_attitude_pitch_over_the_wire(ap_pair):
    import math
    backend, fake = ap_pair
    fake.pitch_rad = math.radians(-18.0)     # FC reports an 18° nose-down dive
    time.sleep(0.25)
    backend.read_switch()                     # drains inbound telemetry (incl. ATTITUDE)
    assert backend.pitch_deg() == pytest.approx(-18.0, abs=1.0)


def test_send_intent_overrides_aetr_channels(ap_pair):
    backend, fake = ap_pair

    intent = GuidanceIntent(
        roll_deg=0.0,
        pitch_deg=-15.0,        # nose down = forward
        yaw_rate_dps=45.0,
        thrust=0.5,             # hold altitude
        timestamp=0.0,
    )
    backend.send_intent(intent)
    time.sleep(0.25)

    assert len(fake.captured_overrides) >= 1
    msg = fake.captured_overrides[-1]

    # Default RCMAP: ch1 roll, ch2 pitch, ch3 throttle, ch4 yaw.
    assert msg.chan1_raw == 1500            # roll 0 -> centered
    assert msg.chan3_raw == 1450            # thrust 0.5 -> hover (stabilize default)
    assert msg.chan4_raw > 1500             # +yaw rate -> stick above center
    assert msg.chan2_raw < 1500             # nose-down pitch -> stick below center
    # ch5..8 released (0) so the pilot keeps mode + engage switches on the radio.
    assert msg.chan5_raw == 0
    assert msg.chan7_raw == 0


def test_release_hands_all_channels_back(ap_pair):
    backend, fake = ap_pair

    backend.release()
    time.sleep(0.25)

    assert len(fake.captured_overrides) >= 1
    msg = fake.captured_overrides[-1]
    # 0 on every channel = "use the receiver" -> full manual handback.
    assert (msg.chan1_raw, msg.chan2_raw, msg.chan3_raw, msg.chan4_raw) == (0, 0, 0, 0)


# ---- adaptive hover (STABILIZE companion vertical-velocity hold) ----

def _stab_backend(gain=100.0):
    return ArduPilotBackend(
        device="udpin:127.0.0.1:1", baud=0, switch_channel=7,
        track_threshold_us=1300, dive_threshold_us=1700,
        mapping=ArduCopterRcMapping(control_mode="stabilize", hover_throttle_us=1400,
                                    hover_learn=True, hover_learn_gain=gain),
    )


def _hold(thrust=0.5):
    return GuidanceIntent(0.0, 0.0, 0.0, thrust, 0.0)


def _seed(b, climb):
    t = time.monotonic() - 0.1          # dt ~0.1s, telemetry fresh
    b._hover_t = t
    b._climb_t = t
    b._climb_mps = climb


def test_adaptive_hover_raises_throttle_when_descending():
    b = _stab_backend()
    _seed(b, -2.0)                      # descending
    b._adaptive_throttle(_hold())
    assert b._hover_pwm > 1400.0        # learned to add throttle


def test_adaptive_hover_lowers_throttle_when_climbing():
    b = _stab_backend()
    _seed(b, +2.0)                      # climbing
    b._adaptive_throttle(_hold())
    assert b._hover_pwm < 1400.0


def test_adaptive_hover_frozen_during_commanded_dive():
    b = _stab_backend()
    _seed(b, -2.0)                      # descending, but...
    out = b._adaptive_throttle(_hold(thrust=0.0))   # ...a commanded dive
    assert b._hover_pwm == 1400.0       # learning frozen
    assert out <= 1010                 # throttle near minimum (real dive)


def test_adaptive_hover_frozen_when_telemetry_stale():
    b = _stab_backend()
    b._hover_t = time.monotonic() - 0.1
    b._climb_t = time.monotonic() - 2.0   # stale (>0.5 s)
    b._climb_mps = -2.0
    b._adaptive_throttle(_hold())
    assert b._hover_pwm == 1400.0


def test_adaptive_hover_clamped_to_max():
    b = _stab_backend()
    b._hover_pwm = 1690.0
    _seed(b, -50.0)                     # huge descent would push past the clamp
    b._adaptive_throttle(_hold())
    assert b._hover_pwm <= 1700.0


def test_gentle_thrust_stick_descent_is_not_swallowed_by_the_hold_band():
    # The open-loop thrust-stick path: a small thrust offset (e.g. 0.38) must pass
    # the hold band (0.05) and descend, not be cancelled by the hover PI loop.
    b = _stab_backend()
    _seed(b, 0.0)                       # level, fresh telemetry
    out = b._adaptive_throttle(_hold(thrust=0.38))   # gentle commanded descent
    assert out < 1400                  # below the learned hover → it descends
    assert b._hover_pwm == 1400.0      # learning frozen (not "holding")


def test_gentle_climb_commit_raises_throttle_above_hover():
    b = _stab_backend()
    _seed(b, 0.0)
    out = b._adaptive_throttle(_hold(thrust=0.62))   # gentle commanded climb (above target)
    assert out > 1400                  # above hover → it climbs
    assert b._hover_pwm == 1400.0


# ---- airframe pitch feedback (agnostic DIVE LOS-elevation framing) ----

def test_vertical_rate_integral_reduces_steady_state_droop():
    # While tracking a commanded rate the pure-P term leaves droop; the integral
    # accumulates the rate error and drives the throttle harder until it reaches
    # the setpoint. With measured climb stuck above the (descending) setpoint, the
    # throttle must drop further each tick and the integral term goes negative.
    b = _stab_backend()
    b._climb_mps = -2.0                              # measured descent, but we want -4
    intent = GuidanceIntent(0.0, 0.0, 0.0, 0.5, 0.0, vertical_rate_mps=-4.0)
    outs = []
    for _ in range(6):
        now = time.monotonic()
        b._hover_t = now - 0.1                       # simulate dt = 0.1 s
        b._climb_t = now
        outs.append(b._adaptive_throttle(intent))
    assert outs[-1] < outs[0]                        # throttle pushed further down
    assert b._vrate_i < 0                            # integral driving more descent


def test_vertical_rate_integral_resets_when_not_tracking():
    b = _stab_backend()
    b._climb_mps = -2.0
    b._climb_t = time.monotonic()
    b._hover_t = time.monotonic() - 0.1
    b._adaptive_throttle(GuidanceIntent(0.0, 0.0, 0.0, 0.5, 0.0, vertical_rate_mps=-4.0))
    assert b._vrate_i != 0.0
    # a plain hold (no commanded rate) clears the integral (no windup carryover)
    b._climb_t = time.monotonic()
    b._hover_t = time.monotonic() - 0.1
    b._adaptive_throttle(_hold())
    assert b._vrate_i == 0.0


def test_pitch_deg_reports_fresh_attitude():
    import math
    b = _stab_backend()
    b._pitch_rad = math.radians(-25.0)        # nose 25° down (a dive)
    b._pitch_t = time.monotonic()
    assert b.pitch_deg() == pytest.approx(-25.0, abs=0.01)


def test_pitch_deg_falls_back_to_level_when_stale():
    import math
    b = _stab_backend()
    b._pitch_rad = math.radians(-25.0)
    b._pitch_t = time.monotonic() - 2.0       # stale (>0.5 s) → unsafe to trust
    assert b.pitch_deg() == 0.0


def test_pitch_deg_falls_back_to_level_before_any_attitude():
    b = _stab_backend()
    assert b.pitch_deg() == 0.0               # never received ATTITUDE yet


# ---- control_ready interlock (don't override into the wrong FC mode) ----

def test_control_ready_true_only_in_matching_mode():
    b = _stab_backend()                 # control_mode = stabilize -> expects mode 0
    b._current_mode = 0                  # FC in STABILIZE
    assert b.control_ready() is True
    b._current_mode = 2                  # FC in ALT_HOLD -> mismatch
    assert b.control_ready() is False
    b._current_mode = None               # unknown -> not ready (safe default)
    assert b.control_ready() is False


def test_control_ready_respects_althold_control_mode():
    b = ArduPilotBackend(device="udpin:127.0.0.1:1", baud=0, switch_channel=7,
                         track_threshold_us=1300, dive_threshold_us=1700,
                         mapping=ArduCopterRcMapping(control_mode="althold"))
    b._current_mode = 2                  # ALT_HOLD
    assert b.control_ready() is True
    b._current_mode = 0                  # STABILIZE -> mismatch for althold
    assert b.control_ready() is False


def test_streams_rerequested_periodically(ap_pair, monkeypatch):
    # RC_CHANNELS + VFR_HUD must be re-asked after a link blip, not only at startup.
    backend, _ = ap_pair
    calls = []
    monkeypatch.setattr(backend, "_request_streams", lambda *a, **k: calls.append(1))
    backend._last_stream_req = 0.0
    backend._drain()                    # >5 s since 0 -> re-request
    assert len(calls) == 1
    backend._drain()                    # immediate -> gated, no re-request
    assert len(calls) == 1
    backend._last_stream_req -= _STREAM_REREQUEST_S + 1   # pretend interval elapsed
    backend._drain()
    assert len(calls) == 2


def _bits_backend_with_params(params):
    """Spin a backend talking to a FakeArduCopter preloaded with `params`."""
    port = _free_udp_port()
    backend = ArduPilotBackend(device=f"udpin:127.0.0.1:{port}", baud=0, switch_channel=7,
                               track_threshold_us=1300, dive_threshold_us=1700)
    backend.open()
    fake = FakeArduCopter(target_port=port)
    fake.params = dict(params)
    fake.start()
    backend.wait_ready(timeout=3.0)
    return backend, fake


# ---- ch7 auto-engage (auto_guided): DO_SET_MODE into GUIDED_NOGPS ----

def _guided_backend(auto_guided=True, start_mode=0):
    """Backend with control_mode=guided_nogps + a fake starting in `start_mode`."""
    port = _free_udp_port()
    backend = ArduPilotBackend(
        device=f"udpin:127.0.0.1:{port}", baud=0, switch_channel=7,
        track_threshold_us=1300, dive_threshold_us=1700, auto_guided=auto_guided,
        mapping=ArduCopterRcMapping(control_mode="guided_nogps"))
    backend.open()
    fake = FakeArduCopter(target_port=port)
    fake.custom_mode = start_mode
    fake.start()
    backend.wait_ready(timeout=3.0)
    return backend, fake


def _pump(backend, fake, seconds, until=None):
    """Drive the backend (drains telemetry + services the mode command) for up to
    `seconds`, stopping early when `until()` is true. Returns whether it stopped early."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        backend.read_switch()        # _drain -> _service_mode
        if until is not None and until():
            return True
        time.sleep(0.02)
    return until() if until is not None else False


def test_set_engaged_commands_and_confirms_guided_nogps():
    backend, fake = _guided_backend(start_mode=0)            # FC in STABILIZE
    try:
        _pump(backend, fake, 0.5, lambda: backend._current_mode == 0)
        backend.set_engaged(True)                            # ch7 -> TRACK/DIVE
        ok = _pump(backend, fake, 2.0, lambda: backend._current_mode == 20)
        assert ok and fake.custom_mode == 20                 # FC now in GUIDED_NOGPS
        assert 20 in fake.set_mode_cmds
        assert backend._target_mode is None                  # confirmed -> no pending retry
    finally:
        backend.close(); fake.stop()


def test_disengage_restores_prior_mode():
    backend, fake = _guided_backend(start_mode=0)            # prior mode = STABILIZE
    try:
        _pump(backend, fake, 0.5, lambda: backend._current_mode == 0)
        backend.set_engaged(True)
        assert _pump(backend, fake, 2.0, lambda: fake.custom_mode == 20)
        backend.set_engaged(False)                           # ch7 -> STANDBY
        assert _pump(backend, fake, 2.0, lambda: fake.custom_mode == 0)  # restored
    finally:
        backend.close(); fake.stop()


def test_already_guided_engage_saves_nothing_and_disengage_sends_no_restore():
    backend, fake = _guided_backend(start_mode=20)           # already GUIDED_NOGPS
    try:
        _pump(backend, fake, 0.5, lambda: backend._current_mode == 20)
        backend.set_engaged(True)                            # no prior mode to save
        assert backend._saved_mode is None
        backend.set_engaged(False)
        _pump(backend, fake, 0.5)
        assert fake.custom_mode == 20                        # never knocked out of guided
    finally:
        backend.close(); fake.stop()


def test_mode_command_retries_until_confirmed_on_dropped_commands():
    backend, fake = _guided_backend(start_mode=0)
    fake.drop_first_mode_cmds = 2                            # UART drops the first two
    try:
        _pump(backend, fake, 0.5, lambda: backend._current_mode == 0)
        backend.set_engaged(True)
        ok = _pump(backend, fake, 3.0, lambda: fake.custom_mode == 20)
        assert ok and len(fake.set_mode_cmds) >= 3           # re-sent past the drops
    finally:
        backend.close(); fake.stop()


def _offline_guided_backend():
    """Backend with NO transport (never opened): _send_mode/_drain no-op, so the
    mode-command and switch-debounce state machines can be unit-driven directly."""
    return ArduPilotBackend(
        device="udpin:127.0.0.1:1", baud=0, switch_channel=7,
        track_threshold_us=1300, dive_threshold_us=1700, auto_guided=True,
        mapping=ArduCopterRcMapping(control_mode="guided_nogps"))


def test_orphan_recovery_hands_back_when_restart_orphaned_guided():
    # A restart left the FC in GUIDED_NOGPS (20) with dead sticks, no engage this
    # session, armed, switch STANDBY -> command STABILIZE (0) to hand control back.
    b = _offline_guided_backend()
    b._armed = True
    b._current_mode = 20
    assert b.recover_orphaned_mode() is True
    assert b._target_mode == 0                       # commanding the recover mode (STABILIZE)


def test_orphan_recovery_is_one_shot():
    b = _offline_guided_backend()
    b._armed = True; b._current_mode = 20
    assert b.recover_orphaned_mode() is True
    b._current_mode = 20                             # still stuck (command not yet confirmed)
    assert b.recover_orphaned_mode() is False        # never fires twice


def test_orphan_recovery_noop_when_fc_not_in_our_mode():
    b = _offline_guided_backend()
    b._armed = True; b._current_mode = 0             # FC already in STABILIZE — pilot has control
    assert b.recover_orphaned_mode() is False
    assert b._target_mode is None


def test_orphan_recovery_noop_after_our_own_engage():
    # If WE engaged this session, GUIDED_NOGPS is ours to restore, not an orphan.
    b = _offline_guided_backend()
    b._armed = True; b._current_mode = 0; b._hb_count = 1
    b.set_engaged(True)                              # we command GUIDED -> _engaged_ever=True
    b._current_mode = 20
    assert b.recover_orphaned_mode() is False


def test_orphan_recovery_waits_for_heartbeat_before_latching():
    # No HEARTBEAT yet (current_mode None): don't latch, keep waiting; then evaluate once.
    b = _offline_guided_backend()
    b._armed = True; b._current_mode = None
    assert b.recover_orphaned_mode() is False
    assert b._orphan_checked is False                # did NOT latch — will re-evaluate
    b._current_mode = 20
    assert b.recover_orphaned_mode() is True


def test_orphan_recovery_noop_when_disarmed():
    b = _offline_guided_backend()
    b._armed = False; b._current_mode = 20           # not flying -> no recovery
    assert b.recover_orphaned_mode() is False


def test_reengage_race_own_restore_heartbeat_is_not_a_pilot_override():
    # Quick STANDBY->TRACK re-toggle: the disengage's restore is still unconfirmed when
    # the re-engage starts. Neither the STALE current_mode (== new target) may false-
    # confirm the new command, nor may the restore's late-landing HEARTBEAT be read as
    # a pilot override (it is our own command arriving).
    b = _offline_guided_backend()
    b._current_mode = 0; b._hb_count = 1
    b.set_engaged(True)                              # cmd GUIDED_NOGPS, saved=STABILIZE
    b._current_mode = 20; b._hb_count = 2
    b._service_mode()
    assert b._target_mode is None                    # engage confirmed
    b.set_engaged(False)                             # restore STABILIZE (pending)
    b.set_engaged(True)                              # re-engage BEFORE restore confirms
    b._service_mode()
    assert b._target_mode == 20, "stale current_mode==target must NOT false-confirm"
    b._current_mode = 0; b._hb_count = 3             # our restore lands late
    b._service_mode()
    assert b._target_mode == 20, "own late restore must not read as pilot override"
    b._current_mode = 2; b._hb_count = 4             # pilot really flips to ALT_HOLD
    b._service_mode()
    assert b._target_mode is None, "a real pilot mode change must cancel"


def test_disengage_skips_restore_when_pilot_already_moved_the_fc():
    # Pilot manual recovery mid-engagement (FC flipped to a third mode), then ch7 ->
    # STANDBY: the companion must NOT restore the saved mode over the pilot's choice.
    b = _offline_guided_backend()
    b._current_mode = 0; b._hb_count = 1
    b.set_engaged(True)
    b._current_mode = 20; b._hb_count = 2
    b._service_mode()                                # engaged + confirmed
    b._current_mode = 2; b._hb_count = 3             # pilot takes ALT_HOLD
    b.set_engaged(False)
    assert b._target_mode is None and b._saved_mode is None


def test_switch_escalation_debounced_deescalation_instant():
    # A brief RC glitch must never engage TRACK/DIVE (flight-3 hardening: 5 samples /
    # 350 ms). De-escalation toward STANDBY commits immediately (disengage never lags).
    # Timestamps are FC time_boot_ms (10 Hz stream = 100 ms apart).
    b = _offline_guided_backend()
    for i, t in enumerate((0, 100, 200, 300)):
        b._update_switch(1800, t)
        assert b._last_switch.mode is GuidanceMode.STANDBY  # <5 samples / <350ms: held
    b._update_switch(1800, 400)                             # 5th sample, 400ms span
    assert b._last_switch.mode is GuidanceMode.DIVE         # now confirmed
    b._update_switch(1000, 500)
    assert b._last_switch.mode is GuidanceMode.STANDBY      # disengage: instant
    # alternating glitches never accumulate an engage
    for t in (600, 700, 800, 900, 1000, 1100):
        b._update_switch(1500 if t % 200 == 0 else 1000, t)
    assert b._last_switch.mode is GuidanceMode.STANDBY
    # a steady, sustained switch DOES engage (5 samples spanning >=350ms)
    for t in (2000, 2100, 2200, 2300, 2400):
        b._update_switch(1500, t)
    assert b._last_switch.mode is GuidanceMode.TRACK


def test_switch_escalation_requires_real_time_span_not_just_sample_count():
    # A burst of queued/duplicated RC_CHANNELS delivered in one drain carries the
    # same (or near-same) FC timestamp — sample COUNT alone must not confirm.
    b = _offline_guided_backend()
    for _ in range(8):
        b._update_switch(1800, 5000)                       # 8 duplicates, zero time span
    assert b._last_switch.mode is GuidanceMode.STANDBY
    b._update_switch(1800, 5100)                           # span 100ms: still < 350ms
    assert b._last_switch.mode is GuidanceMode.STANDBY
    b._update_switch(1800, 5400)                           # span 400ms >= 350ms -> confirmed
    assert b._last_switch.mode is GuidanceMode.DIVE


def test_switch_invalid_pwm_fails_to_standby():
    # Out-of-band PWM (receiver failsafe, or the MAVLink invalid-channel sentinel
    # 65535) must NOT read as engage — flight-3 root cause. 65535 unclamped exceeded
    # dive_threshold and read as DIVE, letting a failsafe frame command GUIDED_NOGPS.
    b = _offline_guided_backend()
    assert b._mode_for(65535) is GuidanceMode.STANDBY
    assert b._mode_for(0) is GuidanceMode.STANDBY
    assert b._mode_for(1800) is GuidanceMode.DIVE          # valid value still works
    # a sustained stream of the invalid sentinel never engages
    for t in range(0, 1000, 100):
        b._update_switch(65535, t)
    assert b._last_switch.mode is GuidanceMode.STANDBY


def test_release_bursts_then_goes_radio_silent():
    # Steady-state STANDBY must put NOTHING on the wire that touches control inputs:
    # release() transmits a short zero-override burst (startup, and after overrides
    # were active), then stops sending entirely.
    from pi_fpv_companion.types import ZERO_INTENT
    b = _offline_guided_backend()
    sent = []
    b._mav = object()                                    # "open" enough for the send path
    b._send_channels = lambda overrides: sent.append(dict(overrides))
    for _ in range(30):
        b.release()
    assert len(sent) == 8, "startup burst then silence"
    b.send_intent(ZERO_INTENT)                           # overrides go active
    assert sent[-1], "intent sends real overrides"
    for _ in range(30):
        b.release()
    assert len(sent) == 8 + 1 + 8, "fresh burst after overrides, then silent again"
    assert all(not o for o in sent[-8:]), "burst frames are all-zero (release)"


def test_hb_liveness_gate_withholds_heartbeats_when_loop_wedges():
    # The GCS heartbeat must report LOOP liveness: startup grace before the first
    # drain, flow while draining, withhold when the loop wedges (so FS_GCS can fire).
    b = _offline_guided_backend()
    now = time.monotonic()
    assert b._hb_should_send(now)                  # startup grace
    assert not b._hb_should_send(b._hb_open_t + 120.0)   # grace is BOUNDED: a process
    # that never reaches the main loop must not claim "healthy GCS" forever
    b._last_drain_t = now - 1.0
    assert b._hb_should_send(now)                  # loop alive
    b._last_drain_t = now - 10.0
    assert not b._hb_should_send(now)              # wedged -> withhold
    b._last_drain_t = now - 0.1
    assert b._hb_should_send(now)                  # resumed


def test_mode_command_cancelled_when_pilot_changes_mode():
    # Flight-2 fix: the pilot's TX mode switch must always win. While the companion is
    # retrying DO_SET_MODE (FC rejecting it), the FC moving to a THIRD mode (pilot or
    # failsafe) cancels the retry instead of fighting it every 0.5 s forever.
    backend, fake = _guided_backend(start_mode=0)
    fake.reject_mode = 20                                    # FC refuses GUIDED_NOGPS
    try:
        _pump(backend, fake, 0.5, lambda: backend._current_mode == 0)
        backend.set_engaged(True)
        _pump(backend, fake, 0.7)                            # retrying against the reject
        assert backend._target_mode == 20
        fake.custom_mode = 2                                 # pilot flips to ALT_HOLD
        ok = _pump(backend, fake, 2.0, lambda: backend._target_mode is None)
        assert ok, "pilot mode change must cancel the pending mode command"
        sent_at_cancel = len(fake.set_mode_cmds)
        _pump(backend, fake, 1.2)                            # > 2 resend intervals
        assert len(fake.set_mode_cmds) == sent_at_cancel     # no more DO_SET_MODE
        assert fake.custom_mode == 2                         # pilot's choice stands
    finally:
        backend.close(); fake.stop()


def test_mode_command_gives_up_after_retry_budget(monkeypatch):
    # Flight-2 fix: the retry loop is bounded — a permanently rejected mode stops
    # being re-commanded after _MODE_RETRY_BUDGET_S instead of forever.
    import pi_fpv_companion.fc.ardupilot as ap_mod
    monkeypatch.setattr(ap_mod, "_MODE_RETRY_BUDGET_S", 1.0)
    backend, fake = _guided_backend(start_mode=0)
    fake.reject_mode = 20                                    # FC refuses GUIDED_NOGPS
    try:
        _pump(backend, fake, 0.5, lambda: backend._current_mode == 0)
        backend.set_engaged(True)
        ok = _pump(backend, fake, 3.0, lambda: backend._target_mode is None)
        assert ok, "retry must give up after the budget"
        assert fake.set_mode_cmds, "it did try before giving up"
        assert fake.custom_mode == 0                         # FC was never moved
        assert backend._saved_mode is None                   # no stale restore left armed
    finally:
        backend.close(); fake.stop()


def test_gcs_heartbeat_flows_without_the_frame_loop(ap_pair):
    # Flight-2 fix: the ~1 Hz GCS heartbeat (FS_GCS liveness) runs on its own thread —
    # it must keep flowing even when nothing drives _drain() (camera stalled/booting).
    backend, fake = ap_pair
    before = len(fake.captured_heartbeats)
    time.sleep(2.5)                       # NO read_switch()/_drain() calls at all
    assert len(fake.captured_heartbeats) - before >= 2


def test_auto_guided_off_never_commands_mode():
    backend, fake = _guided_backend(auto_guided=False, start_mode=0)
    try:
        _pump(backend, fake, 0.5, lambda: backend._current_mode == 0)
        backend.set_engaged(True)
        _pump(backend, fake, 1.0)
        assert fake.set_mode_cmds == [] and fake.custom_mode == 0  # inert
    finally:
        backend.close(); fake.stop()


def test_ensure_param_bits_sets_bit_preserving_other_bits():
    # GUID_OPTIONS bit 3 (=8, ThrustAsThrust) must be OR-ed in without clobbering other
    # bits already set (e.g. bit 0 =1 AllowArmingFromTX). This is THE guided-throttle check.
    from pi_fpv_companion.fc.ardupilot import GUID_OPTIONS_THRUST_AS_THRUST as TAT
    backend, fake = _bits_backend_with_params({"GUID_OPTIONS": 1.0})  # bit0 set, bit3 not
    try:
        status = backend.ensure_param_bits("GUID_OPTIONS", TAT)
        assert status == "set"
        assert int(fake.params["GUID_OPTIONS"]) == (1 | TAT)  # bit0 preserved, bit3 added
    finally:
        backend.close(); fake.stop()


def test_ensure_param_bits_ok_when_already_set():
    from pi_fpv_companion.fc.ardupilot import GUID_OPTIONS_THRUST_AS_THRUST as TAT
    backend, fake = _bits_backend_with_params({"GUID_OPTIONS": float(TAT | 1)})
    try:
        status = backend.ensure_param_bits("GUID_OPTIONS", TAT)
        assert status == "ok"
        assert int(fake.params["GUID_OPTIONS"]) == (TAT | 1)  # untouched
    finally:
        backend.close(); fake.stop()
