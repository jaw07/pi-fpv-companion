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


def test_backend_reads_switch_channel_pwm(ap_pair):
    backend, fake = ap_pair

    fake.rc_channels[6] = 1800       # ch7 (0-indexed 6), >= dive threshold
    time.sleep(0.25)
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
