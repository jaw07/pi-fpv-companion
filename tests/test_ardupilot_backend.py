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
