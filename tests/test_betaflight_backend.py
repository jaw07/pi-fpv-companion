"""Integration tests for BetaflightBackend using a loopback serial fake.

Exercises the real MSP v1 wire protocol — encoded outbound frames, decoded
inbound responses, polling cadence, and the stick-encoding round-trip through
MSP_SET_RAW_RC.
"""
from __future__ import annotations
import time

import pytest

from pi_fpv_companion.fc.betaflight import BetaflightBackend, BetaflightMapping
from pi_fpv_companion.types import GuidanceIntent
from tests.fakes.fake_betaflight import FakeBetaflight, make_loopback_pair


def _mapping() -> BetaflightMapping:
    return BetaflightMapping(
        roll_us_per_deg=12.0,
        pitch_us_per_deg=-12.0,
        yaw_us_per_dps=5.0,
        throttle_us_per_thrust=0.0,
    )


@pytest.fixture
def bf_pair():
    backend_serial, fake_serial = make_loopback_pair()
    backend = BetaflightBackend(
        device="loopback",
        baud=115200,
        switch_channel=7,
        switch_threshold_us=1700,
        mapping=_mapping(),
        serial_factory=lambda: backend_serial,
    )
    backend.open()
    fake = FakeBetaflight(fake_serial)
    yield backend, fake
    backend.close()


def _exchange(backend, fake, rounds: int = 3) -> None:
    """Tick the loop a few times to let request/response settle past the poll-interval gate."""
    for _ in range(rounds):
        fake.pump()
        time.sleep(0.06)   # exceed _POLL_INTERVAL_S so the next poll fires


def test_read_switch_reflects_fc_channel_above_threshold(bf_pair):
    backend, fake = bf_pair
    fake.rc_channels[6] = 1850   # ch7
    # First call sends MSP_RC; fake hasn't pumped yet, no response cached.
    backend.read_switch()
    _exchange(backend, fake)
    s = backend.read_switch()
    assert s.pwm_us == 1850
    assert s.active is True


def test_read_switch_reflects_fc_channel_below_threshold(bf_pair):
    backend, fake = bf_pair
    fake.rc_channels[6] = 1100
    backend.read_switch()
    _exchange(backend, fake)
    s = backend.read_switch()
    assert s.pwm_us == 1100
    assert s.active is False


def test_is_armed_reflects_fc_state(bf_pair):
    backend, fake = bf_pair
    fake.armed = True
    backend.is_armed()
    _exchange(backend, fake)
    assert backend.is_armed() is True

    fake.armed = False
    backend.is_armed()
    _exchange(backend, fake)
    assert backend.is_armed() is False


def test_send_intent_emits_correct_aetr_sticks(bf_pair):
    backend, fake = bf_pair
    intent = GuidanceIntent(roll_deg=5.0, pitch_deg=-4.0, yaw_rate_dps=20.0,
                            thrust=0.5, timestamp=0.0)
    backend.send_intent(intent)
    fake.pump()

    assert len(fake.received_raw_rc) == 1
    a, e, t, r = fake.received_raw_rc[0]
    assert a == 1500 + int(12.0 * 5.0)      # roll 5deg -> 1560
    assert e == 1500 + int(-12.0 * -4.0)    # pitch -4deg -> 1548
    assert t == 1500                        # thrust gain 0 -> FC holds alt
    assert r == 1500 + 5 * 20               # 1600


def test_send_intent_clamps_to_stick_range(bf_pair):
    backend, fake = bf_pair
    intent = GuidanceIntent(roll_deg=1000.0, pitch_deg=-1000.0,
                            yaw_rate_dps=10000.0, thrust=0.5, timestamp=0.0)
    backend.send_intent(intent)
    fake.pump()

    a, e, t, r = fake.received_raw_rc[0]
    for v in (a, e, t, r):
        assert 1000 <= v <= 2000
    assert r == 2000          # yaw saturated high
    assert a == 2000          # roll saturated high
    assert e == 2000          # pitch -1000 * -12 -> saturated high
    assert t == 1500          # thrust gain 0 -> centered
