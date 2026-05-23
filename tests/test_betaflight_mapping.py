from pi_fpv_companion.types import GuidanceIntent
from pi_fpv_companion.fc.betaflight import BetaflightMapping, intent_to_sticks


def _mapping():
    return BetaflightMapping(
        roll_us_per_deg=12.0,
        pitch_us_per_deg=-12.0,           # nose-down (neg pitch) -> elevator up; sign is TX-dependent
        yaw_us_per_dps=5.0,
        throttle_us_per_thrust=0.0,       # FC holds altitude
    )


def _intent(roll=0.0, pitch=0.0, yaw_rate=0.0, thrust=0.5):
    return GuidanceIntent(
        roll_deg=roll, pitch_deg=pitch, yaw_rate_dps=yaw_rate,
        thrust=thrust, timestamp=0.0,
    )


def test_neutral_intent_holds_all_sticks_centered():
    a, e, t, r = intent_to_sticks(_intent(), _mapping())
    assert (a, e, t, r) == (1500, 1500, 1500, 1500)


def test_positive_yaw_rate_moves_rudder():
    _, _, _, r = intent_to_sticks(_intent(yaw_rate=20.0), _mapping())
    assert r == 1500 + 5 * 20             # 1600


def test_forward_pitch_moves_elevator_per_sign_convention():
    # pitch_deg negative = nose-down = forward; with pitch_us_per_deg=-12,
    # elevator goes ABOVE 1500.
    _, e, _, _ = intent_to_sticks(_intent(pitch=-5.0), _mapping())
    assert e == 1500 + 60                  # -12 * -5 = +60


def test_roll_moves_aileron():
    a, _, _, _ = intent_to_sticks(_intent(roll=10.0), _mapping())
    assert a == 1500 + 12 * 10             # 1620


def test_thrust_neutral_keeps_throttle_centered_when_gain_zero():
    _, _, t, _ = intent_to_sticks(_intent(thrust=0.9), _mapping())
    assert t == 1500                       # throttle_us_per_thrust=0 -> FC holds alt


def test_thrust_maps_when_gain_set():
    m = BetaflightMapping(
        roll_us_per_deg=12.0, pitch_us_per_deg=-12.0, yaw_us_per_dps=5.0,
        throttle_us_per_thrust=400.0,
    )
    _, _, t, _ = intent_to_sticks(_intent(thrust=0.75), m)
    assert t == 1500 + int(400.0 * (0.75 - 0.5))   # +100 -> 1600


def test_extreme_intent_clamped_to_stick_range():
    a, e, t, r = intent_to_sticks(
        _intent(roll=1000.0, pitch=-1000.0, yaw_rate=10000.0), _mapping()
    )
    for v in (a, e, t, r):
        assert 1000 <= v <= 2000
    assert r == 2000
    assert a == 2000
    assert e == 2000                       # -12 * -1000 huge positive -> clamp hi
