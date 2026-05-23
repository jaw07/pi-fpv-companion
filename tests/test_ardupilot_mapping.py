from pi_fpv_companion.types import HOVER_THRUST, GuidanceIntent
from pi_fpv_companion.fc.ardupilot import ArduCopterRcMapping, intent_to_rc_overrides

# Pin scale + control_mode so the math is deterministic regardless of defaults.
M = ArduCopterRcMapping(angle_max_deg=30.0, pilot_yaw_rate_dps=90.0, control_mode="althold")


def _intent(roll=0.0, pitch=0.0, yaw_rate=0.0, thrust=HOVER_THRUST):
    return GuidanceIntent(roll_deg=roll, pitch_deg=pitch, yaw_rate_dps=yaw_rate,
                          thrust=thrust, timestamp=0.0)


def test_level_hover_is_all_centered():
    o = intent_to_rc_overrides(_intent(), M)
    assert o == {1: 1500, 2: 1500, 3: 1500, 4: 1500}


def test_throttle_maps_thrust_about_center():
    assert intent_to_rc_overrides(_intent(thrust=0.5), M)[3] == 1500   # hold alt
    assert intent_to_rc_overrides(_intent(thrust=1.0), M)[3] == 2000   # full climb
    assert intent_to_rc_overrides(_intent(thrust=0.0), M)[3] == 1000   # full descend
    assert intent_to_rc_overrides(_intent(thrust=0.25), M)[3] == 1250  # DIVE descent


def test_roll_full_deflection_and_scale():
    assert intent_to_rc_overrides(_intent(roll=30.0), M)[1] == 2000    # full right
    assert intent_to_rc_overrides(_intent(roll=-30.0), M)[1] == 1000   # full left
    assert intent_to_rc_overrides(_intent(roll=15.0), M)[1] == 1750    # half


def test_pitch_full_deflection():
    # mapping math only (FC direction is validated in SITL): -angle_max -> min PWM.
    assert intent_to_rc_overrides(_intent(pitch=-30.0), M)[2] == 1000
    assert intent_to_rc_overrides(_intent(pitch=30.0), M)[2] == 2000


def test_yaw_rate_maps_to_full_stick_at_pilot_rate():
    assert intent_to_rc_overrides(_intent(yaw_rate=90.0), M)[4] == 2000
    assert intent_to_rc_overrides(_intent(yaw_rate=-90.0), M)[4] == 1000
    assert intent_to_rc_overrides(_intent(yaw_rate=45.0), M)[4] == 1750


def test_commands_clamp_beyond_full_deflection():
    assert intent_to_rc_overrides(_intent(roll=120.0), M)[1] == 2000   # clamped
    assert intent_to_rc_overrides(_intent(yaw_rate=-500.0), M)[4] == 1000
    assert intent_to_rc_overrides(_intent(thrust=5.0), M)[3] == 2000


def test_sign_flip_inverts_axis():
    m = ArduCopterRcMapping(angle_max_deg=30.0, pilot_yaw_rate_dps=90.0,
                            roll_sign=-1, pitch_sign=-1, yaw_sign=-1)
    assert intent_to_rc_overrides(_intent(roll=30.0), m)[1] == 1000     # flipped
    assert intent_to_rc_overrides(_intent(pitch=-30.0), m)[2] == 2000   # flipped
    assert intent_to_rc_overrides(_intent(yaw_rate=90.0), m)[4] == 1000  # flipped


def test_custom_channel_mapping():
    m = ArduCopterRcMapping(angle_max_deg=30.0, roll_channel=5, pitch_channel=6,
                            throttle_channel=7, yaw_channel=8)
    o = intent_to_rc_overrides(_intent(roll=30.0), m)
    assert set(o.keys()) == {5, 6, 7, 8}
    assert o[5] == 2000


# ---- STABILIZE control_mode: direct throttle centred on hover ----

S = ArduCopterRcMapping(control_mode="stabilize", hover_throttle_us=1450)


def test_stabilize_throttle_hovers_at_half_thrust():
    # thrust 0.5 -> hover throttle (not 1500); a true dive cuts power to min.
    assert intent_to_rc_overrides(_intent(thrust=0.5), S)[3] == 1450  # hover
    assert intent_to_rc_overrides(_intent(thrust=0.0), S)[3] == 1000  # full cut (dive)
    assert intent_to_rc_overrides(_intent(thrust=1.0), S)[3] == 2000  # full power
    # halfway between hover and min/max
    assert intent_to_rc_overrides(_intent(thrust=0.25), S)[3] == 1225  # 1450 - 0.5*(1450-1000)
    assert intent_to_rc_overrides(_intent(thrust=0.75), S)[3] == 1725  # 1450 + 0.5*(2000-1450)


def test_stabilize_leaves_roll_pitch_yaw_identical_to_althold():
    # only throttle differs between modes; lean/yaw mapping is the same.
    a = ArduCopterRcMapping(control_mode="althold")
    i = _intent(roll=20.0, pitch=-12.0, yaw_rate=60.0)
    oa, os = intent_to_rc_overrides(i, a), intent_to_rc_overrides(i, S)
    assert (oa[1], oa[2], oa[4]) == (os[1], os[2], os[4])
