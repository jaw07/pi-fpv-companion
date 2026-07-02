"""Unit tests for the STANDBY safety-contract verifier (pure logic, no hardware)."""
from __future__ import annotations

from pi_fpv_companion.safety_contract import ContractChecker, ContractConfig

STANDBY = 1000
ENGAGED = 1500


def _ck(**kw):
    return ContractChecker(cfg=ContractConfig(**kw))


def test_clean_standby_passes():
    c = _ck()
    c.on_heartbeat(0.0, armed=False)
    c.on_rc_channels(0.0, STANDBY)
    # only zero-override "hand back" frames, within the burst budget
    for i in range(5):
        c.on_rc_override(i * 0.03, [0] * 8)
    assert c.passed, c.report()


def test_nonzero_override_in_standby_is_flagged():
    c = _ck()
    c.on_heartbeat(0.0, armed=True)
    c.on_rc_channels(0.0, STANDBY)
    c.on_rc_override(0.1, [1500, 1500, 1450, 1500, 0, 0, 0, 0])
    assert not c.passed
    assert any(v.kind == "STANDBY-no-override" for v in c.violations)


def test_attitude_target_in_standby_is_flagged():
    c = _ck()
    c.on_heartbeat(0.0, armed=True)
    c.on_rc_channels(0.0, STANDBY)
    c.on_attitude_target(0.1)
    assert any(v.kind == "STANDBY-no-attitude-target" for v in c.violations)


def test_attitude_target_while_disarmed_is_flagged_even_when_engaged():
    # Disarmed contract holds regardless of switch state (launch-at-arm guard).
    c = _ck()
    c.on_heartbeat(0.0, armed=False)
    c.on_rc_channels(0.0, ENGAGED)            # engaged, but DISARMED
    c.on_attitude_target(0.1)
    assert any(v.kind == "DISARMED-no-attitude-target" for v in c.violations)


def test_engaged_armed_attitude_target_is_allowed():
    # The mission: engaged + armed -> body rates are expected, NOT a violation.
    c = _ck()
    c.on_heartbeat(0.0, armed=True)
    c.on_rc_channels(0.0, ENGAGED)
    for i in range(30):
        c.on_attitude_target(i * 0.03)
    assert c.passed, c.report()


def test_override_burst_then_silence_passes_but_sustained_fails():
    c = _ck(release_burst_max=12)
    c.on_heartbeat(0.0, armed=True)
    c.on_rc_channels(0.0, STANDBY)
    for i in range(12):                        # the allowed hand-back burst
        c.on_rc_override(i * 0.03, [0] * 8)
    assert c.passed
    for i in range(12, 40):                    # but it must then go SILENT
        c.on_rc_override(i * 0.03, [0] * 8)
    assert not c.passed
    assert any(v.kind == "STANDBY-radio-silence" for v in c.violations)


def test_engaging_resets_the_standby_burst_counter():
    # A real STANDBY->engage->STANDBY cycle: the burst budget resets each STANDBY.
    c = _ck(release_burst_max=12)
    c.on_heartbeat(0.0, armed=True)
    c.on_rc_channels(0.0, STANDBY)
    for i in range(10):
        c.on_rc_override(i * 0.03, [0] * 8)
    c.on_rc_channels(1.0, ENGAGED)             # engage (resets counter)
    c.on_rc_channels(2.0, STANDBY)             # disengage
    for i in range(10):
        c.on_rc_override(2.0 + i * 0.03, [0] * 8)
    assert c.passed, c.report()


def test_mode_command_while_disarmed_standby_flagged():
    c = _ck()
    c.on_heartbeat(0.0, armed=False)
    for i in range(5):                          # settle in STANDBY (> standby_edge_frames)
        c.on_rc_channels(i * 0.1, STANDBY)
    c.on_set_mode(0.6, 20)
    assert any(v.kind == "STANDBY-no-mode-cmd" for v in c.violations)


def test_mode_command_while_ARMED_standby_flagged():
    # flight-3 regression: an ARMED in-flight DO_SET_MODE in settled STANDBY (the
    # auto_guided hijack) must FAIL — previously only the disarmed case was caught.
    c = _ck()
    c.on_heartbeat(0.0, armed=True)
    for i in range(5):
        c.on_rc_channels(i * 0.1, STANDBY)
    c.on_set_mode(0.6, 20)
    assert any(v.kind == "STANDBY-no-mode-cmd" for v in c.violations), c.report()


def test_mode_command_at_restore_edge_is_allowed():
    # A DO_SET_MODE within a frame or two of leaving engaged is the legit restore edge.
    c = _ck()
    c.on_heartbeat(0.0, armed=True)
    c.on_rc_channels(0.0, ENGAGED)              # engaged
    c.on_rc_channels(0.1, STANDBY)              # just disengaged -> restore edge
    c.on_set_mode(0.11, 0)                      # restore prior mode: allowed
    assert c.passed, c.report()
