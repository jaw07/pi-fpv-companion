"""Wire-level proof of the STANDBY command contract, end to end.

The unit tests in test_safety_contract.py check the CHECKER; these check the
COMPANION — real Pipeline, real ArduPilotBackend, real MAVLink serialisation —
against a fake FC that can be armed and placed in an arbitrary flight mode. That
covers the states a disarmed bench FC with no TX bound can never reach, and it is
the regression net for the flight-3 class of failure ("pilot flew entirely in
STANDBY yet the controls were seized").

Slower than the rest of the suite (a few seconds each) because they run the real
loop in real time. That is the point: they exercise the threading, the debounce
and the release-burst timing, not a mocked approximation.
"""
from __future__ import annotations

import pytest

from tests.fakes.sil_wire_audit import run_scenario

pytestmark = pytest.mark.slow

STABILIZE, GUIDED_NOGPS = 0, 20
STANDBY_US, TRACK_US = 1000, 1500


def _assert_no_control_traffic(audit, why: str):
    assert audit.count("set_attitude_target") == 0, f"{why}: sent body rates"
    assert audit.count("rc_channels_override_nonzero") == 0, f"{why}: sent stick overrides"
    assert audit.count("do_set_mode") == 0, f"{why}: commanded a flight mode"
    assert audit.checker.passed, f"{why}: {audit.checker.report()}"


def test_standby_disarmed_touches_no_control_input():
    audit = run_scenario(armed=False, fc_mode=STABILIZE, ch7_us=STANDBY_US, seconds=2.0)
    _assert_no_control_traffic(audit, "STANDBY + disarmed")


def test_standby_while_ARMED_touches_no_control_input():
    """The flight-3 case: the pilot is flying manually with ch7 in STANDBY. The
    companion must be completely silent on every control surface."""
    audit = run_scenario(armed=True, fc_mode=STABILIZE, ch7_us=STANDBY_US, seconds=2.0)
    _assert_no_control_traffic(audit, "STANDBY + ARMED")


def test_standby_still_hands_back_the_sticks():
    """Silence must not mean "never released". The companion emits a SHORT burst of
    all-zero overrides (0 = "use the receiver") and then goes quiet — the handback is
    the first thing on the wire, and it is bounded."""
    audit = run_scenario(armed=True, fc_mode=STABILIZE, ch7_us=STANDBY_US, seconds=2.0)
    zero = audit.count("rc_channels_override_zero")
    assert 0 < zero <= 12, f"expected a bounded release burst, got {zero} frames"


def test_orphan_recovery_is_the_only_mode_command_in_standby():
    """The ONE documented exception: if a restart orphaned a prior engage — FC still in
    GUIDED_NOGPS, armed, and we never engaged this session — the companion commands the
    recover mode to hand the sticks back. It must fire EXACTLY once, command STABILIZE,
    and still touch no other control surface.

    Note this deliberately breaches "STANDBY commands no mode", which is why the
    contract checker flags it. If this test ever starts failing because the recovery
    stopped firing, a camera-watchdog restart mid-engage would leave the pilot with
    dead sticks."""
    audit = run_scenario(armed=True, fc_mode=GUIDED_NOGPS, ch7_us=STANDBY_US, seconds=2.5)
    assert audit.count("do_set_mode") == 1, (
        f"orphan recovery must fire exactly once, got {audit.count('do_set_mode')}")
    assert audit.count("set_attitude_target") == 0
    assert audit.count("rc_channels_override_nonzero") == 0
    # ...and it is the only thing the checker objects to.
    kinds = {v.kind for v in audit.checker.violations}
    assert kinds == {"STANDBY-no-mode-cmd"}, f"unexpected violations: {kinds}"


def test_orphan_recovery_does_not_fire_when_the_fc_is_not_in_our_mode():
    """Guard check: a normal armed STANDBY (FC in STABILIZE) is not an orphan, so
    nothing may be commanded. Without this, the recovery would be free to fire at any
    armed startup."""
    audit = run_scenario(armed=True, fc_mode=STABILIZE, ch7_us=STANDBY_US, seconds=2.0)
    assert audit.count("do_set_mode") == 0


def test_orphan_recovery_does_not_fire_while_disarmed():
    """A disarmed craft in GUIDED_NOGPS is on the bench, not orphaned mid-flight —
    commanding a mode there is uninvited traffic, not a rescue."""
    audit = run_scenario(armed=False, fc_mode=GUIDED_NOGPS, ch7_us=STANDBY_US, seconds=2.0)
    assert audit.count("do_set_mode") == 0
    _assert_no_control_traffic(audit, "disarmed in GUIDED_NOGPS")


@pytest.mark.parametrize("auto_guided", [True, False])
def test_engage_cycle_confines_control_traffic_to_the_engaged_window(auto_guided):
    """STANDBY -> TRACK -> STANDBY. Body rates may flow ONLY while engaged; the only
    control-affecting thing permitted while the switch reads STANDBY is the mode
    restore on the disengage edge."""
    audit = run_scenario(armed=True, fc_mode=STABILIZE, ch7_us=STANDBY_US,
                         engage_us=TRACK_US, seconds=4.5, auto_guided=auto_guided)
    standby_events = [e for e in audit.control_events() if e[1] == "STANDBY"]
    assert all(kind == "do_set_mode" for kind, _, _ in standby_events), (
        f"control traffic while STANDBY: "
        f"{sorted({k for k, _, _ in standby_events if k != 'do_set_mode'})}")
    assert audit.checker.passed, audit.checker.report()
    if auto_guided:
        # With auto-engage on, ch7 alone drives the FC into GUIDED_NOGPS and back.
        assert audit.count("do_set_mode") >= 1
        assert audit.count("set_attitude_target") > 0, "never actually engaged"
    else:
        # With it off the companion never commands a mode; the FC stays in STABILIZE,
        # so the control_ready interlock keeps it released the whole time.
        assert audit.count("do_set_mode") == 0
        assert audit.count("set_attitude_target") == 0
