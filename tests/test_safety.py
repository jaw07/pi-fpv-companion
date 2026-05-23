from pi_fpv_companion.types import (
    Detection,
    FilteredTarget,
    GuidanceIntent,
    GuidanceMode,
    SwitchState,
    ZERO_INTENT,
)
from pi_fpv_companion.guidance.safety import SafetyConfig, gate


CFG = SafetyConfig(watchdog_timeout_s=0.25, require_armed=True, min_track_quality=0.35)
INTENT = GuidanceIntent(roll_deg=0.0, pitch_deg=-8.0, yaw_rate_dps=10.0, thrust=0.5, timestamp=1.0)
TARGET = FilteredTarget(
    detection=Detection(x=360, y=288, w=40, h=40, confidence=0.9, class_id=0),
    track_id=1,
    vx_px_s=0.0,
    vy_px_s=0.0,
    quality=0.9,
    timestamp=1.0,
)
LOWQ_TARGET = FilteredTarget(
    detection=Detection(x=360, y=288, w=40, h=40, confidence=0.2, class_id=0),
    track_id=1, vx_px_s=0.0, vy_px_s=0.0, quality=0.2, timestamp=1.0,
)
SWITCH_ON = SwitchState(active=True, pwm_us=1500, timestamp=1.0, mode=GuidanceMode.TRACK)
SWITCH_OFF = SwitchState(active=False, pwm_us=1000, timestamp=1.0, mode=GuidanceMode.STANDBY)


def test_all_gates_pass_returns_proposed_intent():
    r = gate(INTENT, TARGET, SWITCH_ON, armed=True, now=1.0, cfg=CFG)
    assert not r.muted
    assert r.intent == INTENT
    assert r.reason == ""


def test_standby_mutes():
    r = gate(INTENT, TARGET, SWITCH_OFF, armed=True, now=1.0, cfg=CFG)
    assert r.muted
    assert r.reason == "standby"
    assert r.intent == ZERO_INTENT


def test_disarmed_mutes_when_required():
    r = gate(INTENT, TARGET, SWITCH_ON, armed=False, now=1.0, cfg=CFG)
    assert r.muted
    assert r.reason == "fc not armed"


def test_disarmed_passes_when_not_required():
    cfg = SafetyConfig(watchdog_timeout_s=0.25, require_armed=False)
    r = gate(INTENT, TARGET, SWITCH_ON, armed=False, now=1.0, cfg=cfg)
    assert not r.muted


def test_no_target_mutes():
    r = gate(INTENT, None, SWITCH_ON, armed=True, now=1.0, cfg=CFG)
    assert r.muted
    assert r.reason == "no target"


def test_stale_target_mutes():
    r = gate(INTENT, TARGET, SWITCH_ON, armed=True, now=1.5, cfg=CFG)
    assert r.muted
    assert r.reason == "target stale"


def test_fresh_target_at_exact_window_boundary_still_passes():
    # now == target.ts + watchdog_timeout_s exactly -> still considered fresh
    r = gate(INTENT, TARGET, SWITCH_ON, armed=True, now=1.0 + CFG.watchdog_timeout_s, cfg=CFG)
    assert not r.muted


def test_standby_takes_precedence_over_other_failures():
    # When everything is bad, STANDBY should still be reported first
    stale_target_far_future = TARGET
    r = gate(INTENT, stale_target_far_future, SWITCH_OFF, armed=False, now=99.0, cfg=CFG)
    assert r.muted
    assert r.reason == "standby"


def test_low_track_quality_mutes():
    r = gate(INTENT, LOWQ_TARGET, SWITCH_ON, armed=True, now=1.0, cfg=CFG)
    assert r.muted
    assert r.reason == "low track quality"
    assert r.intent == ZERO_INTENT


def test_quality_at_floor_still_passes():
    floor_tgt = FilteredTarget(
        detection=Detection(x=360, y=288, w=40, h=40, confidence=0.35, class_id=0),
        track_id=1, vx_px_s=0.0, vy_px_s=0.0, quality=0.35, timestamp=1.0,
    )
    r = gate(INTENT, floor_tgt, SWITCH_ON, armed=True, now=1.0, cfg=CFG)
    assert not r.muted
