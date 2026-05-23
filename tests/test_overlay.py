"""Smoke tests for the overlay drawer. We don't pixel-perfect-compare; we just
verify the function runs against a real image without errors and modifies it."""
import numpy as np

from pi_fpv_companion.guidance.safety import GateResult
from pi_fpv_companion.types import (
    Detection,
    FilteredTarget,
    GuidanceIntent,
    SwitchState,
    ZERO_INTENT,
)
from pi_fpv_companion.video.overlay import draw_overlay


def _bundle_inputs(muted=False, has_target=True):
    img = np.full((576, 720, 3), 32, dtype=np.uint8)
    target = None
    if has_target:
        # The overlay receives the FilteredTarget the pipeline produces — not
        # the raw tracker Target. (This mismatch shipped a runtime AttributeError
        # past the suite once; the test now uses the real type.)
        target = FilteredTarget(
            detection=Detection(x=400, y=300, w=60, h=60, confidence=0.9, class_id=0, class_name="t"),
            track_id=1, vx_px_s=0.0, vy_px_s=0.0, quality=0.9, timestamp=0.0,
        )
    intent = GuidanceIntent(roll_deg=0.0, pitch_deg=-8.0, yaw_rate_dps=5.0, thrust=0.5, timestamp=0.0)
    switch = SwitchState(active=not muted, pwm_us=1800, timestamp=0.0)
    gated = GateResult(
        intent=ZERO_INTENT if muted else intent,
        muted=muted,
        reason="switch off" if muted else "",
    )
    return img, target, intent, switch, gated


def test_overlay_runs_with_target_and_active():
    img, target, intent, switch, gated = _bundle_inputs(muted=False, has_target=True)
    before = img.copy()
    draw_overlay(img, target, intent, switch, armed=True, gated=gated)
    assert not np.array_equal(img, before)         # something was drawn


def test_overlay_runs_with_muted_and_no_target():
    img, _, intent, switch, gated = _bundle_inputs(muted=True, has_target=False)
    before = img.copy()
    draw_overlay(img, None, intent, switch, armed=False, gated=gated)
    assert not np.array_equal(img, before)         # HUD still draws even without target


def test_overlay_draws_bbox_near_target():
    img, target, intent, switch, gated = _bundle_inputs(muted=False, has_target=True)
    draw_overlay(img, target, intent, switch, armed=True, gated=gated)
    # The bbox rectangle is drawn around (400, 300) with size 60. There should be
    # at least one non-background-color pixel on the rectangle perimeter.
    bg = (32, 32, 32)
    perim_pixels = [
        tuple(img[300 - 30, x]) for x in range(370, 430)
    ]
    assert any(p != bg for p in perim_pixels)
