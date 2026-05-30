"""Bounding box + HUD drawing onto a BGR frame.

Pure cv2 calls. Mutates the image in place. Designed to be cheap — the same
overlay code runs on the Pi where every millisecond counts.
"""
from __future__ import annotations
from typing import Optional

import cv2

from pi_fpv_companion.guidance.safety import GateResult
from pi_fpv_companion.types import FilteredTarget, GuidanceIntent, GuidanceMode, SwitchState


_CROSSHAIR_LEN = 14
_CROSSHAIR_COLOR = (200, 200, 200)
_BBOX_ACTIVE = (0, 255, 0)
_BBOX_MUTED = (0, 165, 255)
_BBOX_LOST = (96, 96, 96)
_BBOX_CANDIDATE = (180, 180, 60)   # other selectable detections (multi-target)
_HUD_TEXT = (255, 255, 255)        # data text: white on a black outline (below)
_OUTLINE = (0, 0, 0)
# Mode status colours (BGR): standby grey, track green, dive red.
_STANDBY_COLOR = (200, 200, 200)
_TRACK_COLOR = (0, 255, 0)
_DIVE_COLOR = (0, 0, 255)
_HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
# Analog displays overscan — they crop ~5-8% off every edge — so keep all HUD
# elements inside a "safe area" inset by this fraction of each dimension. Bump it
# if your monitor still clips the corners; drop it if you have margin to spare.
_SAFE_FRAC = 0.06


def _put_text(image, text, org, color, scale=0.5, thickness=1) -> None:
    """Draw text with a black outline so it stays legible over any background.
    Plain coloured glyphs blend in — and thin strokes get averaged away when the
    preview is downscaled, so HUD callers bump scale/thickness to compensate."""
    cv2.putText(image, text, org, _HUD_FONT, scale, _OUTLINE, thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, org, _HUD_FONT, scale, color, thickness, cv2.LINE_AA)


def draw_overlay(
    image,
    target: Optional[FilteredTarget],
    intent: GuidanceIntent,
    switch: SwitchState,
    armed: bool,
    gated: GateResult,
    tracks=None,
) -> None:
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    mx, my = int(_SAFE_FRAC * w), int(_SAFE_FRAC * h)   # overscan-safe margins

    cv2.line(image, (cx - _CROSSHAIR_LEN, cy), (cx + _CROSSHAIR_LEN, cy), _CROSSHAIR_COLOR, 1)
    cv2.line(image, (cx, cy - _CROSSHAIR_LEN), (cx, cy + _CROSSHAIR_LEN), _CROSSHAIR_COLOR, 1)

    # Candidate targets (multi-target tracker): show every detection faintly so the
    # operator can see what's selectable; the locked one is drawn boldly below.
    if tracks:
        locked = target.track_id if target is not None else None
        for tr in tracks:
            if tr.track_id == locked:
                continue
            d = tr.detection
            cv2.rectangle(image, (int(d.x - d.w / 2), int(d.y - d.h / 2)),
                          (int(d.x + d.w / 2), int(d.y + d.h / 2)), _BBOX_CANDIDATE, 1)
            _put_text(image, f"id{tr.track_id}",
                      (int(d.x - d.w / 2), max(my, int(d.y - d.h / 2) - 4)), _BBOX_CANDIDATE)

    if target is not None:
        d = target.detection
        x1, y1 = int(d.x - d.w / 2), int(d.y - d.h / 2)
        x2, y2 = int(d.x + d.w / 2), int(d.y + d.h / 2)
        # Box colour reflects track quality (FilteredTarget) + gate state:
        # gray = degraded/coasting track, orange = muted, green = active.
        if target.quality < 0.35:
            color = _BBOX_LOST
        elif gated.muted:
            color = _BBOX_MUTED
        else:
            color = _BBOX_ACTIVE
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"id{target.track_id} q{target.quality:.2f}"
        # Keep the label inside the safe area even when the box is near an edge.
        _put_text(image, label, (max(mx, x1), max(my + 12, y1 - 6)), color)

    # Per architecture-audit §2: the FC draws the flight OSD (battery, attitude,
    # mode) — the Pi must NOT duplicate it. Keep this to target/track + guidance
    # engagement state only.
    mode = switch.mode
    if mode is GuidanceMode.STANDBY:
        status, scolor = "STANDBY", _STANDBY_COLOR
    else:
        scolor = _DIVE_COLOR if mode is GuidanceMode.DIVE else _TRACK_COLOR
        # Engaged but a safety gate is holding it (no target, stale, low quality…)
        status = mode.name if not gated.muted else f"{mode.name}  HOLD:{gated.reason}"
    line_h = 24
    y = h - my - line_h
    _put_text(image, status, (mx, y), scolor, scale=0.6, thickness=2)
    _put_text(
        image,
        f"yaw {intent.yaw_rate_dps:+5.0f}dps  pitch {intent.pitch_deg:+4.0f}deg",
        (mx, y + line_h), _HUD_TEXT, scale=0.6, thickness=2,
    )
