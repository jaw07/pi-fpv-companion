"""Safety gate: decides whether the proposed intent should be sent to the FC.

Five independent gates must all pass:
  1. Pilot switch is active (read from FC RC channel)
  2. FC is armed (when require_armed is true)
  3. A current (filtered) target exists
  4. Its last update is within the watchdog window
  5. Track quality is above the floor — the "confidently wrong" mitigation
     (audit §5). The filter collapses quality on implausible jumps, class
     flips, and confidence decay; below the floor we will NOT command the
     aircraft toward what is probably a misdetection / drifted track.

When any gate fails, return ZERO_INTENT. The FC's own MAVLink/MSP failsafe
handles full UART silence; emitting zeros keeps the link alive and the FC
explicitly commanded to hold, rather than relying on timeout behavior. The
ultimate authority remains the pilot's flight-mode switch (audit §1).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from pi_fpv_companion.types import (
    FilteredTarget,
    GuidanceIntent,
    GuidanceMode,
    SwitchState,
    ZERO_INTENT,
)


@dataclass(frozen=True)
class SafetyConfig:
    watchdog_timeout_s: float
    require_armed: bool
    min_track_quality: float = 0.35   # below this, mute (probable wrong target)


@dataclass(frozen=True)
class GateResult:
    intent: GuidanceIntent
    muted: bool
    reason: str   # human-readable mute reason; empty when passing


def gate(
    proposed: GuidanceIntent,
    target: Optional[FilteredTarget],
    switch: SwitchState,
    armed: bool,
    now: float,
    cfg: SafetyConfig,
) -> GateResult:
    if switch.mode is GuidanceMode.STANDBY:
        return GateResult(ZERO_INTENT, True, "standby")
    if cfg.require_armed and not armed:
        return GateResult(ZERO_INTENT, True, "fc not armed")
    if target is None:
        return GateResult(ZERO_INTENT, True, "no target")
    if (now - target.timestamp) > cfg.watchdog_timeout_s:
        return GateResult(ZERO_INTENT, True, "target stale")
    if target.quality < cfg.min_track_quality:
        return GateResult(ZERO_INTENT, True, "low track quality")
    return GateResult(proposed, False, "")
