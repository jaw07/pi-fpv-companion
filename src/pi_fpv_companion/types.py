"""Core data types passed between pipeline layers."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class GuidanceMode(Enum):
    """Pilot-selected guidance mode from the 3-position RC switch.

    STANDBY: disengaged — nothing is sent to the FC.
    TRACK:   lock one target, yaw to keep it centred + hold range (pitch
             closure regulated to desired_bbox_frac). Follows; does not commit.
    DIVE:    commit — saturate forward (nose-down) lean to close the gap and
             dive onto the target.
    """
    STANDBY = "standby"
    TRACK = "track"
    DIVE = "dive"


@dataclass(frozen=True)
class Detection:
    """A single detection from either the on-Pi detector or the IMX500 sensor."""
    x: float           # bbox center x in pixels
    y: float           # bbox center y in pixels
    w: float           # bbox width in pixels
    h: float           # bbox height in pixels
    confidence: float
    class_id: int
    class_name: str = ""


@dataclass(frozen=True)
class Target:
    """The currently locked target as the tracker / associator sees it."""
    detection: Detection
    track_id: int
    lost_frames: int   # consecutive frames since last confirmed update
    timestamp: float   # monotonic seconds


@dataclass(frozen=True)
class GuidanceIntent:
    """Backend-agnostic command intent emitted by the visual servo.

    ATTITUDE domain — the universal control surface for a GPS-denied quad.
    A bare FPV quad has no EKF position/velocity estimate, so velocity
    commands are not serviceable (see docs/architecture-audit.md §1). Both
    backends consume this:
      - ArduPilot ALT_HOLD     -> RC_CHANNELS_OVERRIDE AETR sticks (lean angle
        + yaw rate + throttle/climb-rate), GPS-denied
      - Betaflight ANGLE mode  -> AETR sticks (angle = stick deflection)

    Sign conventions (MAVLink/aero body frame):
      roll_deg  : + = roll right.  0 for pure-pursuit (lateral via yaw).
      pitch_deg : + = nose UP (decelerate/back).  NEGATIVE = nose down =
                  accelerate FORWARD toward the target.
      yaw_rate_dps : + = yaw right (clockwise from above).
      thrust    : 0..1. 0.5 = neutral. With ArduPilot GUID_OPTIONS set so
                  thrust is interpreted as climb-rate, 0.5 = hold altitude.
                  v1 leaves vertical to the FC: always neutral.
    """
    roll_deg: float
    pitch_deg: float
    yaw_rate_dps: float
    thrust: float
    timestamp: float


@dataclass(frozen=True)
class FilteredTarget:
    """A tracker `Target` after alpha-beta filtering + quality assessment.

    This is what the visual servo and safety gate actually consume — never the
    raw tracker output. It carries an image-plane velocity estimate (for the
    servo's feedforward term, removing structural pursuit lag) and a `quality`
    score in [0, 1]. Quality collapses on the failure modes the raw tracker
    can't detect: implausible centroid jumps (misdetection), class flips
    (locked a person, now it's a chair), and confidence decay. The safety gate
    mutes guidance below a quality floor — the "confidently wrong" mitigation.
    """
    detection: Detection      # smoothed position / size
    track_id: int
    vx_px_s: float            # estimated image-plane velocity
    vy_px_s: float
    quality: float            # 0..1; below SafetyConfig.min_track_quality -> muted
    timestamp: float


@dataclass(frozen=True)
class SwitchState:
    """Pilot's guidance-mode switch as read from the FC RC channel.

    `mode` is the 3-position selection; `active` (== mode != STANDBY) is kept
    for the Betaflight 2-state engage path and convenience.
    """
    active: bool
    pwm_us: int
    timestamp: float
    mode: GuidanceMode = GuidanceMode.STANDBY


# Muted / no-guidance intent: wings level, no turn, neutral thrust (FC holds
# altitude). This is the belt; the real gate is the pilot's flight-mode switch
# (when not in GUIDED_NOGPS the FC ignores us entirely — audit §1).
HOVER_THRUST = 0.5
ZERO_INTENT = GuidanceIntent(0.0, 0.0, 0.0, HOVER_THRUST, 0.0)
