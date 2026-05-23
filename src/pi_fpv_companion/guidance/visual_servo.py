"""Image-based visual servoing: map a tracked target's pixel position into a
backend-agnostic ATTITUDE intent (the GPS-denied control surface).

  horizontal pixel error -> yaw RATE   (turn the nose toward the target)
  "approach"             -> forward PITCH (nose-down lean = accelerate at it)
  roll                   -> 0           (pure pursuit; lateral via yaw only)
  thrust                 -> neutral     (FC holds altitude in v1)

Yaw is P + velocity FEEDFORWARD (audit §4): pure-P against a moving target
leaves a structural steady-state lag (the target sits permanently off-centre,
biased in its direction of travel). The feedforward term, fed by the
alpha-beta filter's image-plane velocity estimate, cancels that lag.

Approach is CLOSURE-REGULATED (audit §4): the forward (nose-down) lean is
proportional to how much smaller the target's apparent size is than the
desired hold size. Far target (small bbox) -> full forward lean (saturated);
as it grows to the hold size -> lean eases to zero; if it overshoots (too
close) -> nose-up to back off. This replaces the old constant-forward-velocity
behavior that drove the aircraft into the subject at constant speed.

Consumes a `FilteredTarget` (never the raw tracker output) — its bbox size is
the alpha-beta-smoothed value, so closure isn't chattering on raw detection
size noise. See track/target_filter.py / audit §5.
"""
from __future__ import annotations
from dataclasses import dataclass

from pi_fpv_companion.types import HOVER_THRUST, FilteredTarget, GuidanceIntent, GuidanceMode


@dataclass(frozen=True)
class ServoConfig:
    frame_width: int
    frame_height: int
    max_yaw_rate_dps: float
    max_pitch_deg: float          # pitch clamp, both directions (approach + back-off ceiling)
    pixel_deadzone_px: float
    yaw_p_gain: float             # deg/s of yaw rate per pixel of horizontal error
    yaw_ff_gain: float            # deg/s of yaw rate per (px/s) of target image vx
    desired_bbox_frac: float      # target bbox-height / frame-height at the hold distance
    closure_p_gain: float         # deg of pitch per unit of (size_frac - desired) error
    # DIVE-only: keep the target centred vertically (pitch P on the vertical pixel
    # error, mirroring yaw on the horizontal one) AND lean forward to close.
    # TRACK ignores both and uses the range-hold closure above instead.
    pitch_p_gain: float = 0.15    # deg of pitch per pixel of VERTICAL error (DIVE aim)
    dive_forward_deg: float = 10.0  # constant forward (nose-down) lean while diving
    # Gravity dive (DIVE only): command a descent (climb-rate below neutral) to
    # trade altitude for speed, SCALED by how well-centred the target is — full
    # commit dead-centre, zero once it drifts past dive_center_frac. 0 = disabled
    # (altitude held). Needs FC GUID_OPTIONS=climb-rate + an altitude floor YOU
    # manage; the flight-mode switch is the abort. Bench-validate before flight.
    dive_descent: float = 0.0       # climb-rate-down at full commit (0..0.5 below neutral)
    dive_center_frac: float = 0.30  # normalised centring error within which to commit power
    # Operator-correctable sign overrides (audit §6). A mirrored/flipped camera
    # inverts the error->command sign -> divergent positive feedback ("spins
    # away from target"). MUST be bench-validated (docs/deployment-safety.md §4).
    yaw_sign: float = 1.0         # set -1.0 if the bench self-test shows inversion
    pitch_sign: float = 1.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _deadband(v: float, dz: float) -> float:
    return 0.0 if abs(v) < dz else v


def compute_intent(
    target: FilteredTarget, cfg: ServoConfig, mode: GuidanceMode = GuidanceMode.TRACK
) -> GuidanceIntent:
    """Map the filtered target's pixel state to an attitude intent.

    TRACK holds range (closure regulated to desired_bbox_frac); DIVE commits and
    saturates forward lean to close + dive. Yaw centring is identical in both."""
    cx = cfg.frame_width / 2.0
    det = target.detection

    # Horizontal: P on the centring error + feedforward on the target's image
    # velocity. P alone leaves a structural lag against a moving target; the
    # feedforward (target moving right -> pre-emptively yaw right) cancels it.
    dx = _deadband(det.x - cx, cfg.pixel_deadzone_px)
    yaw_rate = _clamp(
        cfg.yaw_sign * (cfg.yaw_p_gain * dx + cfg.yaw_ff_gain * target.vx_px_s),
        -cfg.max_yaw_rate_dps, cfg.max_yaw_rate_dps,
    )

    # Closure regulation: lean proportional to the apparent-size error.
    #   size_frac  = bbox height / frame height (a monotone range proxy)
    #   size_err   = size_frac - desired   (negative when far, ~0 at hold dist,
    #                positive when too close)
    #   pitch      = closure_p_gain * size_err
    #     far  -> negative pitch (nose down, accelerate forward)
    #     hold -> ~0
    #     too close -> positive pitch (nose up, back off) — collision guard
    thrust = HOVER_THRUST
    if mode is GuidanceMode.DIVE:
        # Commit, but KEEP IT CENTRED: pitch tracks the vertical pixel error
        # (target low in frame -> nose down, high -> nose up) exactly as yaw
        # tracks the horizontal one, plus a constant forward lean so it closes.
        # Bbox size / range-hold is ignored — this is the dive.
        cy = cfg.frame_height / 2.0
        dy = _deadband(det.y - cy, cfg.pixel_deadzone_px)
        pitch = _clamp(
            -cfg.pitch_sign * cfg.pitch_p_gain * dy - cfg.dive_forward_deg,
            -cfg.max_pitch_deg, cfg.max_pitch_deg,
        )
        # Gravity dive: trade altitude for speed. The target is normally BELOW us
        # (we attack from above) so "low in frame" is the expected, wanted state —
        # pitch aims down at it and we descend toward it. So gate the descent on
        # HORIZONTAL aim (don't dive sideways past it — let yaw centre first), and
        # only ease it off if the target is actually ABOVE centre (above us), where
        # descending would drop away from it.
        ex = (det.x - cx) / (cfg.frame_width / 2.0)   # +right
        ey = (det.y - cy) / (cfg.frame_height / 2.0)  # +below us, -above us
        if cfg.dive_center_frac > 0:
            commit = _clamp(1.0 - abs(ex) / cfg.dive_center_frac, 0.0, 1.0)
            if ey < 0.0:   # target above us -> taper descent to zero
                commit *= _clamp(1.0 + ey / cfg.dive_center_frac, 0.0, 1.0)
        else:
            commit = 0.0
        thrust = _clamp(HOVER_THRUST - cfg.dive_descent * commit, 0.0, 1.0)
    else:
        size_frac = (det.h / cfg.frame_height) if cfg.frame_height else 0.0
        size_err = size_frac - cfg.desired_bbox_frac
        pitch = _clamp(
            cfg.pitch_sign * cfg.closure_p_gain * size_err,
            -cfg.max_pitch_deg, cfg.max_pitch_deg,
        )

    return GuidanceIntent(
        roll_deg=0.0,
        pitch_deg=pitch,
        yaw_rate_dps=yaw_rate,
        thrust=thrust,
        timestamp=target.timestamp,
    )
