"""Image-based visual servoing: map a tracked target's pixel position into a
backend-agnostic ATTITUDE intent (the GPS-denied control surface).

  horizontal pixel error -> yaw RATE   (turn the nose toward the target)
  "approach"             -> forward PITCH (nose-down lean = accelerate at it)
  roll                   -> 0           (pure pursuit; lateral via yaw only)
  thrust                 -> neutral in TRACK (FC/adaptive-hover holds altitude);
                            in DIVE it moves altitude onto the target — see the
                            agnostic DIVE block below and docs/dive-guidance.md

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
    # Vertical centring (pitch P on the vertical pixel error, mirroring yaw on the
    # horizontal one). DIVE uses pitch_p_gain to aim the dive; TRACK adds a (usually
    # gentler) track_vcenter_gain ON TOP of range-hold so a forward lean — which
    # tilts the fixed camera down and makes the target rise in frame — is corrected
    # back toward centre instead of letting the target drift out the top.
    pitch_p_gain: float = 0.15    # deg of pitch per px of VERTICAL error (TRACK/DIVE aim)
    track_vcenter_gain: float = 0.10  # TRACK: deg of pitch per px of vertical error
                                  # (keeps target centred as lean tilts the camera; 0 = off)
    dive_forward_deg: float = 10.0  # forward (nose-down) commit lean while diving
    dive_center_frac: float = 0.30  # normalised horizontal aim error within which to commit vertical
    # --- DIVE closed-loop vertical (constant-bearing homing) ---------------------
    # The fixed camera couples pitch (forward closure) and vertical aim. To break
    # the coupling, DIVE uses PITCH for forward closure and the THROTTLE (a
    # commanded vertical RATE the backend tracks against VFR_HUD.climb) to hold the
    # target's vertical FRAME position. Holding a target at a fixed frame point is a
    # constant bearing → a collision course, so the flight path follows the line of
    # sight automatically — descend onto a target below, hold for one level ahead,
    # climb toward one above. No attitude/FoV needed; the frame error IS the signal.
    #   vertical_rate = -gain * (vertical frame error / half-frame), clamped, then
    #   gated on horizontal aim (centre yaw before committing power).
    # 0 gain = disabled (altitude held; DIVE just leans in). Bench/SITL-validate.
    dive_vrate_gain: float = 0.0       # m/s of climb command per unit normalised vert error
    dive_max_descent_mps: float = 8.0  # clamp on commanded descent (+ down)
    dive_max_climb_mps: float = 4.0    # clamp on commanded climb (gravity-limited, < descent)
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
    target: FilteredTarget, cfg: ServoConfig, mode: GuidanceMode = GuidanceMode.TRACK,
) -> GuidanceIntent:
    """Map the filtered target's pixel state to an attitude intent.

    TRACK holds range (closure regulated to desired_bbox_frac) at constant altitude.
    DIVE commits: PITCH leans forward to close, and a commanded vertical RATE
    (constant-bearing homing — the backend tracks it on VFR_HUD.climb) holds the
    target's vertical frame position so the flight path follows the line of sight,
    moving altitude onto the target whether it is below, level, or above. Yaw
    centring is identical in both."""
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
    vertical_rate_mps = None
    if mode is GuidanceMode.DIVE:
        cy = cfg.frame_height / 2.0
        # PITCH: a fixed, gentle forward (nose-down) commit lean — NOT vertical aim
        # (the throttle handles vertical via the rate loop below) and NOT closure-
        # regulated (a steep lean over-depresses the fixed camera and pushes an
        # ABOVE target out the top faster than the gravity-limited climb can
        # re-centre it). Gentle enough to keep any target framed while the vertical
        # rate loop flies the flight path onto it; clamped nose-down (commit).
        pitch = _clamp(-cfg.dive_forward_deg, -cfg.max_pitch_deg, 0.0)

        # VERTICAL: constant-bearing homing. Command a climb rate that drives the
        # target's vertical frame error to zero — holding it at a fixed frame point
        # is a constant bearing, i.e. a collision course, so the flight path tracks
        # the LOS for a target below / level / above. Below centre (drifted low) ->
        # descend (raises it back); above centre -> climb. Gated on horizontal aim
        # so we centre yaw before committing power. The backend tracks the rate on
        # VFR_HUD.climb (its P-loop's steady-state error is integrated out here).
        e_y = _deadband(det.y - cy, cfg.pixel_deadzone_px) / (cfg.frame_height / 2.0)
        ex = (det.x - cx) / (cfg.frame_width / 2.0)
        commit = _clamp(1.0 - abs(ex) / cfg.dive_center_frac, 0.0, 1.0) \
            if cfg.dive_center_frac > 0 else 0.0
        if cfg.dive_vrate_gain > 0.0:
            vertical_rate_mps = commit * _clamp(
                -cfg.dive_vrate_gain * e_y,
                -cfg.dive_max_descent_mps, cfg.dive_max_climb_mps,
            )
    else:
        # TRACK: range-hold (pitch ~ apparent-size error) PLUS a vertical-centring
        # term. The fixed camera tilts with the airframe, so leaning forward to
        # close makes the target rise in frame; the vcentre term pitches back up to
        # re-centre it (limiting the lean) so it stays in view. The two share the
        # pitch budget — closure dominates, vcentre keeps the target from drifting
        # out the top. (Accommodating a target at a different *altitude* is a
        # throttle job, not pitch — see docs/dive-guidance.md.)
        cy = cfg.frame_height / 2.0
        dy = _deadband(det.y - cy, cfg.pixel_deadzone_px)
        size_frac = (det.h / cfg.frame_height) if cfg.frame_height else 0.0
        size_err = size_frac - cfg.desired_bbox_frac
        pitch = _clamp(
            cfg.pitch_sign * cfg.closure_p_gain * size_err
            - cfg.pitch_sign * cfg.track_vcenter_gain * dy,
            -cfg.max_pitch_deg, cfg.max_pitch_deg,
        )

    return GuidanceIntent(
        roll_deg=0.0,
        pitch_deg=pitch,
        yaw_rate_dps=yaw_rate,
        thrust=thrust,
        timestamp=target.timestamp,
        vertical_rate_mps=vertical_rate_mps,
    )
