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
import math
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
    pitch_p_gain: float = 0.15    # deg of pitch per px of VERTICAL error (DIVE aim)
    track_vcenter_gain: float = 0.10  # TRACK: deg of pitch per px of vertical error
                                  # (keeps target centred as lean tilts the camera; 0 = off)
    dive_forward_deg: float = 10.0  # constant forward (nose-down) lean while diving
    # Gravity dive (DIVE only): command a descent (thrust below neutral) to trade
    # altitude for speed, SCALED by how well-centred the target is — full commit
    # dead-centre, zero once it drifts past dive_center_frac. 0 = disabled
    # (altitude held). thrust maps to the throttle stick (STABILIZE: direct cut;
    # ALT_HOLD: climb-rate-down, capped by PILOT_SPEED_DN) — RC override, NO
    # GUID_OPTIONS. YOU own the altitude floor; the flight-mode switch is the
    # abort. Bench-validate before flight.
    dive_descent: float = 0.0       # thrust-down at full commit (0..0.5 below neutral)
    dive_center_frac: float = 0.30  # normalised centring error within which to commit power
    # Agnostic vertical framing (DIVE). A centred dive on a target BELOW us
    # pancakes: the line-of-sight depression sweeps down faster than we close, so
    # the target falls out the bottom (or we descend into the ground short of it).
    # Biasing the vertical setpoint toward the LEADING edge of the engagement
    # reserves frame for the LOS angle to grow into AND sustains the forward lean
    # that actually closes the gap. The bias DIRECTION is keyed on the target's
    # true line-of-sight elevation (aircraft pitch + in-frame elevation), NOT its
    # raw frame position — a ground target correctly framed high still reads
    # "below the horizon", so we keep diving instead of falsely flipping to climb.
    # This makes DIVE altitude-agnostic: dive onto a below target, pursue a level
    # one, climb toward an above one. 0 = legacy centred behaviour (camera_vfov
    # then unused). Needs camera_vfov_deg to convert vertical pixels<->angle.
    # Bench/SITL validate; steep engagements stay bounded by max_pitch_deg + VFoV.
    dive_vertical_bias_frac: float = 0.0   # bias setpoint this fraction of half-frame
    dive_los_band_deg: float = 8.0         # LOS-elev band over which the bias/commit ramps
    # VERTICAL field of view spanned by the frame height — the ONLY pixel<->angle
    # conversion the servo needs (the dive's LOS elevation). Default 52.3° is the
    # Raspberry Pi AI Camera (Sony IMX500) spec (HFoV 66.3°, VFoV 52.3°, full-FoV
    # sensor mode). Must match the actual capture's vertical FoV if a different
    # lens / crop is used.
    camera_vfov_deg: float = 52.3
    # DIVE is commit-only: cap the nose-UP authority so the forward lean dominates
    # and an above-target climb is driven by throttle, not a stall-inducing pitch-up.
    # None -> no extra cap (legacy: bounded only by max_pitch_deg).
    dive_pitch_up_max_deg: float | None = None
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
    aircraft_pitch_deg: float = 0.0,
) -> GuidanceIntent:
    """Map the filtered target's pixel state to an attitude intent.

    TRACK holds range (closure regulated to desired_bbox_frac); DIVE commits and
    saturates forward lean to close + dive. Yaw centring is identical in both.

    `aircraft_pitch_deg` (nose-up +, from FC ATTITUDE telemetry) is only used by
    the agnostic DIVE vertical framing (cfg.dive_vertical_bias_frac > 0): added to
    the in-frame elevation it gives the target's TRUE line-of-sight elevation, so
    the dive/climb decision tracks world geometry rather than frame position.
    Defaults to 0 (level) — with the bias off, the result is independent of it."""
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
        cy = cfg.frame_height / 2.0
        # --- Engagement direction from the target's TRUE line-of-sight elevation.
        # in-frame elevation (target above the boresight = +) + airframe pitch
        # (nose-up +) = LOS elevation above the horizon. Keying on this rather than
        # the raw frame position is what makes a ground target framed high still
        # read "below us" (keep diving) instead of flipping to "climb".
        #   s = +1 target below us (dive/descend) ... -1 above us (climb) ... 0 level
        fpx_v = (cfg.frame_height / 2.0) / math.tan(math.radians(cfg.camera_vfov_deg) / 2.0)
        frame_elev_deg = math.degrees(math.atan((cy - det.y) / fpx_v))
        los_elev_deg = frame_elev_deg + aircraft_pitch_deg
        s = _clamp(-los_elev_deg / cfg.dive_los_band_deg, -1.0, 1.0)

        # Vertical aim: bias the setpoint toward the LEADING edge of the dive (up
        # for a below target, down for an above one), so the target rides into the
        # frame the LOS angle sweeps through as we close — instead of pancaking out
        # the bottom. The bias also sustains the forward lean that actually closes.
        setpoint_y = cy - cfg.dive_vertical_bias_frac * (cfg.frame_height / 2.0) * s
        dy = _deadband(det.y - setpoint_y, cfg.pixel_deadzone_px)
        # DIVE is commit: cap nose-UP so forward lean dominates (closing an ABOVE
        # target is the throttle's job, not nose-up that would stall the approach).
        up = cfg.dive_pitch_up_max_deg if cfg.dive_pitch_up_max_deg is not None \
            else cfg.max_pitch_deg
        pitch = _clamp(
            -cfg.pitch_sign * cfg.pitch_p_gain * dy - cfg.dive_forward_deg,
            -cfg.max_pitch_deg, up,
        )

        # Vertical commit (gravity dive / powered climb), gated on HORIZONTAL aim
        # (centre yaw before committing power) and signed + ramped by the LOS
        # elevation: descend onto a below target, climb toward an above one, hold
        # for a level one.
        ex = (det.x - cx) / (cfg.frame_width / 2.0)   # +right
        commit = _clamp(1.0 - abs(ex) / cfg.dive_center_frac, 0.0, 1.0) \
            if cfg.dive_center_frac > 0 else 0.0
        thrust = _clamp(HOVER_THRUST - cfg.dive_descent * commit * s, 0.0, 1.0)
    else:
        # TRACK: range-hold (pitch ~ apparent-size error) PLUS a vertical-centring
        # term. The fixed camera tilts with the airframe, so leaning forward to
        # close makes the target rise in frame; the vcentre term pitches back up to
        # re-centre it (limiting the lean) so it stays in view. The two share the
        # pitch budget — closure dominates, vcentre keeps the target from drifting
        # out the top. (Accommodating a target at a different *altitude* is a
        # throttle job, not pitch — see docs camera-pitch-coupling.)
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
    )
