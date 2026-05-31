"""Framing controller — an attitude/thrust visual-servo for the GUIDED_NOGPS path.

Where visual_servo.py drives STABILIZE via RC-stick angle overrides (pitch=forward
lean, throttle=climb-rate), this drives GUIDED_NOGPS via SET_ATTITUDE_TARGET
(attitude + thrust). The control law is the proven quad-interceptor shape:

  * PITCH frames the target — a PID drives the target's vertical frame position to a
    goal NEAR THE TOP (vert_goal), so the nose pitches down onto a below target (which
    also carries the aircraft forward), keeping it framed all the way in.
  * THRUST descends — a PID on the target's TRUE angle below the horizon (in-frame
    elevation + measured airframe pitch + camera mount tilt); below the horizon ->
    thrust < hover -> descend onto it; level -> hover; above -> climb.
  * YAW/ROLL blend centres it horizontally — yaw turns toward a far-off-axis target,
    roll banks to slide onto a near one.

Outputs a GuidanceIntent with pitch_deg/roll_deg as ATTITUDE and thrust as a [0,1]
throttle (vertical_rate_mps stays None — the descent is the thrust, not a rate). The
closed-loop sim models exactly this (pitch->forward accel, thrust->vertical), so the
law tunes in the sim before Gazebo; the GUIDED_NOGPS node sends it as SET_ATTITUDE_TARGET.
"""
from __future__ import annotations
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from pi_fpv_companion.types import FilteredTarget, GuidanceIntent, GuidanceMode

HOVER_THRUST = 0.5


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


@dataclass
class PID:
    """PID with a windowed-average derivative (less noise than a single-step diff) and
    an integral clamp. Errors and output are plain floats in caller-chosen units."""
    kp: float
    ki: float = 0.0
    kd: float = 0.0
    out_limit: float = 1e9
    i_limit: float = 1e9
    deriv_window: int = 5
    _i: float = 0.0
    _hist: deque = field(default_factory=lambda: deque(maxlen=8))

    def reset(self) -> None:
        self._i = 0.0
        self._hist.clear()

    def update(self, error: float, dt: float) -> float:
        self._i = _clamp(self._i + error * dt, -self.i_limit, self.i_limit)
        self._hist.append((error, dt))
        d = 0.0
        if self.kd != 0.0 and len(self._hist) >= 2:
            # average slope over the window (windowed derivative)
            e0, _ = self._hist[0]
            span = sum(h[1] for h in list(self._hist)[1:])
            if span > 0:
                d = (error - e0) / span
        return _clamp(self.kp * error + self.ki * self._i + self.kd * d,
                      -self.out_limit, self.out_limit)


@dataclass(frozen=True)
class FramingConfig:
    frame_width: int
    frame_height: int
    hfov_deg: float = 66.3
    vfov_deg: float = 52.3
    camera_pitch_deg: float = 0.0          # fixed mount tilt (0 = bore-sight level with the airframe)
    hori_goal: float = 0.5                 # normalised x goal (centre)
    vert_goal: float = 0.15                # normalised y goal — NEAR THE TOP (quad: maximise forward thrust)
    # PITCH framing PID (deg of attitude per rad of vertical angle error)
    pitch_kp: float = 35.0
    pitch_ki: float = 2.0
    pitch_kd: float = 6.0
    max_pitch_deg: float = 30.0
    # THRUST descent PID (throttle offset per rad of true angle-to-target)
    thrust_kp: float = 0.6
    thrust_ki: float = 0.05
    thrust_kd: float = 0.1
    max_thrust_off: float = 0.4            # thrust stays in [0.5-off, 0.5+off]
    descent_pitch_fold: float = 1.0        # fraction of measured pitch folded into the descent angle.
                                           # The framing pitch noses down EXTRA to hold the target near
                                           # the top (vert_goal), so it overstates the depression — at
                                           # 1.0 the descent over-counts that and drops vertically (lands
                                           # short / loses a far target). <1 keeps the descent shallow
                                           # enough to glide in. 0 = descend on in-frame elevation only.
    # YAW (deg/s per rad horizontal error) and ROLL (deg bank per rad horizontal error)
    yaw_kp: float = 90.0
    max_yaw_rate_dps: float = 90.0
    roll_kp: float = 40.0
    roll_kd: float = 8.0
    max_roll_deg: float = 25.0
    yaw_roll_blend_rad: float = 0.20       # |horiz err| at/above which it's pure yaw; below, roll fades in
    pixel_deadzone_px: float = 6.0


@dataclass
class FramingState:
    """Per-engagement PID state (owned by the caller; reset on a new lock / mode change)."""
    pitch_pid: Optional[PID] = None
    thrust_pid: Optional[PID] = None
    roll_pid: Optional[PID] = None
    last_t: Optional[float] = None

    def ensure(self, cfg: FramingConfig) -> None:
        if self.pitch_pid is None:
            self.pitch_pid = PID(cfg.pitch_kp, cfg.pitch_ki, cfg.pitch_kd, cfg.max_pitch_deg, i_limit=0.5)
            self.thrust_pid = PID(cfg.thrust_kp, cfg.thrust_ki, cfg.thrust_kd, cfg.max_thrust_off, i_limit=2.0)
            self.roll_pid = PID(cfg.roll_kp, 0.0, cfg.roll_kd, cfg.max_roll_deg)

    def reset(self) -> None:
        for p in (self.pitch_pid, self.thrust_pid, self.roll_pid):
            if p is not None:
                p.reset()
        self.last_t = None


def compute_framing_intent(target: FilteredTarget, cfg: FramingConfig, state: FramingState,
                           now: float, pitch_deg_measured: float = 0.0,
                           mode: GuidanceMode = GuidanceMode.DIVE) -> GuidanceIntent:
    state.ensure(cfg)
    dt = _clamp((now - state.last_t) if state.last_t is not None else 0.0, 0.0, 0.2)
    state.last_t = now
    det = target.detection
    W, H = cfg.frame_width, cfg.frame_height
    vfov = math.radians(cfg.vfov_deg)
    hfov = math.radians(cfg.hfov_deg)
    cx_n, cy_n = det.x / W, det.y / H                      # normalised [0,1], 0 = top/left

    # HORIZONTAL: angle error (rad), + = target to the right of goal.
    dx_px = det.x - cfg.hori_goal * W
    if abs(dx_px) < cfg.pixel_deadzone_px:
        dx_px = 0.0
    horiz_err = (dx_px / W) * hfov

    # PITCH framing: drive the target's vertical position to vert_goal (near the top).
    # vert_goal - cy_n < 0 for a target below the goal -> nose DOWN (camera looks down,
    # the low target rises toward the goal) and that nose-down carries the quad forward.
    vert_err = (cfg.vert_goal - cy_n) * vfov
    pitch_deg = state.pitch_pid.update(vert_err, dt)
    pitch_deg = _clamp(pitch_deg, -cfg.max_pitch_deg, cfg.max_pitch_deg)

    # THRUST descent: PID on the target's TRUE angle below the horizon (in-frame
    # elevation from CENTRE + measured airframe pitch + fixed camera tilt). Below the
    # horizon (negative) -> thrust below hover -> descend onto it.
    ang_to_target = (0.5 - cy_n) * vfov + cfg.descent_pitch_fold * math.radians(pitch_deg_measured + cfg.camera_pitch_deg)
    thrust = HOVER_THRUST + state.thrust_pid.update(ang_to_target, dt)
    thrust = _clamp(thrust, HOVER_THRUST - cfg.max_thrust_off, HOVER_THRUST + cfg.max_thrust_off)

    # YAW/ROLL blend: yaw to turn toward a far target, roll to bank onto a near one.
    if cfg.yaw_roll_blend_rad > 0:
        alpha = _clamp(abs(horiz_err) / cfg.yaw_roll_blend_rad, 0.0, 1.0)
    else:
        alpha = 1.0
    yaw_rate = _clamp(cfg.yaw_kp * horiz_err * alpha, -cfg.max_yaw_rate_dps, cfg.max_yaw_rate_dps)
    roll_deg = _clamp(state.roll_pid.update(horiz_err, dt) * (1.0 - alpha),
                      -cfg.max_roll_deg, cfg.max_roll_deg)

    return GuidanceIntent(
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        yaw_rate_dps=yaw_rate,
        thrust=thrust,
        timestamp=target.timestamp,
        vertical_rate_mps=None,
    )
