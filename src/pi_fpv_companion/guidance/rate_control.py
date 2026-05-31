"""Rate-control visual servo for the GUIDED_NOGPS **body-rate** path (the reference-quad
interceptor surface). This is the production port of the SITL-validated rate law.

Where visual_servo.py drives STABILIZE via RC-stick angle overrides, this commands BODY
RATES + thrust (the backend sends them as SET_ATTITUDE_TARGET, mask 0b10000000, identity
quaternion). Rates are integrated by the airframe, so a noisy detector box yields SMOOTH
motion — an absolute-attitude quaternion snaps to each frame and jitters.

Control law (validated in Gazebo/ArduCopter SITL):
  * PITCH rate frames the target vertically to vert_goal; clamped at an attitude limit.
  * THRUST = PURSUIT: hover + PID on (line-of-sight-below-horizon - flight-path-angle), so
    the velocity vector points AT the target -> a straight-line dive whose angle == the
    target's depression. (Needs real throttle: the backend must set GUID_OPTIONS bit 3,
    else the FC reads thrust as a climb-rate and the dive planes.)
  * YAW centres the target horizontally (smooth); ROLL banks near centre with a gentle
    return-to-level. A DEADZONE zeros both for a near-centred target (kills the camera-pan
    "shaking" from sub-degree box noise); a SIZE-gain reduces horizontal authority when the
    target is small/far (noisy box).
  * vert_goal is near CENTRE (~0.4), not the top: a quad must fly INTO a ground target, not
    nose over and skim under it (the reference's 0.15/top is for chasing a forward AIR target).
  * IMPACT latch: target lost near the ground -> stop (level, cut throttle) for good.

Hover thrust is learned in flight (TWR-independent) by the caller and passed in via
RateState.hover; this module just trims throttle about it.
"""
from __future__ import annotations
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from pi_fpv_companion.types import FilteredTarget


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


@dataclass
class PID:
    """PID with a windowed-average derivative (less noise than a single-step diff) and an
    integral clamp. kp is settable so the caller can blend it (yaw/roll authority)."""
    kp: float
    ki: float = 0.0
    kd: float = 0.0
    out_limit: float = 1e9
    i_limit: float = 1e9
    _i: float = 0.0
    _hist: deque = field(default_factory=lambda: deque(maxlen=5))

    def reset(self) -> None:
        self._i = 0.0
        self._hist.clear()

    def update(self, error: float, dt: float) -> float:
        self._i = _clamp(self._i + error * dt, -self.i_limit, self.i_limit)
        self._hist.append(error)
        d = 0.0
        if self.kd != 0.0 and len(self._hist) > 1 and dt > 0:
            d = (self._hist[-1] - self._hist[0]) / (dt * (len(self._hist) - 1))
        return _clamp(self.kp * error + self.ki * self._i + self.kd * d,
                      -self.out_limit, self.out_limit)


@dataclass(frozen=True)
class RateConfig:
    frame_width: int
    frame_height: int
    hfov_deg: float = 66.3
    vfov_deg: float = 52.3
    camera_pitch_deg: float = 0.0          # fixed mount tilt (0 = bore-sight level)
    hori_goal: float = 0.5
    vert_goal: float = 0.40                # CENTRE (on the velocity vector): fly INTO a ground target
    # PITCH rate PID (rad/s per rad of vertical angle error)
    pitch_kp: float = 1.0
    pitch_ki: float = 0.0
    pitch_kd: float = 0.08                 # low: derivative on the noisy box was a shake source
    max_pitch_rad: float = 0.70            # ~40deg attitude limit on the pitch rate
    # THRUST pursuit PID about learned hover (out spans the throttle band)
    thrust_kp: float = 0.85
    thrust_ki: float = 0.30                # strong: drives velocity onto LOS despite hover error
    thrust_kd: float = 0.10
    thrust_out: float = 0.30
    thrust_ilim: float = 0.30
    aim_bias_rad: float = -0.06            # aim the velocity vector slightly INTO the target
    # YAW/ROLL (rad/s per rad horizontal error); yaw centres, roll banks gently
    yaw_kp: float = 2.0
    roll_kp: float = 2.5
    yaw_kd: float = 0.04
    roll_kd: float = 0.04
    roll_return: float = 4.0               # rad/s per rad of measured roll -> return to level
    horiz_deadzone_rad: float = 0.030      # |err| below this -> zero yaw/roll (anti pan-shake)
    horiz_thresh: float = 0.05             # yaw/roll blend knee
    max_horiz_err: float = 0.4
    # Differential low-pass on commanded rates (yaw heaviest: it was the oscillation)
    ema_pitch: float = 0.22
    ema_yaw: float = 0.18
    ema_roll: float = 0.35
    ema_thrust: float = 0.30
    # Acquisition / impact
    search_pitch_rad: float = -0.436       # ~-25deg: nose down to bring a below target into the FOV
    impact_agl_m: float = 12.0             # target lost below this AGL -> latch STOP


@dataclass
class RateState:
    """Per-engagement state (owned by the caller; reset on a new lock / mode change)."""
    pitch_pid: Optional[PID] = None
    thrust_pid: Optional[PID] = None
    yaw_pid: Optional[PID] = None
    roll_pid: Optional[PID] = None
    hover: float = 0.30                    # learned hover thrust (set by the caller)
    impacted: bool = False
    sm_pr: float = 0.0
    sm_yr: float = 0.0
    sm_rr: float = 0.0
    sm_thr: float = 0.30
    last_t: Optional[float] = None

    def ensure(self, cfg: RateConfig) -> None:
        if self.pitch_pid is None:
            self.pitch_pid = PID(cfg.pitch_kp, cfg.pitch_ki, cfg.pitch_kd, out_limit=math.pi / 2)
            self.thrust_pid = PID(cfg.thrust_kp, cfg.thrust_ki, cfg.thrust_kd,
                                  out_limit=cfg.thrust_out, i_limit=cfg.thrust_ilim)
            self.yaw_pid = PID(cfg.yaw_kp, 0.0, cfg.yaw_kd, out_limit=math.pi / 2, i_limit=0.5)
            self.roll_pid = PID(cfg.roll_kp, 0.01, cfg.roll_kd, out_limit=math.pi / 2, i_limit=0.5)

    def reset(self) -> None:
        for p in (self.pitch_pid, self.thrust_pid, self.yaw_pid, self.roll_pid):
            if p is not None:
                p.reset()
        self.impacted = False
        self.sm_pr = self.sm_yr = self.sm_rr = 0.0
        self.sm_thr = self.hover
        self.last_t = None


@dataclass(frozen=True)
class RateIntent:
    """Body-rate command for the guided_nogps surface (consumed by send_body_rates)."""
    roll_rate: float       # rad/s, + = roll right
    pitch_rate: float      # rad/s, + = nose up
    yaw_rate: float        # rad/s, + = yaw right
    thrust: float          # 0..1 (real throttle; 0.5 != hover on a high-TWR quad)
    phase: str             # "RATE" | "SEARCH" | "STOP"


def _preprocess(cfg: RateConfig, cx_n: float, cy_n: float, roll_rad: float):
    """De-rotate the target pixel about frame centre by the airframe roll, widen the FOV for
    that roll, and return (horiz_err, vert_err, ang_to_target) in radians."""
    hfov, vfov = math.radians(cfg.hfov_deg), math.radians(cfg.vfov_deg)
    dx, dy = cx_n - 0.5, cy_n - 0.5
    c, s = math.cos(roll_rad), math.sin(roll_rad)
    rx, ry = c * dx - s * dy, s * dx + c * dy
    cx_n, cy_n = rx + 0.5, ry + 0.5
    hf = abs(hfov * c) + abs(vfov * s)
    vf = abs(vfov * c) + abs(hfov * s)
    horiz_err = (cx_n - cfg.hori_goal) * hf                 # + = target right of centre
    vert_err = (cfg.vert_goal - cy_n) * vf                  # + = target above the goal row
    in_frame_elev = (0.5 - cy_n) * vf                       # + = target above frame centre
    return horiz_err, vert_err, in_frame_elev


def compute_rate_intent(target: Optional[FilteredTarget], cfg: RateConfig, state: RateState,
                        now: float, *, pitch_rad: float, roll_rad: float, gamma_rad: float,
                        agl_m: float) -> RateIntent:
    """One control step. `pitch_rad`/`roll_rad` = measured airframe attitude, `gamma_rad` =
    flight-path angle (+climb), `agl_m` = height above ground. Returns the smoothed body-rate
    intent. The caller sends it via backend.send_body_rates()."""
    state.ensure(cfg)
    dt = _clamp((now - state.last_t) if state.last_t is not None else 0.0, 0.0, 0.2)
    state.last_t = now

    if state.impacted or (target is None and agl_m < cfg.impact_agl_m):
        # IMPACT latch: lost the target near the ground -> stop for good (level + cut throttle).
        state.impacted = True
        pr = _clamp(2.0 * (0.0 - pitch_rad), -0.6, 0.6)
        rr, yr, thrust = -cfg.roll_return * roll_rad, 0.0, 0.0
        phase = "STOP"
    elif target is not None:
        det = target.detection
        cx_n, cy_n = det.x / cfg.frame_width, det.y / cfg.frame_height
        horiz_err, vert_err, in_frame_elev = _preprocess(cfg, cx_n, cy_n, roll_rad)
        ang_to_target = in_frame_elev + pitch_rad + math.radians(cfg.camera_pitch_deg)
        # PITCH rate frames the target to vert_goal; zero the rate at the attitude limit.
        pr = state.pitch_pid.update(vert_err, dt)
        if (pitch_rad <= -cfg.max_pitch_rad and pr < 0) or (pitch_rad >= cfg.max_pitch_rad and pr > 0):
            pr = 0.0
        # THRUST pursuit: drive the flight-path angle onto the LOS (+aim bias into the target).
        thrust = _clamp(state.hover + state.thrust_pid.update(ang_to_target + cfg.aim_bias_rad - gamma_rad, dt),
                        0.0, 1.0)
        # DEADZONE: near-centred -> no horizontal command (kills the yaw-pan shake from box noise).
        he = 0.0 if abs(horiz_err) < cfg.horiz_deadzone_rad else horiz_err
        ae = abs(he)
        alpha = _clamp((ae - cfg.horiz_thresh) / max(cfg.max_horiz_err, ae - cfg.horiz_thresh), 0.0, 1.0)
        state.yaw_pid.kp = alpha * cfg.yaw_kp                 # yaw dominates far off-axis
        state.roll_pid.kp = (1.0 - alpha) * cfg.roll_kp       # roll banks near centre
        yr = state.yaw_pid.update(he, dt)
        rr = state.roll_pid.update(he, dt) - cfg.roll_return * roll_rad
        sg = _clamp((det.h - 8.0) / 18.0, 0.3, 1.0)           # reduce horizontal authority when far/small
        yr *= sg
        rr *= sg
        phase = "RATE"
    else:
        # SEARCH (still high, no target): nose down to bring a below ground target into the FOV.
        pr = _clamp(2.0 * (cfg.search_pitch_rad - pitch_rad), -0.6, 0.6)
        rr, yr, thrust = -cfg.roll_return * roll_rad, 0.0, state.hover
        phase = "SEARCH"

    # Differential low-pass: pitch heavy (smooth dive), yaw heaviest (it was the oscillation),
    # roll moderate (banking), thrust mid.
    state.sm_pr += cfg.ema_pitch * (pr - state.sm_pr)
    state.sm_yr += cfg.ema_yaw * (yr - state.sm_yr)
    state.sm_rr += cfg.ema_roll * (rr - state.sm_rr)
    state.sm_thr += cfg.ema_thrust * (thrust - state.sm_thr)
    return RateIntent(roll_rate=state.sm_rr, pitch_rate=state.sm_pr, yaw_rate=state.sm_yr,
                      thrust=state.sm_thr, phase=phase)
