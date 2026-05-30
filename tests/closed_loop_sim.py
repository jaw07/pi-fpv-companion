"""Closed-loop airframe + fixed-camera simulator for TRACK / DIVE guidance.

The unit tests in test_visual_servo.py prove `compute_intent` is correct for a
SINGLE frame. They cannot answer the question that actually decides whether the
aircraft holds its lock in flight:

    The camera is BOLTED to the airframe. Every yaw/pitch command the servo
    issues ROTATES the field of view. Does the closed loop keep the target
    inside the frame, or does the guidance steer the FOV off the target and
    lose it?

This module closes the loop so that question can be answered:

    world target ──project──► pixels ──► Detection
        ▲                                   │
        │ (airframe moves/rotates)          ▼
    dynamics ◄── GuidanceIntent ◄── gate ◄── compute_intent ◄── AlphaBetaFilter

Everything between the camera and the dynamics is the REAL production code
(AlphaBetaTargetFilter, compute_intent, safety.gate) — only the airframe
kinematics and the pinhole camera are modelled here.

Geometry (ENU world, z = up; angles in radians internally)
----------------------------------------------------------
  heading ψ : aircraft forward (level) = (cosψ, sinψ, 0). +ψ rotates the nose
              LEFT (CCW from above), so a "yaw right" (+dps) command DECREASES ψ.
  pitch   φ : + = nose UP. Boresight F = (cosφ cosψ, cosφ sinψ, sinφ); nose-down
              (φ<0, the forward lean) depresses the boresight, which makes a
              target ahead-and-below RISE in the frame — the camera-pitch
              coupling the TRACK vcenter term and the DIVE aim are written for.
  right   R = (sinψ, -cosψ, 0)      image +x (to the right)
  up      U = R × F                 image up (smaller py)

  pixel = ( cx + fh·(d·R)/(d·F),  cy - fv·(d·U)/(d·F) )
          fh = (W/2)/tan(HFoV/2),  fv = (H/2)/tan(VFoV/2)   (anamorphic-accurate)

Dynamics (the FC's attitude/throttle decomposition, abstracted)
  - yaw   : FC tracks the commanded yaw RATE directly (fast rate loop).
  - pitch : attitude tracks the command through a first-order lag (tau_att);
            forward accel a = g·tan(-φ) (nose-down → accelerate forward), with
            linear drag → a finite cruise speed.
  - thrust: vertical speed = (thrust-0.5)·2·v_climb_max. 0.5 = hold. This is the
            adaptive-hover abstraction: altitude is a throttle job, decoupled
            from the forward lean (see docs/camera-pitch-coupling).

These are kinematic abstractions, not a 6-DOF model — fidelity is deliberately
spent on the attitude→pixel coupling (the FOV question) rather than aero.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field, replace
from typing import Callable, List, Optional, Tuple

from pi_fpv_companion.types import (
    Detection, Target, GuidanceMode, GuidanceIntent, SwitchState, ZERO_INTENT,
)
from pi_fpv_companion.guidance.visual_servo import ServoConfig, compute_intent
from pi_fpv_companion.guidance.safety import SafetyConfig, gate
from pi_fpv_companion.track.target_filter import AlphaBetaTargetFilter, FilterConfig

Vec = Tuple[float, float, float]
G = 9.81


def _dot(a: Vec, b: Vec) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec, b: Vec) -> Vec:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _norm(a: Vec) -> float:
    return math.sqrt(_dot(a, a))


@dataclass(frozen=True)
class CameraModel:
    """Pinhole projection of a fixed (airframe-bolted) camera.

    Defaults are the Raspberry Pi AI Camera (Sony IMX500) per its product brief:
    HFoV 66.3°, VFoV 52.3° (full-FoV sensor mode). Horizontal and vertical focal
    lengths are modelled separately, so a non-1:1 frame aspect (e.g. 720×576 PAL
    from the 4:3 array) is handled accurately rather than assuming square pixels."""
    width: int
    height: int
    hfov_deg: float = 66.3          # IMX500 horizontal FoV (product brief)
    vfov_deg: float = 52.3          # IMX500 vertical FoV (product brief)
    target_h_m: float = 1.7         # subject physical height (person) — drives bbox size
    target_w_m: float = 0.5

    @property
    def fpx_h(self) -> float:
        return (self.width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)

    @property
    def fpx_v(self) -> float:
        return (self.height / 2.0) / math.tan(math.radians(self.vfov_deg) / 2.0)

    def project(self, d_world: Vec, psi: float, phi: float):
        """Map a world relative-vector (target - aircraft) to a pixel detection.

        Returns (Detection | None, depth, in_frame). None when the target is
        behind the camera; in_frame is False when the projected centre falls
        outside [0,W)×[0,H) (the tracker would then get no box → coast)."""
        cphi, sphi = math.cos(phi), math.sin(phi)
        cpsi, spsi = math.cos(psi), math.sin(psi)
        F = (cphi * cpsi, cphi * spsi, sphi)        # boresight (forward)
        R = (spsi, -cpsi, 0.0)                       # image right
        U = _cross(R, F)                             # image up
        depth = _dot(d_world, F)
        if depth <= 0.05:
            return None, depth, False                # behind / on the lens plane
        px = self.width / 2.0 + self.fpx_h * (_dot(d_world, R) / depth)
        py = self.height / 2.0 - self.fpx_v * (_dot(d_world, U) / depth)
        h = self.fpx_v * self.target_h_m / depth     # bbox height -> vertical focal length
        w = self.fpx_h * self.target_w_m / depth
        in_frame = 0.0 <= px < self.width and 0.0 <= py < self.height
        det = Detection(x=px, y=py, w=w, h=h, confidence=0.9, class_id=0)
        return det, depth, in_frame

    def hold_range(self, desired_bbox_frac: float) -> float:
        """Range at which bbox-height/frame-height == desired (TRACK steady state)."""
        return self.fpx_v * self.target_h_m / (desired_bbox_frac * self.height)


@dataclass
class Airframe:
    """Kinematic state. Position in ENU metres, z = altitude."""
    pos: Vec = (0.0, 0.0, 0.0)
    psi: float = 0.0                 # heading (rad)
    phi: float = 0.0                 # pitch attitude (rad), + = nose up
    v_fwd: float = 0.0               # body forward speed (m/s)
    tau_att: float = 0.12            # attitude (pitch) first-order time constant (s)
    # Defaults grounded in SITL (scripts/measure_dive_sitl.py on ArduCopter 4.6.3):
    # STABILIZE full throttle-cut gives ~16 m/s descent, and a 30° lean yields only
    # ~3–5 m/s groundspeed in a 4 s window → descent capability far exceeds forward
    # closure. drag is set so cruise = g·tan(lean)/drag matches that modest forward
    # speed; v_climb_max so a full throttle deflection approaches the measured 16 m/s.
    drag: float = 1.1                # linear drag (1/s) → cruise = g·tan(pitch)/drag
    v_climb_max: float = 16.0        # descent speed at full throttle-cut (SITL: ~16 m/s)
    # Climbing is much slower than descending (gravity): SITL measured +1.2 m/s
    # climb vs -4.2 m/s descent at symmetric throttle offsets → ~0.3×. Modelled so
    # the sim doesn't overstate an above-target climb.
    climb_factor: float = 0.3
    # Adaptive-hover hold band (STABILIZE): a thrust within ±this of neutral is
    # treated as "hold altitude" and produces NO vertical motion — exactly as the
    # backend's adaptive-hover PI loop does (ArduCopterRcMapping.hover_learn_band).
    # Modelled here so the sim is faithful: a DIVE whose commit falls inside the
    # band would silently fail to descend, and the sim must show that.
    hover_hold_band: float = 0.05
    # Closed-loop rate tracking efficiency: the backend's PI loop reaches the
    # commanded vertical rate (SITL with the rate integral: cmd -3 → settled
    # ~-3.0 m/s); a small margin (0.9) models ramp-up lag. The servo's framing
    # loop absorbs any residual.
    vrate_track_eff: float = 0.9

    def step(self, intent: GuidanceIntent, dt: float) -> None:
        # Yaw: +dps = yaw RIGHT = clockwise from above = DECREASING ψ.
        self.psi -= math.radians(intent.yaw_rate_dps) * dt
        # Pitch attitude tracks the command through a first-order lag.
        cmd_phi = math.radians(intent.pitch_deg)
        k = 1.0 - math.exp(-dt / self.tau_att)
        self.phi += (cmd_phi - self.phi) * k
        # Forward accel from the lean: nose-down (φ<0) → accelerate forward.
        a_fwd = G * math.tan(-self.phi)
        self.v_fwd += (a_fwd - self.drag * self.v_fwd) * dt
        # Vertical: a closed-loop rate command (DIVE) is tracked by the backend's
        # climb-rate loop (under-delivered by vrate_track_eff, clamped to physical
        # capability — descent up to v_climb_max, climb gravity-limited). Otherwise a
        # thrust stick, with a near-neutral value held by adaptive hover.
        if intent.vertical_rate_mps is not None:
            v_vert = intent.vertical_rate_mps * self.vrate_track_eff
            cap_up = self.v_climb_max * self.climb_factor
            v_vert = max(-self.v_climb_max, min(cap_up, v_vert))
        else:
            dev = intent.thrust - 0.5
            if abs(dev) < self.hover_hold_band:
                v_vert = 0.0                               # held by adaptive hover
            else:
                v_vert = dev * 2.0 * self.v_climb_max
                if dev > 0.0:
                    v_vert *= self.climb_factor            # climbing is slower (gravity)
        self.pos = (
            self.pos[0] + self.v_fwd * math.cos(self.psi) * dt,
            self.pos[1] + self.v_fwd * math.sin(self.psi) * dt,
            self.pos[2] + v_vert * dt,
        )


@dataclass
class TickLog:
    t: float
    px: float
    py: float
    in_frame: bool
    depth: float
    range_m: float
    alt: float
    quality: float
    muted: bool
    reason: str
    pitch_cmd: float
    yaw_cmd: float
    thrust: float
    vrate_cmd: float    # commanded vertical rate (m/s, +up); 0 when none


@dataclass
class Trajectory:
    ticks: List[TickLog] = field(default_factory=list)

    # ---- FOV-retention metrics (the question the user asked) ----
    @property
    def ever_left_frame(self) -> bool:
        """True if the target left the frame on any tick AFTER the first lock."""
        seen = False
        for tk in self.ticks:
            if tk.in_frame:
                seen = True
            elif seen:
                return True
        return False

    @property
    def first_exit_t(self) -> Optional[float]:
        seen = False
        for tk in self.ticks:
            if tk.in_frame:
                seen = True
            elif seen:
                return tk.t
        return None

    def lost_before_impact(self, radius_m: float) -> bool:
        """Left the frame while the target was still FAR (range > radius). A
        terminal frame-exit inside the radius is the target filling/passing the
        frame at impact — expected homing endgame, not a tracking loss."""
        seen = False
        for tk in self.ticks:
            if tk.in_frame:
                seen = True
            elif seen and tk.range_m > radius_m:
                return True
        return False

    def _norm_excursion(self, tk: TickLog, W: int, H: int) -> float:
        """Peak normalised distance from centre on either axis (0=centre, 1=edge)."""
        return max(abs(tk.px - W / 2.0) / (W / 2.0), abs(tk.py - H / 2.0) / (H / 2.0))

    def peak_excursion(self, W: int, H: int) -> float:
        return max((self._norm_excursion(tk, W, H) for tk in self.ticks if tk.in_frame),
                   default=float("inf"))

    @property
    def min_range(self) -> float:
        return min((tk.range_m for tk in self.ticks), default=float("inf"))

    @property
    def final_range(self) -> float:
        return self.ticks[-1].range_m if self.ticks else float("inf")

    @property
    def altitude_lost(self) -> float:
        if not self.ticks:
            return 0.0
        return self.ticks[0].alt - self.ticks[-1].alt

    @property
    def muted_ticks(self) -> int:
        return sum(1 for tk in self.ticks if tk.muted)


@dataclass
class SimWorld:
    camera: CameraModel
    servo: ServoConfig
    safety: SafetyConfig
    airframe: Airframe
    target_pos: Vec                       # initial world position (m)
    target_vel: Vec = (0.0, 0.0, 0.0)     # world velocity (m/s)
    target_accel: Vec = (0.0, 0.0, 0.0)   # world accel (m/s²) — a maneuvering target
    filter_cfg: FilterConfig = field(default_factory=FilterConfig)
    armed: bool = True
    impact_range_m: float = 1.5           # stop when the aircraft reaches the target
    # --- Perception realism (exercises the filter quality/innovation gating +
    # safety watchdog that perfect detections never stress). Seeded → deterministic.
    detection_noise_px: float = 0.0       # gaussian std added to bbox centre x/y
    detection_dropout_prob: float = 0.0   # per-frame chance the detector returns nothing
    detect_latency_frames: int = 0        # detections are this many frames stale
    seed: int = 12345
    # glitch(i, clean_detection) -> a (possibly corrupted) Detection or None, for
    # injecting a misdetection / teleport / class-flip at a chosen tick.
    glitch: Optional[Callable[[int, Detection], Optional[Detection]]] = None
    # Optional real tracker (IouAssociator / MultiObjectTracker). When set, the
    # observed detection is routed through its consume() instead of being injected
    # as a Target directly — so the closed loop exercises the real association
    # (IoU/distance gate) under the moving FOV, not just the servo.
    tracker: object = None

    def run(self, mode: GuidanceMode, dt: float = 1.0 / 30.0,
            duration_s: float = 12.0, dive_after_s: Optional[float] = None) -> Trajectory:
        """Fly `mode` for the whole run, OR — when `dive_after_s` is set — fly
        TRACK first and switch to DIVE at that time (the operator committing after
        following), so the TRACK→DIVE handoff and filter continuity are exercised."""
        flt = AlphaBetaTargetFilter(self.filter_cfg)
        traj = Trajectory()
        af = self.airframe
        servo = self.servo
        tpos = self.target_pos
        tvel = self.target_vel
        rng_gen = random.Random(self.seed)
        det_buffer: List[Optional[Detection]] = []   # for detector latency
        dive_entered_t: Optional[float] = None        # for the DIVE lean soft-start
        t = 0.0
        n = int(duration_s / dt)
        for i in range(n):
            t += dt
            mode = (GuidanceMode.TRACK if (dive_after_s is not None and t < dive_after_s)
                    else (GuidanceMode.DIVE if dive_after_s is not None else mode))
            switch = SwitchState(active=True, pwm_us=1800, timestamp=t, mode=mode)
            if mode is GuidanceMode.DIVE:
                if dive_entered_t is None:
                    dive_entered_t = t
                dive_elapsed_s = t - dive_entered_t
            else:
                dive_entered_t = None
                dive_elapsed_s = 1e9
            tvel = (tvel[0] + self.target_accel[0] * dt,
                    tvel[1] + self.target_accel[1] * dt,
                    tvel[2] + self.target_accel[2] * dt)
            tpos = (tpos[0] + tvel[0] * dt, tpos[1] + tvel[1] * dt, tpos[2] + tvel[2] * dt)
            d_world = (tpos[0] - af.pos[0], tpos[1] - af.pos[1], tpos[2] - af.pos[2])
            rng = _norm(d_world)
            det, depth, in_frame = self.camera.project(d_world, af.psi, af.phi)

            # --- Perception realism: latency, dropout, noise, injected glitch ---
            clean = det if (det is not None and in_frame) else None
            det_buffer.append(clean)
            obs = det_buffer[-1 - self.detect_latency_frames] \
                if len(det_buffer) > self.detect_latency_frames else None
            if obs is not None and self.detection_dropout_prob > 0.0 \
                    and rng_gen.random() < self.detection_dropout_prob:
                obs = None                                     # detector missed this frame
            if obs is not None and self.detection_noise_px > 0.0:
                obs = replace(obs, x=obs.x + rng_gen.gauss(0.0, self.detection_noise_px),
                              y=obs.y + rng_gen.gauss(0.0, self.detection_noise_px))
            if self.glitch is not None:
                obs = self.glitch(i, obs)                      # inject misdetection / teleport

            # The tracker only produces a confirmed box when the target is in
            # frame. Out of frame (or behind) → no measurement → the filter
            # coasts and quality decays, exactly as on the aircraft. If a real
            # tracker is provided, route the detection through its association;
            # otherwise inject the observed detection directly as a Target.
            if self.tracker is not None:
                raw = self.tracker.consume(None, [obs] if obs is not None else [], t)
            else:
                raw = Target(detection=obs, track_id=1, lost_frames=0, timestamp=t) \
                    if obs is not None else None
            filtered = flt.update(raw, self.camera.width, self.camera.height, t)

            if filtered is None:
                intent = ZERO_INTENT
                muted, reason, q = True, "no target", 0.0
            else:
                proposed = compute_intent(filtered, servo, mode, dive_elapsed_s)
                res = gate(proposed, filtered, switch, self.armed, t, self.safety)
                intent = res.intent
                muted, reason, q = res.muted, res.reason, filtered.quality

            af.step(intent, dt)
            traj.ticks.append(TickLog(
                t=t, px=(det.x if det else float("nan")),
                py=(det.y if det else float("nan")),
                in_frame=bool(det is not None and in_frame),
                depth=depth, range_m=rng, alt=af.pos[2], quality=q,
                muted=muted, reason=reason,
                pitch_cmd=intent.pitch_deg, yaw_cmd=intent.yaw_rate_dps,
                thrust=intent.thrust, vrate_cmd=(intent.vertical_rate_mps or 0.0),
            ))
            if rng <= self.impact_range_m:
                break
        return traj


# ----------------------------------------------------------------------------
# Builders that mirror config/imx500.yaml so the sim exercises the SHIPPING gains
# ----------------------------------------------------------------------------

def imx500_servo(width: int = 720, height: int = 576, **overrides) -> ServoConfig:
    base = dict(
        frame_width=width, frame_height=height,
        max_yaw_rate_dps=60.0, max_pitch_deg=15.0, pixel_deadzone_px=20.0,
        yaw_p_gain=0.15, yaw_ff_gain=0.05, desired_bbox_frac=0.30,
        closure_p_gain=50.0, pitch_p_gain=0.15, track_vcenter_gain=0.10,
        dive_forward_deg=25.0, dive_climb_forward_deg=6.0, dive_max_pitch_deg=30.0,
        dive_center_frac=0.30,
        dive_vrate_gain=17.0, dive_max_descent_mps=8.0, dive_max_climb_mps=4.0,
        yaw_sign=1.0, pitch_sign=1.0,
    )
    base.update(overrides)
    return ServoConfig(**base)


def imx500_safety(**overrides) -> SafetyConfig:
    base = dict(watchdog_timeout_s=0.250, require_armed=True, min_track_quality=0.35)
    base.update(overrides)
    return SafetyConfig(**base)
