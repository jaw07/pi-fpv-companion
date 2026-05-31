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
behavior that drove the aircraft into the subject at constant speed. TRACK holds
the distance AT ENGAGEMENT — it captures the apparent size when you flick to
TRACK and keeps that gap, so it never flies in to a fixed standoff (it maintains;
it does not close). A closure INTEGRAL (closure_i_gain, PI control via
ClosureState) removes the residual: pure-P holds a target moving away FARTHER than
the captured distance (a steady size error is needed to sustain the chase lean),
and the integral winds up to supply that lean so the captured distance is held
exactly. (A feedforward can't do this: at steady state the apparent size is
constant, so its rate is zero.) Back-calculation anti-windup keeps the integral
off the pitch clamp.

Consumes a `FilteredTarget` (never the raw tracker output) — its bbox size is
the alpha-beta-smoothed value, so closure isn't chattering on raw detection
size noise. See track/target_filter.py / audit §5.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

from pi_fpv_companion.types import HOVER_THRUST, FilteredTarget, GuidanceIntent, GuidanceMode


@dataclass
class ClosureState:
    """Per-lock state for the TRACK closure loop.

    TRACK holds the distance AT ENGAGEMENT: on a fresh lock (or the first TRACK
    frame) it CAPTURES the target's current apparent size as the hold setpoint,
    then the PI loop keeps the target at that size — i.e. that distance — as the
    target moves. It does NOT converge to a fixed standoff, so locking a far
    target keeps it far and locking a near one keeps it near. It maintains; it
    never closes in.

    Pure-P alone leaves a steady-state RANGE offset against a target moving away
    (a residual size error is needed to produce the chase lean); the integral
    winds up to supply that lean so the captured distance is held exactly.
    Feedforward can't fix it — at steady state the apparent size is constant, so
    the size RATE is ~0.

    The setpoint is the MEDIAN apparent-size over the first `settle_n` frames of
    the lock, not the single first frame: the first detection after a fresh lock
    is often transient (motion blur, a GUIDED->STABILIZE attitude handoff, a
    partial bbox) and capturing it verbatim freezes a wrong hold distance for the
    whole engagement — the loop then leans hard to "close" a gap that isn't there.
    Medianing the settle window discards that one bad sample, and because the
    running setpoint tracks the samples DURING the window the range error stays
    ~0 there, so engage produces no dramatic pitch transient.

    Owned by the caller (one instance per pipeline), reset when the lock changes
    or guidance leaves TRACK, so the setpoint and windup never carry across
    targets/modes."""
    integral: float = 0.0          # accumulated range-error·seconds
    track_id: int = -1
    last_t: Optional[float] = None
    setpoint_inv: Optional[float] = None   # inverse-size (1/size_frac), median over the settle window
    settle_n: int = 5              # frames to median the engage setpoint over before freezing it
    _settle: list = field(default_factory=list)   # inverse-size samples collected during settle

    def reset(self) -> None:
        self.integral = 0.0
        self.track_id = -1
        self.last_t = None
        self.setpoint_inv = None
        self._settle = []

    def hold_setpoint(self, inv: float, track_id: int, now: float) -> float:
        """Capture the engage-distance hold setpoint and return it. On a fresh lock
        it restarts a short settle window; while settling, the setpoint tracks the
        running median of the samples so far (range error stays ~0, no engage
        transient); once `settle_n` samples are in, the median is frozen and held
        for the rest of the lock, so the loop holds THAT distance."""
        if track_id != self.track_id:
            self.track_id = track_id
            self.setpoint_inv = None
            self._settle = []
            self.integral = 0.0
            self.last_t = now
        if len(self._settle) < self.settle_n:
            self._settle.append(inv)
            self.setpoint_inv = sorted(self._settle)[len(self._settle) // 2]  # running median
        return self.setpoint_inv

    def accumulate(self, err: float, now: float) -> float:
        """Advance the integral by err·dt. Anti-windup clamping is applied by the
        caller (back-calculation against saturation)."""
        dt = max(0.0, now - self.last_t) if self.last_t is not None else 0.0
        self.last_t = now
        self.integral += err * dt
        return self.integral


@dataclass
class DiveState:
    """Per-dive state that low-passes the forward (nose-down) lean.

    The dive lean is ADAPTIVE to the commanded descent (steep when descending onto
    a below target, gentle when level/climbing). But the commanded descent varies
    frame-to-frame — when the vertical homing momentarily centres the target the
    descent eases and the lean would collapse toward gentle, then steepen again as
    the target sinks: the nose NODS up and down (and that pitch nod feeds back into
    the vertical framing). A committed dive should fly a STEADY collision course to
    the target centroid, so the lean is smoothed here over dive_lean_tau_s. Owned by
    the caller (one per pipeline), reset on leaving DIVE so a new dive starts fresh."""
    lean: float = 0.0
    last_t: Optional[float] = None

    def reset(self) -> None:
        self.lean = 0.0
        self.last_t = None

    def smooth(self, target_lean: float, now: float, tau: float) -> float:
        if tau <= 0.0 or self.last_t is None:   # first frame of the dive: snap, no lag
            self.last_t = now
            self.lean = target_lean
            return target_lean
        dt = max(0.0, now - self.last_t)
        self.last_t = now
        self.lean += (target_lean - self.lean) * (1.0 - math.exp(-dt / tau))
        return self.lean


@dataclass(frozen=True)
class ServoConfig:
    frame_width: int
    frame_height: int
    max_yaw_rate_dps: float
    max_pitch_deg: float          # pitch clamp, both directions (approach + back-off ceiling)
    pixel_deadzone_px: float
    yaw_p_gain: float             # deg/s of yaw rate per pixel of horizontal error
    yaw_ff_gain: float            # deg/s of yaw rate per (px/s) of target image vx
    # NOMINAL hold size, used only for the STANDBY HUD preview (and as a fallback
    # when no ClosureState is supplied). In flight TRACK does NOT converge to this:
    # it captures the apparent size at the moment of engagement and holds THAT
    # distance (see ClosureState). Setting it just frames the preview sensibly.
    desired_bbox_frac: float      # target bbox-height / frame-height (preview nominal)
    # Closure gains operate on the RANGE-LINEAR error (hold_setpoint_inv -
    # 1/size_frac), which is ∝ (hold_distance - range) — NOT on raw size error. This
    # conditions the loop the same at every distance (see compute_intent). Because
    # the units are inverse-size, the gain scale differs from a raw-size loop
    # (closure_p_gain ~ a few, not tens).
    closure_p_gain: float         # deg of pitch per unit range-linear error
    # Closure INTEGRAL (TRACK): deg of pitch per (range-linear-error · second). Pure-P
    # holds the target FARTHER than the engage distance when it is moving away (a
    # residual error is needed to sustain the chase lean); the integral winds up to
    # supply that lean so the steady-state offset → 0. Back-calculation anti-windup
    # (in compute_intent) stops it winding against the pitch clamp, and ClosureState
    # resets it per lock / on leaving TRACK. 0 = off (pure-P closure).
    closure_i_gain: float = 0.0
    # Vertical centring (pitch P on the vertical pixel error, mirroring yaw on the
    # horizontal one). DIVE uses pitch_p_gain to aim the dive; TRACK adds a (usually
    # gentler) track_vcenter_gain ON TOP of range-hold so a forward lean — which
    # tilts the fixed camera down and makes the target rise in frame — is corrected
    # back toward centre instead of letting the target drift out the top.
    # Lead pursuit: aim at where the target WILL be (current + lead_time · image
    # velocity) instead of where it is. Shortens the intercept path against a
    # crossing target (pure pursuit tail-chases). The alpha-beta filter supplies
    # the velocity. 0 = pure pursuit. Applies to yaw and the DIVE vertical aim.
    lead_time_s: float = 0.0
    pitch_p_gain: float = 0.15    # deg of pitch per px of VERTICAL error (TRACK/DIVE aim)
    # TRACK pitch = pure range-hold closure. A vertical re-centring nudge was tried
    # (track_vcenter_gain) but on a fixed camera it FIGHTS the range-hold: pulling a
    # low ground target up to centre means nose-down, which drifts the aircraft
    # forward and closes the range it is meant to hold (and at higher gain it nods).
    # TRACK holds RANGE; it does not vertically centre — the target sits at its
    # natural depression (low for a target you are above) but stays framed, and DIVE
    # does the vertical aiming. Left as a tunable but 0 by default (off).
    track_vcenter_gain: float = 0.0   # deg of pitch per px of vertical error; 0 = off (see above)
    # DIVE forward lean is ADAPTIVE to the engagement: STEEP when descending onto a
    # target BELOW the flight path (a fast, committed ground attack — and a steep
    # nose-down also points the fixed camera down at the target, keeping it framed),
    # but GENTLE when level/climbing toward an ABOVE target (a steep lean there
    # over-depresses the camera and pushes the target out the top faster than the
    # gravity-limited climb can re-centre it). Ramps from gentle→steep with the
    # commanded descent rate. dive_max_pitch_deg is DIVE's own (steeper) clamp,
    # separate from the gentler TRACK max_pitch_deg.
    dive_forward_deg: float = 10.0       # STEEP lean at full descent (fast ground attack)
    dive_climb_forward_deg: float = 6.0  # gentle lean when level / climbing (keeps it framed)
    dive_max_pitch_deg: float = 30.0     # DIVE nose-down clamp (steeper than TRACK)
    # Soft-start: ramp the steep lean in over this many seconds at DIVE commit so
    # the target doesn't slew across the frame faster than the tracker/filter can
    # follow (a snap to full lean briefly out-runs the velocity estimate). 0 = snap.
    dive_lean_ramp_s: float = 0.5
    # Low-pass time constant (s) on the dive forward lean (via DiveState), so the
    # nose travels STEADILY to the target instead of nodding steep<->gentle as the
    # commanded descent fluctuates. ~1 s holds a steady collision course while still
    # adapting slowly to a genuine below→above change. 0 = no smoothing.
    dive_lean_tau_s: float = 0.0
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
    dive_vrate_damp: float = 0.0       # DERIVATIVE damping: m/s per (normalised vert error / s).
                                       # Opposes the rate of change of the vertical frame error
                                       # (filter vy) so the vertical homing doesn't oscillate
                                       # against the backend rate-loop lag + pitch coupling. 0 = pure-P.
    dive_max_descent_mps: float = 8.0  # clamp on commanded descent (+ down)
    dive_max_climb_mps: float = 4.0    # clamp on commanded climb (gravity-limited, < descent)
    # PITCH-FOLDING (Peregrine): the camera is bolted to the airframe, so the nose-down
    # dive lean DEPRESSES the boresight — a ground target far below can then appear
    # near frame CENTRE even though the aircraft is high above it. A vertical homing on
    # frame position alone reads that as "on bearing", commands ~no descent, and
    # OVERFLIES the target at altitude. Folding the MEASURED airframe pitch into the
    # vertical error recovers the target's TRUE angle below the horizon (frame offset +
    # boresight depression), so the dive descends whenever the target is truly below —
    # it converges to the target on the horizon, i.e. the aircraft down at the target's
    # level (impact). vfov_deg converts the measured pitch into normalised frame units.
    # 0 = off (frame-only homing); 1 = full fold. Needs the backend to supply pitch.
    dive_pitch_fold: float = 0.0       # fraction of measured airframe pitch folded into the vert error
    vfov_deg: float = 52.3             # camera vertical FoV (IMX500) — converts pitch deg <-> frame units
    # Operator-correctable sign overrides (audit §6). A mirrored/flipped camera
    # inverts the error->command sign -> divergent positive feedback ("spins
    # away from target"). MUST be bench-validated (docs/deployment-safety.md §4).
    yaw_sign: float = 1.0         # set -1.0 if the bench self-test shows inversion
    pitch_sign: float = 1.0


# Floor on size_frac before inverting for the range-linear closure error, so a
# zero/degenerate box (h≈0) can't produce an infinite range estimate. 0.005 of the
# frame height caps the inferred range; the safety gate mutes such low-quality
# tracks anyway, this just keeps the arithmetic bounded.
_MIN_SIZE_FRAC = 0.005


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _deadband(v: float, dz: float) -> float:
    return 0.0 if abs(v) < dz else v


def compute_intent(
    target: FilteredTarget, cfg: ServoConfig, mode: GuidanceMode = GuidanceMode.TRACK,
    dive_elapsed_s: float = 1e9, closure: Optional[ClosureState] = None,
    dive: Optional[DiveState] = None, pitch_deg_measured: float = 0.0,
) -> GuidanceIntent:
    """Map the filtered target's pixel state to an attitude intent.

    `dive_elapsed_s` is the time since DIVE was engaged; it soft-starts the steep
    lean (dive_lean_ramp_s). Defaults to fully ramped, so TRACK and any caller that
    doesn't track it are unaffected.

    `closure` is the TRACK closure state. It captures the engage-distance setpoint
    and (when cfg.closure_i_gain > 0) carries the PI integral that holds it exactly;
    the caller passes it only while actually in TRACK and resets it on leaving TRACK
    / changing target. None = no setpoint capture + pure-P (STANDBY preview / DIVE).

    TRACK holds the distance at engagement (PI closure) at constant altitude.
    DIVE commits: PITCH leans forward to close, and a commanded vertical RATE
    (constant-bearing homing — the backend tracks it on VFR_HUD.climb) holds the
    target's vertical frame position so the flight path follows the line of sight,
    moving altitude onto the target whether it is below, level, or above. Yaw
    centring is identical in both."""
    cx = cfg.frame_width / 2.0
    det = target.detection
    # Lead pursuit: aim at the target's predicted position (current + lead · image
    # velocity), so yaw/dive aim at the intercept instead of tail-chasing a crosser.
    aim_x = det.x + cfg.lead_time_s * target.vx_px_s
    aim_y = det.y + cfg.lead_time_s * target.vy_px_s

    # Horizontal: P on the centring error + feedforward on the target's image
    # velocity. P alone leaves a structural lag against a moving target; the
    # feedforward (target moving right -> pre-emptively yaw right) cancels it.
    dx = _deadband(aim_x - cx, cfg.pixel_deadzone_px)
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
        # VERTICAL: constant-bearing homing. Command a climb rate that drives the
        # target's vertical frame error to zero — holding it at a fixed frame point
        # is a constant bearing, i.e. a collision course, so the flight path tracks
        # the LOS for a target below / level / above. Below centre (drifted low) ->
        # descend (raises it back); above centre -> climb. Gated on horizontal aim
        # so we centre yaw before committing power. The backend tracks the rate on
        # VFR_HUD.climb (its P-loop's steady-state error is integrated out here).
        e_y = _deadband(aim_y - cy, cfg.pixel_deadzone_px) / (cfg.frame_height / 2.0)
        # Pitch-fold: add the boresight depression (nose-down -> camera looks down) so
        # the error reflects the target's TRUE angle below the horizon, not just where
        # the nose-down lean parks it in frame. Without this a high dive reads a far
        # ground target as centred and overflies it; with it the error stays positive
        # (below horizon) until the aircraft is down at the target's level -> it
        # descends onto the target instead of flying over. (boresight depression =
        # -pitch; nose-down pitch is negative, so this adds positive = descend.)
        e_y += cfg.dive_pitch_fold * (-pitch_deg_measured) / (cfg.vfov_deg / 2.0)
        ex = (aim_x - cx) / (cfg.frame_width / 2.0)
        commit = _clamp(1.0 - abs(ex) / cfg.dive_center_frac, 0.0, 1.0) \
            if cfg.dive_center_frac > 0 else 0.0
        if cfg.dive_vrate_gain > 0.0:
            # PD, not P: a pure-P rate command on the vertical frame error oscillates
            # against the backend's rate-tracking lag + the pitch/camera coupling (the
            # up/down "wiggle" seen in the Gazebo dive). The derivative term opposes the
            # rate of change of the error (the filter's vertical image velocity) and
            # damps it. e_y_rate = vy_px_s / (frame_height/2) is d(e_y)/dt.
            e_y_rate = target.vy_px_s / (cfg.frame_height / 2.0)
            vertical_rate_mps = commit * _clamp(
                -cfg.dive_vrate_gain * e_y - cfg.dive_vrate_damp * e_y_rate,
                -cfg.dive_max_descent_mps, cfg.dive_max_climb_mps,
            )

        # PITCH: forward commit lean. A committed dive holds the STEEP lean so it keeps
        # intercept speed all the way to the target — tying the lean to the INSTANTANEOUS
        # descent (the old design) collapsed it to gentle the moment the vertical homing
        # centred the target, so a high dive descended onto the line of sight and then
        # CRAWLED forward at ~1 m/s, never arriving. So stay steep by default and ease
        # toward gentle only when actually CLIMBING to an ABOVE target (there a steep
        # lean over-depresses the fixed camera and pushes the target out the top faster
        # than the gravity-limited climb can re-centre it). A below/level target keeps
        # the camera framed via the nose-down lean itself. The throttle (vertical rate),
        # not pitch, does the vertical aiming.
        ramp = max(1e-3, 0.25 * cfg.dive_max_descent_mps)
        climb_mps = max(0.0, vertical_rate_mps) if vertical_rate_mps is not None else 0.0
        climb_frac = _clamp(climb_mps / ramp, 0.0, 1.0)               # 0 = level/descending, 1 = climbing hard
        committed = cfg.dive_forward_deg \
            - (cfg.dive_forward_deg - cfg.dive_climb_forward_deg) * climb_frac
        # Soft-start: at commit, ramp the lean up from gentle over dive_lean_ramp_s so
        # the camera doesn't slew faster than the filter can track.
        soft = _clamp(dive_elapsed_s / cfg.dive_lean_ramp_s, 0.0, 1.0) \
            if cfg.dive_lean_ramp_s > 0.0 else 1.0
        lean = cfg.dive_climb_forward_deg + (committed - cfg.dive_climb_forward_deg) * soft
        # Low-pass the lean so the nose travels STEADILY to the target centroid: the
        # adaptive lean would otherwise flip steep<->gentle as the commanded descent
        # crosses its threshold (the vertical homing momentarily centring the target
        # eases the descent), nodding the pitch up and down. A committed dive holds a
        # steady collision course. (dive_lean_tau_s = 0 → no smoothing.)
        if dive is not None and cfg.dive_lean_tau_s > 0.0:
            lean = dive.smooth(lean, target.timestamp, cfg.dive_lean_tau_s)
        pitch = _clamp(-lean, -cfg.dive_max_pitch_deg, 0.0)
    else:
        # TRACK: range-hold (PI on the range-linear closure error) PLUS a vertical-
        # centring term. The fixed camera tilts with the airframe, so leaning forward to
        # close makes the target rise in frame; the vcentre term pitches back up to
        # re-centre it (limiting the lean) so it stays in view. The two share the
        # pitch budget — closure dominates, vcentre keeps the target from drifting
        # out the top. (Accommodating a target at a different *altitude* is a
        # throttle job, not pitch — see docs/dive-guidance.md.)
        cy = cfg.frame_height / 2.0
        dy = _deadband(det.y - cy, cfg.pixel_deadzone_px)
        size_frac = (det.h / cfg.frame_height) if cfg.frame_height else 0.0
        # RANGE-LINEAR closure error. Apparent size is ∝ 1/range, so 1/size_frac is
        # ∝ range; regulating on it (rather than raw size) makes the loop behave
        # identically at all distances. Raw size error falls off as 1/range — a far
        # target responds sluggishly, which an integral then turns into a slow limit
        # cycle. range_err = 1/desired - 1/size_frac ∝ (hold_range - range):
        #   < 0  -> target too FAR (small box)  -> nose down, chase
        #   ~ 0  -> at the hold distance
        #   > 0  -> too CLOSE (big box)          -> nose up, back off
        inv = 1.0 / size_frac if size_frac > _MIN_SIZE_FRAC else 1.0 / _MIN_SIZE_FRAC
        # Hold the distance AT ENGAGEMENT: the setpoint is the inverse-size captured
        # on the first TRACK frame of this lock, NOT a fixed desired_bbox_frac — so
        # TRACK maintains whatever gap you locked at and never flies in. (No closure
        # state — STANDBY preview / DIVE — falls back to the nominal desired_bbox_frac
        # for the HUD preview only; the gate mutes that command anyway.)
        hold_inv = closure.hold_setpoint(inv, target.track_id, target.timestamp) \
            if closure is not None else (1.0 / cfg.desired_bbox_frac)
        range_err = hold_inv - inv     # <0 target drifted FARTHER than hold -> nose down
        # PI closure: the integral cancels the steady-state range offset that pure-P
        # leaves on a receding target. The integrator is only active when the caller
        # supplies its state (i.e. actually in TRACK) and the gain is enabled. This is
        # the WHOLE of the TRACK pitch — it holds RANGE and nothing else (see
        # track_vcenter_gain): the closure noses down when the target drifts farther
        # (smaller) and noses up when it drifts closer (bigger), so the aircraft holds
        # the engage distance instead of closing in.
        integ = closure.accumulate(range_err, target.timestamp) \
            if (closure is not None and cfg.closure_i_gain > 0.0) else 0.0
        # Vertical re-centring nudge (track_vcenter_gain, 0 by default): on a fixed
        # camera it fights the range-hold (centring a low target = nose-down = forward
        # drift = closing the range), so it is off — TRACK lets the target sit at its
        # natural vertical position and DIVE does the vertical aiming.
        vcenter = cfg.track_vcenter_gain * dy
        unclamped = (
            cfg.pitch_sign * (cfg.closure_p_gain * range_err + cfg.closure_i_gain * integ)
            - cfg.pitch_sign * vcenter
        )
        pitch = _clamp(unclamped, -cfg.max_pitch_deg, cfg.max_pitch_deg)
        # Back-calculation anti-windup: when the command saturates, roll the integral
        # back to exactly the value that holds pitch at the clamp, so it can't keep
        # winding (and instantly unwinds the moment the error reverses).
        if closure is not None and cfg.closure_i_gain > 0.0 and pitch != unclamped:
            closure.integral = (
                cfg.pitch_sign * pitch - cfg.closure_p_gain * range_err + vcenter
            ) / cfg.closure_i_gain

    return GuidanceIntent(
        roll_deg=0.0,
        pitch_deg=pitch,
        yaw_rate_dps=yaw_rate,
        thrust=thrust,
        timestamp=target.timestamp,
        vertical_rate_mps=vertical_rate_mps,
    )
