"""ArduPilot backend over MAVLink — GPS-denied control via RC override.

A bare analog FPV quad has no GPS / position estimate. The robust GPS-denied way
to let the companion fly it is a self-levelling PILOT mode, with the companion
injecting AETR stick values via **RC_CHANNELS_OVERRIDE**. Two modes (control_mode;
must match the FC's flight mode — see docs/gps-denied-modes.md):

  - **STABILIZE (default)** — direct throttle, no altitude hold. Enables a true
    steep dive (SITL: ~16 m/s, ~77deg path vs ALT_HOLD's ~1-5 m/s). The companion
    owns altitude (no baro floor); hover is at `hover_throttle_us`.
  - ALT_HOLD — throttle is climb-rate (0.5=hold via baro), descent capped at
    PILOT_SPEED_DN. Gentler / safer altitude, but cannot dive aggressively.

Both need only baro + IMU (no GPS, no EKF origin). Both are pilot modes, so
handback is automatic: when the companion stops overriding (`release()`),
ArduPilot reverts the channels to the real RC radio — no GUIDED "ignore pilot +
hover on timeout" lockout.

  intent.roll_deg     -> roll stick  (lean angle; full stick = ANGLE_MAX)
  intent.pitch_deg    -> pitch stick (lean angle; negative = nose-down = forward)
  intent.yaw_rate_dps -> yaw stick   (yaw rate; full stick = pilot_yaw_rate_dps)
  intent.thrust       -> throttle    (see _throttle_pwm: stabilize=direct/hover-centred,
                         althold=climb-rate-centred)

Engagement is decided by the pipeline from the engage switch (`switch_channel`):
TRACK/DIVE -> `send_intent` (override the AETR channels), STANDBY -> `release`
(hand the channels back to the pilot). The pilot owns the flight-MODE channel and
must keep the craft in the matching self-levelling mode; the companion never
changes flight mode.

Stick signs are TX/RCMAP/airframe dependent — bench/SITL-validate them
(deployment-safety.md §4); `ArduCopterRcMapping` exposes per-axis sign flips.

`pymavlink` is imported lazily so the module is importable without it.
"""
from __future__ import annotations
import logging
import math
import time
from dataclasses import dataclass
from typing import Dict, Optional

from pi_fpv_companion.types import GuidanceIntent, GuidanceMode, SwitchState

_log = logging.getLogger(__name__)

# Re-ask for telemetry streams this often. ArduPilot drops a serial link's stream
# subscriptions on reconnect, so a one-shot request at startup is not enough —
# without periodic re-request the engage switch (RC_CHANNELS) and adaptive-hover
# climb rate (VFR_HUD) silently stop after any link blip.
_STREAM_REREQUEST_S = 5.0

# ArduCopter flight-mode numbers we drive via RC override. The interlock
# (control_ready) refuses to override unless the FC is actually in the mode that
# matches control_mode — so a stabilize mapping can't be pushed into ALT_HOLD /
# LOITER / a GPS mode by mistake. None for control_mode -> no interlock.
_EXPECTED_MODE = {"stabilize": 0, "althold": 2, "guided_nogps": 20}   # STABILIZE / ALT_HOLD / GUIDED_NOGPS custom_mode

# GUID_OPTIONS bit 3 (=8): SetAttitudeTarget interprets the thrust field as THRUST,
# not a climb-rate. MANDATORY for the guided_nogps rate path — without it ArduCopter
# reads SET_ATTITUDE_TARGET.thrust as a climb-rate command (0.5 = hold altitude), so
# "throttle 0" never descends and the dive planes. Verified in the preflight param check.
GUID_OPTIONS_THRUST_AS_THRUST = 8


@dataclass(frozen=True)
class ArduCopterRcMapping:
    """Maps the attitude/rate intent to AETR RC-override PWMs for a self-levelling
    ArduCopter pilot mode. Channels follow RCMAP (default roll=1, pitch=2,
    throttle=3, yaw=4); roll/pitch/yaw centre at level.

    `control_mode` sets how thrust maps to the throttle stick — it MUST match the
    flight mode the FC is actually in:
      - "althold":  throttle = climb RATE; 0.5 -> centre (hold altitude via baro),
                    bounded by PILOT_SPEED_UP/DN. Safe default; gentle dives.
      - "stabilize": throttle = DIRECT; 0.5 -> hover_throttle_us, 0 -> motors min,
                    1 -> max. No altitude hold -> a true dive (cut throttle + nose
                    down), but the companion owns altitude (no baro floor)."""
    angle_max_deg: float = 45.0          # full roll/pitch stick = this lean (FC ANGLE_MAX default 4500cdeg)
    pilot_yaw_rate_dps: float = 180.0    # full yaw stick = this yaw rate (SITL-measured ~180)
    roll_channel: int = 1
    pitch_channel: int = 2
    throttle_channel: int = 3
    yaw_channel: int = 4
    roll_sign: int = 1                   # +1 / -1 (bench/SITL validate, audit §4)
    pitch_sign: int = 1
    yaw_sign: int = 1
    center_us: int = 1500
    half_range_us: int = 500             # full deflection from center (1000..2000)
    control_mode: str = "stabilize"      # "stabilize" | "althold" (default stabilize: dive-capable)
    hover_throttle_us: int = 1450        # stabilize: starting hover guess (learner refines it)
    # Adaptive hover (stabilize only): trim the hover throttle from measured climb
    # rate so the craft holds altitude in TRACK without manual tuning — a companion
    # vertical-velocity hold (STABILIZE has none). Frozen during a commanded dive
    # and when telemetry is stale; output clamped to [hover_min_us, hover_max_us].
    hover_learn: bool = True
    # PI velocity hold: Kp damps climb-rate immediately (stops oscillation), Ki
    # slowly trims the learned hover throttle to remove the steady-state bias.
    hover_learn_kp: float = 50.0         # PWM per (m/s) of climb (immediate damping)
    hover_learn_gain: float = 20.0       # Ki: PWM per (m/s) of climb per second (slow trim)
    # Hold deadband for the open-loop THRUST-STICK vertical path: the PI loop holds
    # altitude while |thrust-0.5| < this, outside it the stick passes through.
    # (The closed-loop DIVE commands a vertical RATE instead — see _adaptive_throttle
    # — which is tracked directly and does not use this band.) TRACK emits EXACTLY
    # 0.5, so this only needs to catch neutral.
    hover_learn_band: float = 0.05
    hover_min_us: int = 1200             # safety clamp on the learned hover
    hover_max_us: int = 1700
    # Climb rate (m/s) that maps to a full throttle stick when a commanded
    # vertical_rate must be flown open-loop (no fresh VFR_HUD to close the loop).
    rate_openloop_full_mps: float = 8.0
    # Integral on the vertical-rate error (only while tracking a commanded rate):
    # kills the pure-P steady-state droop so the loop reaches the setpoint.
    rate_i_gain: float = 25.0            # PWM per (m/s) of rate error per second
    rate_i_max_us: float = 250.0         # anti-windup clamp on the integral term


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _throttle_pwm(intent: GuidanceIntent, m: ArduCopterRcMapping) -> int:
    """thrust(0..1) -> throttle PWM. althold: symmetric climb-rate about centre.
    stabilize: direct throttle with hover at thrust 0.5 (so TRACK ~holds and a
    full-down stick really cuts power for a dive)."""
    t = _clamp((intent.thrust - 0.5) / 0.5, -1.0, 1.0)   # -1 (down) .. +1 (up)
    if m.control_mode == "stabilize":
        span = (2000 - m.hover_throttle_us) if t >= 0 else (m.hover_throttle_us - 1000)
        return int(round(m.hover_throttle_us + t * span))
    return int(round(m.center_us + t * m.half_range_us))


def intent_to_rc_overrides(intent: GuidanceIntent, m: ArduCopterRcMapping) -> Dict[int, int]:
    """Return {channel: pwm_us} for the four AETR channels. roll/pitch/yaw centre
    at level; throttle per control_mode (see _throttle_pwm)."""
    def defl(frac: float, sign: int) -> int:
        return int(round(m.center_us + sign * _clamp(frac, -1.0, 1.0) * m.half_range_us))
    return {
        m.roll_channel: defl(intent.roll_deg / m.angle_max_deg, m.roll_sign),
        m.pitch_channel: defl(intent.pitch_deg / m.angle_max_deg, m.pitch_sign),
        m.yaw_channel: defl(intent.yaw_rate_dps / m.pilot_yaw_rate_dps, m.yaw_sign),
        m.throttle_channel: _throttle_pwm(intent, m),
    }


class ArduPilotBackend:
    """MAVLink backend. Works against any connection string pymavlink accepts:
    a serial device path with baud, `udpin:host:port`, `tcp:host:port`, etc."""

    def __init__(
        self,
        device: str,
        baud: int,
        switch_channel: int,
        track_threshold_us: int,
        dive_threshold_us: int,
        mapping: Optional[ArduCopterRcMapping] = None,
        select_channel: int = 0,
    ) -> None:
        self._device = device
        self._baud = baud
        self._switch_channel = switch_channel
        self._select_channel = select_channel    # 0 = disabled
        self._select_pwm_us = 0
        # 3-position mode switch: pwm >= dive -> DIVE, >= track -> TRACK, else STANDBY.
        self._track_threshold_us = track_threshold_us
        self._dive_threshold_us = dive_threshold_us
        self._mapping = mapping or ArduCopterRcMapping()
        self._mavutil = None
        self._mav = None
        self._last_switch: Optional[SwitchState] = None
        self._armed = False
        # Adaptive-hover state (stabilize): learned hover PWM trimmed from climb rate.
        self._hover_pwm: float = float(self._mapping.hover_throttle_us)
        self._hover_t: float = 0.0           # last adapt time (for dt)
        self._climb_mps: float = 0.0         # latest VFR_HUD.climb (+up)
        self._gs_mps: float = 0.0            # latest VFR_HUD.groundspeed (forward speed)
        self._alt_m: float = 0.0             # latest VFR_HUD.alt
        self._climb_t: float = 0.0           # when _climb_mps was last updated
        self._pitch_rad: float = 0.0         # latest ATTITUDE.pitch (+nose-up)
        self._roll_rad: float = 0.0          # latest ATTITUDE.roll (+bank-right)
        self._yaw_rad: float = 0.0           # latest ATTITUDE.yaw (heading)
        self._pitch_t: float = 0.0           # when _pitch_rad / _roll_rad was last updated
        self._x_m: float = 0.0               # latest LOCAL_POSITION_NED north (m)
        self._y_m: float = 0.0               # latest LOCAL_POSITION_NED east (m)
        self._pos_t: float = 0.0             # when _x_m / _y_m was last updated
        self._vrate_i: float = 0.0           # vertical-rate-loop integral term (PWM)
        self._last_stream_req: float = 0.0   # last telemetry-stream (re)request
        self._vfr_warned: bool = False       # warned once that VFR_HUD isn't arriving
        self._current_mode: Optional[int] = None   # latest HEARTBEAT custom_mode (FC flight mode)
        self._interlock_warned: bool = False        # warned once that FC mode != expected

    def open(self) -> None:
        """Bind / connect the transport. Does NOT block on heartbeat — call
        `wait_ready()` separately if you need target_system populated first."""
        from pymavlink import mavutil  # lazy
        self._mavutil = mavutil
        self._mav = mavutil.mavlink_connection(
            self._device, baud=self._baud, autoreconnect=True
        )

    def wait_ready(self, timeout: float = 10.0) -> None:
        """Block until first HEARTBEAT seen so target_system/component are known,
        then request the telemetry streams: RC_CHANNELS (engage switch) and
        VFR_HUD (climb rate, for adaptive hover)."""
        self._mav.wait_heartbeat(timeout=timeout)
        self._request_streams()

    def read_param(self, name: str, timeout: float = 5.0) -> Optional[tuple]:
        """Read one FC parameter. Returns (value, mav_param_type) or None on timeout.
        Call at startup (before the main loop drains messages)."""
        if self._mav is None:
            return None
        self._mav.mav.param_request_read_send(
            self._mav.target_system, self._mav.target_component, name.encode(), -1)
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            pv = self._mav.recv_match(type="PARAM_VALUE", blocking=True,
                                      timeout=max(0.1, end - time.monotonic()))
            if pv is not None and pv.param_id.strip("\x00") == name:
                return float(pv.param_value), int(pv.param_type)
        return None

    def ensure_params(self, desired: Dict[str, float], tol: float = 0.5,
                      timeout: float = 2.0) -> Dict[str, str]:
        """Confirm each desired FC parameter and WRITE any that differ, verifying the
        write. The user authorised this (startup FC validation/auto-config). Only the
        listed params are touched; every action is logged. Returns {name: status}
        where status is 'ok' | 'set' | 'read-fail' | 'write-fail'.

        Bounded for startup: each read uses a short timeout, and the FIRST
        unresponsive read aborts the whole pass (a wrong-baud / dead FC must not
        stall boot for timeout × N params)."""
        result: Dict[str, str] = {}
        aborted = False
        for name, want in desired.items():
            if aborted:
                result[name] = "read-fail"
                continue
            cur = self.read_param(name, timeout=timeout)
            if cur is None:
                _log.warning("FC param %s: no response — aborting param validation "
                             "(is the FC link up?)", name)
                result[name] = "read-fail"
                aborted = True          # FC not responding; don't wait on the rest
                continue
            value, ptype = cur
            if abs(value - want) <= tol:
                _log.info("FC param %s = %g (ok)", name, value)
                result[name] = "ok"
                continue
            _log.warning("FC param %s = %g, want %g — writing", name, value, want)
            self._mav.mav.param_set_send(self._mav.target_system, self._mav.target_component,
                                         name.encode(), float(want), ptype)
            check = self.read_param(name)
            if check is not None and abs(check[0] - want) <= tol:
                _log.warning("FC param %s -> %g (written, verified)", name, want)
                result[name] = "set"
            else:
                got = check[0] if check else float("nan")
                _log.error("FC param %s write FAILED (still %g, wanted %g)", name, got, want)
                result[name] = "write-fail"
        return result

    def ensure_param_bits(self, name: str, bits: int, timeout: float = 2.0) -> str:
        """Confirm specific BITS are set in a bitmask FC parameter, OR-ing them in (and
        verifying) without clobbering the other bits. Used for GUID_OPTIONS bit 3
        (ThrustAsThrust) on the guided_nogps path — see GUID_OPTIONS_THRUST_AS_THRUST.
        Returns 'ok' (already set) | 'set' (written) | 'read-fail' | 'write-fail'."""
        cur = self.read_param(name, timeout=timeout)
        if cur is None:
            _log.warning("FC param %s: no response — cannot verify bits 0x%X", name, bits)
            return "read-fail"
        value, ptype = cur
        ivalue = int(round(value))
        if (ivalue & bits) == bits:
            _log.info("FC param %s = %d (bits 0x%X already set)", name, ivalue, bits)
            return "ok"
        want = ivalue | bits
        _log.warning("FC param %s = %d, setting bits 0x%X -> %d", name, ivalue, bits, want)
        self._mav.mav.param_set_send(self._mav.target_system, self._mav.target_component,
                                     name.encode(), float(want), ptype)
        check = self.read_param(name)
        if check is not None and (int(round(check[0])) & bits) == bits:
            _log.warning("FC param %s -> %d (bits 0x%X set, verified)", name, want, bits)
            return "set"
        got = int(round(check[0])) if check else -1
        _log.error("FC param %s bit-set FAILED (still %d, wanted bits 0x%X)", name, got, bits)
        return "write-fail"

    def _request_streams(self, rate_hz: int = 10) -> None:
        """ArduPilot does not stream RC_CHANNELS / VFR_HUD / ATTITUDE on a MAVLink
        serial port until a GCS asks. Without RC_CHANNELS the engage switch is
        stuck; without VFR_HUD adaptive hover has no climb-rate feedback; without
        ATTITUDE the agnostic dive has no airframe pitch and falls back to 0
        (treating in-frame elevation as true LOS elevation). Try the modern
        per-message interval, fall back to the legacy data streams."""
        if self._mav is None:
            return
        mav = self._mavutil.mavlink
        for msg_id in (mav.MAVLINK_MSG_ID_RC_CHANNELS, mav.MAVLINK_MSG_ID_VFR_HUD,
                       mav.MAVLINK_MSG_ID_ATTITUDE):
            try:
                self._mav.mav.command_long_send(
                    self._mav.target_system, self._mav.target_component,
                    mav.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                    msg_id, int(1_000_000 / rate_hz), 0, 0, 0, 0, 0,
                )
            except Exception:
                pass
        for stream in (mav.MAV_DATA_STREAM_RC_CHANNELS, mav.MAV_DATA_STREAM_EXTRA2,
                       mav.MAV_DATA_STREAM_EXTRA1):
            try:
                self._mav.mav.request_data_stream_send(
                    self._mav.target_system, self._mav.target_component,
                    stream, rate_hz, 1,
                )
            except Exception:
                pass

    def _mode_for(self, pwm: int) -> GuidanceMode:
        if pwm >= self._dive_threshold_us:
            return GuidanceMode.DIVE
        if pwm >= self._track_threshold_us:
            return GuidanceMode.TRACK
        return GuidanceMode.STANDBY

    def close(self) -> None:
        if self._mav is not None:
            self._mav.close()
            self._mav = None

    def _drain(self) -> None:
        if self._mav is None:
            return
        # Keep RC_CHANNELS + VFR_HUD flowing across link reconnects (see
        # _STREAM_REREQUEST_S) — cheap, and the alternative is a stuck engage
        # switch / starved adaptive hover after any blip.
        now = time.monotonic()
        if now - self._last_stream_req > _STREAM_REREQUEST_S:
            self._request_streams()
            self._last_stream_req = now
        while True:
            msg = self._mav.recv_match(blocking=False)
            if msg is None:
                break
            t = msg.get_type()
            if t == "HEARTBEAT":
                armed_bit = self._mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                self._armed = bool(msg.base_mode & armed_bit)
                self._current_mode = int(msg.custom_mode)
            elif t == "RC_CHANNELS":
                pwm = getattr(msg, f"chan{self._switch_channel}_raw")
                mode = self._mode_for(pwm)
                self._last_switch = SwitchState(
                    active=mode is not GuidanceMode.STANDBY,
                    pwm_us=pwm,
                    timestamp=time.monotonic(),
                    mode=mode,
                )
                if self._select_channel:
                    self._select_pwm_us = getattr(msg, f"chan{self._select_channel}_raw")
            elif t == "VFR_HUD":
                self._climb_mps = float(msg.climb)   # +up; baro-derived (no GPS needed)
                self._alt_m = float(msg.alt)         # altitude (AMSL-ish from baro)
                self._gs_mps = float(msg.groundspeed)  # forward speed (m/s) for flight-path angle
                self._climb_t = time.monotonic()
            elif t == "ATTITUDE":
                self._pitch_rad = float(msg.pitch)   # +nose-up (aerospace convention)
                self._roll_rad = float(msg.roll)     # +bank-right
                self._yaw_rad = float(msg.yaw)       # heading (rad)
                self._pitch_t = time.monotonic()
            elif t == "LOCAL_POSITION_NED":
                self._x_m = float(msg.x)             # NED north (m from origin)
                self._y_m = float(msg.y)             # NED east  (m from origin)
                self._pos_t = time.monotonic()

    def select_pwm(self) -> int:
        """Latest PWM on the target-select channel (0 if disabled / not yet seen).
        The pipeline edge-detects this to cycle the locked target (multi_iou)."""
        return self._select_pwm_us

    def pitch_deg(self) -> float:
        """Airframe pitch in degrees (+nose-up) from ATTITUDE, for the agnostic
        dive's LOS-elevation framing. Returns 0.0 (level) when no fresh ATTITUDE
        has arrived — the dive then keys on in-frame elevation alone, which is a
        safe degradation (it just can't tell a high-framed ground target from a
        truly-above one until telemetry resumes)."""
        if not self._pitch_t or (time.monotonic() - self._pitch_t) > 0.5:
            return 0.0
        return math.degrees(self._pitch_rad)

    def roll_deg(self) -> float:
        """Airframe roll in degrees (+bank-right) from ATTITUDE, for roll-compensating
        the frame error (the bolted camera rolls with the airframe). Returns 0.0 (level)
        on stale telemetry — compensation then no-ops, which is safe."""
        if not self._pitch_t or (time.monotonic() - self._pitch_t) > 0.5:
            return 0.0
        return math.degrees(self._roll_rad)

    def yaw_deg(self) -> float:
        """Airframe heading in degrees from ATTITUDE (0 on stale telemetry). Used by the
        attitude-control path to seed/track the yaw setpoint."""
        if not self._pitch_t or (time.monotonic() - self._pitch_t) > 0.5:
            return 0.0
        return math.degrees(self._yaw_rad)

    def pos_xy(self) -> tuple:
        """Latest LOCAL_POSITION_NED (north, east) in m from the EKF origin. For
        diagnostics/measurement only (the GPS-denied flight path never uses position).
        Returns (0.0, 0.0) on stale telemetry."""
        if not self._pos_t or (time.monotonic() - self._pos_t) > 1.0:
            return (self._x_m, self._y_m)
        return (self._x_m, self._y_m)

    def flight_path_angle_rad(self) -> float:
        """Flight-path angle (rad, +climb / -descent) from VFR_HUD climb-rate and
        groundspeed. For pursuit guidance: drive this onto the line-of-sight to the
        target so the velocity vector points straight at it. Returns 0.0 on stale
        telemetry (degrades to level)."""
        if not self._climb_t or (time.monotonic() - self._climb_t) > 0.5:
            return 0.0
        return math.atan2(self._climb_mps, max(self._gs_mps, 5.0))

    def alt_m(self) -> float:
        """Latest VFR_HUD altitude (m). For telemetry/diagnostics."""
        return self._alt_m

    def read_switch(self) -> SwitchState:
        self._drain()
        if self._last_switch is None:
            return SwitchState(active=False, pwm_us=0, timestamp=time.monotonic(),
                               mode=GuidanceMode.STANDBY)
        return self._last_switch

    def is_armed(self) -> bool:
        self._drain()
        return self._armed

    def control_ready(self) -> bool:
        """Interlock: only override the sticks if the FC is actually in the flight
        mode that matches control_mode (else our throttle mapping is wrong for the
        active mode — e.g. stabilize direct-throttle pushed into ALT_HOLD). The
        pipeline releases to the pilot when this is False. Warns once on mismatch.
        No expected mode for this control_mode -> no interlock (returns True)."""
        expected = _EXPECTED_MODE.get(self._mapping.control_mode)
        if expected is None:
            return True
        if self._current_mode == expected:
            self._interlock_warned = False
            return True
        if not self._interlock_warned:
            _log.warning("engage requested but FC flight mode=%s, need %d for "
                         "control_mode=%s — staying RELEASED to the pilot; put the FC "
                         "in that mode.", self._current_mode, expected, self._mapping.control_mode)
            self._interlock_warned = True
        return False

    def send_body_rates(self, roll_rate: float, pitch_rate: float, yaw_rate: float,
                        thrust: float) -> None:
        """guided_nogps RATE surface: command BODY RATES (rad/s) + thrust via
        SET_ATTITUDE_TARGET (type_mask 0b10000000 = ignore-attitude, identity quaternion).
        Rates are integrated by the airframe so a noisy detector box yields smooth motion
        (an absolute-attitude quaternion snaps to each frame and jitters). thrust is real
        throttle 0..1 (REQUIRES GUID_OPTIONS bit 3 — see GUID_OPTIONS_THRUST_AS_THRUST;
        without it the FC treats thrust as a climb-rate and the dive planes)."""
        tb = int(time.time() * 1000) & 0xFFFFFFFF
        self._mav.mav.set_attitude_target_send(
            tb, self._mav.target_system, self._mav.target_component, 0b10000000,
            [1.0, 0.0, 0.0, 0.0], float(roll_rate), float(pitch_rate), float(yaw_rate),
            float(_clamp(thrust, 0.0, 1.0)))

    def send_intent(self, intent: GuidanceIntent) -> None:
        """Override the AETR channels from the intent (the rest released to the
        pilot). Called by the pipeline while engaged; ZERO_INTENT maps to centred
        sticks = hold level + altitude. In stabilize with hover_learn, the throttle
        is trimmed by the adaptive-hover loop instead of the fixed guess."""
        if self._mav is None:
            return
        overrides = intent_to_rc_overrides(intent, self._mapping)
        m = self._mapping
        if m.control_mode == "stabilize" and m.hover_learn:
            overrides[m.throttle_channel] = self._adaptive_throttle(intent)
        self._send_channels(overrides)

    def _adaptive_throttle(self, intent: GuidanceIntent) -> int:
        """Companion vertical-velocity controller for STABILIZE (PI on climb rate).

        Tracks a commanded climb rate against VFR_HUD.climb:
          - `intent.vertical_rate_mps` set (DIVE constant-bearing homing) -> track
            that rate (+up). The outer framing loop in the servo integrates out the
            P-controller's steady-state error.
          - else -> hold altitude (setpoint 0): Kp damps climb immediately, Ki
            slowly trims the learned hover so it self-levels.
        When holding and telemetry is fresh, Ki trims the learned hover. A commanded
        `thrust` off 0.5 (open-loop dive, no rate given) modulates around the hover.
        Output clamped to valid PWM; learned hover clamped to [hover_min, max]."""
        m = self._mapping
        now = time.monotonic()
        dt = now - self._hover_t if self._hover_t else 0.0
        self._hover_t = now
        cmd_rate = intent.vertical_rate_mps          # +up m/s, or None
        rate_mode = cmd_rate is not None
        holding = (abs(cmd_rate) < 0.2) if rate_mode else (abs(intent.thrust - 0.5) < m.hover_learn_band)
        fresh = (now - self._climb_t) < 0.5 if self._climb_t else False
        if fresh:
            self._vfr_warned = False
        elif holding and not self._vfr_warned:
            _log.warning("adaptive hover: no fresh VFR_HUD.climb — holding at fixed "
                         "hover %d. Is VFR_HUD streamed (SR*_EXTRA2)?", int(self._hover_pwm))
            self._vfr_warned = True

        if fresh and (rate_mode or holding):
            setpoint = cmd_rate if rate_mode else 0.0
            err = setpoint - self._climb_mps
            if holding:
                # ~Holding: trim the learned hover (Ki) so it self-levels; no rate
                # integral (it would wind up against the hover trim).
                self._vrate_i = 0.0
                if 0.0 < dt < 0.3:
                    self._hover_pwm = _clamp(
                        self._hover_pwm + m.hover_learn_gain * (-self._climb_mps) * dt,
                        m.hover_min_us, m.hover_max_us,
                    )
            elif 0.0 < dt < 0.3:
                # Tracking a commanded rate: integrate the rate error so the loop
                # reaches the setpoint (kills the pure-P droop), with anti-windup.
                self._vrate_i = _clamp(self._vrate_i + m.rate_i_gain * err * dt,
                                       -m.rate_i_max_us, m.rate_i_max_us)
            out = self._hover_pwm + m.hover_learn_kp * err + self._vrate_i
        elif rate_mode:
            # No fresh climb telemetry: can't close the rate loop -> open-loop map
            # the commanded rate to a throttle offset (degraded but still descends).
            self._vrate_i = 0.0
            t = _clamp(cmd_rate / m.rate_openloop_full_mps, -1.0, 1.0)
            span = (2000 - self._hover_pwm) if t >= 0 else (self._hover_pwm - 1000)
            out = self._hover_pwm + t * span
        else:
            self._vrate_i = 0.0
            t = _clamp((intent.thrust - 0.5) / 0.5, -1.0, 1.0)
            span = (2000 - self._hover_pwm) if t >= 0 else (self._hover_pwm - 1000)
            out = self._hover_pwm + t * span
        return int(round(_clamp(out, 1000.0, 2000.0)))

    def release(self) -> None:
        """Hand every channel back to the pilot's RC radio (override value 0 =
        'use the receiver'). Called by the pipeline in STANDBY — instant manual
        handback, the core safety property of the ALT_HOLD path."""
        self._vrate_i = 0.0          # clear the rate-loop integral so a later
                                     # STANDBY->DIVE doesn't start with a stale bias
        if self._mav is None:
            return
        self._send_channels({})

    def _send_channels(self, overrides: Dict[int, int]) -> None:
        """Send RC_CHANNELS_OVERRIDE for ch1..8. Channels in `overrides` carry the
        given PWM; all others are 0 = released to the receiver. Releasing 5..8
        keeps the pilot's flight-MODE and engage switches under radio control."""
        chans = [0] * 8
        for ch, pwm in overrides.items():
            if 1 <= ch <= 8:
                chans[ch - 1] = int(pwm)
        self._mav.mav.rc_channels_override_send(
            self._mav.target_system, self._mav.target_component, *chans
        )
