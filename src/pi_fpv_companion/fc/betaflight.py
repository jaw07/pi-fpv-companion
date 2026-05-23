"""Betaflight backend over MSP.

Betaflight has no native guidance API. The only injection point is MSP_SET_RAW_RC,
which overrides the receiver's stick values. We translate the visual-servo intent
into stick channel values, assuming the FC is in ANGLE mode (pitch/roll sticks
command target angles rather than rates).

This is necessarily approximate — Betaflight doesn't accept velocity targets, only
angle/rate setpoints. The mapping has tunable gains that need bench calibration.
The sign of pitch_us_per_mps is config-dependent (TX layout / RC channel mapping).

Polling: read_switch() and is_armed() trigger MSP_RC and MSP_STATUS requests if
enough time has elapsed since the last poll, drain any pending responses, and
return cached state. First call after open() may return defaults; second call
returns fresh data once the FC has responded.

`serial` is imported lazily (or substituted via serial_factory) so the module is
importable without pyserial and testable without a real serial port.
"""
from __future__ import annotations
import struct
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from pi_fpv_companion.fc.msp import HEADER_RESPONSE, MspDecoder, encode
from pi_fpv_companion.types import GuidanceIntent, GuidanceMode, SwitchState


# MSP command IDs
MSP_STATUS = 101
MSP_RC = 105
MSP_SET_RAW_RC = 200

_POLL_INTERVAL_S = 0.05      # 20 Hz max poll rate per command
_ARM_FLAG_BIT = 1 << 0       # bit 0 of MSP_STATUS.flag is the ARM box on default Betaflight setups


@dataclass(frozen=True)
class BetaflightMapping:
    """Tunable mapping from the ATTITUDE intent to MSP_SET_RAW_RC sticks
    (AETR order). Assumes Betaflight is in ANGLE mode, where roll/pitch stick
    deflection commands a target lean ANGLE — which is exactly our intent's
    domain, so the mapping is now a clean angle->stick scale.

    Neutral 1500 us; range typically 1000-2000 us. Sign of the per-deg gains
    is TX/rcmap dependent and must be bench-calibrated.

    Demo-only path: BF failsafe keys off the RX link, not MSP — a hung Pi at
    full override does NOT failsafe. Leave throttle/arm on the physical RX.
    """
    roll_us_per_deg: float
    pitch_us_per_deg: float
    yaw_us_per_dps: float
    throttle_us_per_thrust: float    # thrust 0..1 -> elevator-of-throttle delta
    throttle_neutral_us: int = 1500
    stick_min_us: int = 1000
    stick_max_us: int = 2000


def _clamp_us(v: float, mapping: BetaflightMapping) -> int:
    return int(max(mapping.stick_min_us, min(mapping.stick_max_us, v)))


def intent_to_sticks(
    intent: GuidanceIntent, mapping: BetaflightMapping
) -> tuple[int, int, int, int]:
    """Return (aileron, elevator, throttle, rudder) in microseconds, AETR order.

    roll/pitch angle -> aileron/elevator deflection (ANGLE mode = stick is the
    angle setpoint); yaw RATE -> rudder; thrust (0..1, 0.5 neutral) -> throttle.
    """
    aileron = _clamp_us(1500 + mapping.roll_us_per_deg * intent.roll_deg, mapping)
    elevator = _clamp_us(1500 + mapping.pitch_us_per_deg * intent.pitch_deg, mapping)
    throttle = _clamp_us(
        mapping.throttle_neutral_us
        + mapping.throttle_us_per_thrust * (intent.thrust - 0.5),
        mapping,
    )
    rudder = _clamp_us(1500 + mapping.yaw_us_per_dps * intent.yaw_rate_dps, mapping)
    return aileron, elevator, throttle, rudder


class BetaflightBackend:
    """MSP-over-serial backend. Conforms to the FlightController Protocol."""

    def __init__(
        self,
        device: str,
        baud: int,
        switch_channel: int,
        switch_threshold_us: int,
        mapping: BetaflightMapping,
        serial_factory: Optional[Callable[[], object]] = None,
    ) -> None:
        self._device = device
        self._baud = baud
        self._switch_channel = switch_channel
        self._switch_threshold_us = switch_threshold_us
        self._mapping = mapping
        self._serial_factory = serial_factory
        self._serial = None
        self._decoder = MspDecoder(accept=HEADER_RESPONSE)
        self._last_poll: Dict[int, float] = {}
        self._last_switch: Optional[SwitchState] = None
        self._armed: bool = False

    def open(self) -> None:
        if self._serial_factory is not None:
            self._serial = self._serial_factory()
        else:
            import serial  # lazy
            self._serial = serial.Serial(self._device, self._baud, timeout=0.0)

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def _request_if_due(self, cmd: int) -> None:
        now = time.monotonic()
        if now - self._last_poll.get(cmd, 0.0) >= _POLL_INTERVAL_S:
            self._serial.write(encode(cmd))
            self._last_poll[cmd] = now

    def _drain(self) -> None:
        if self._serial is None:
            return
        n = getattr(self._serial, "in_waiting", 0)
        if n <= 0:
            return
        data = self._serial.read(n)
        if not data:
            return
        for cmd, payload in self._decoder.feed(data):
            self._handle_response(cmd, payload)

    def _handle_response(self, cmd: int, payload: bytes) -> None:
        if cmd == MSP_RC:
            n_chans = len(payload) // 2
            if n_chans >= self._switch_channel:
                offset = (self._switch_channel - 1) * 2
                pwm = struct.unpack_from("<H", payload, offset)[0]
                active = pwm > self._switch_threshold_us
                # Betaflight demo path is a 2-state engage: on -> TRACK.
                self._last_switch = SwitchState(
                    active=active,
                    pwm_us=pwm,
                    timestamp=time.monotonic(),
                    mode=GuidanceMode.TRACK if active else GuidanceMode.STANDBY,
                )
        elif cmd == MSP_STATUS:
            # cycleTime u16, i2cErrCnt u16, sensors u16, flag u32, currentSet u8
            if len(payload) >= 10:
                flag = struct.unpack_from("<I", payload, 6)[0]
                self._armed = bool(flag & _ARM_FLAG_BIT)

    def read_switch(self) -> SwitchState:
        self._request_if_due(MSP_RC)
        self._drain()
        if self._last_switch is None:
            return SwitchState(active=False, pwm_us=0, timestamp=time.monotonic(),
                               mode=GuidanceMode.STANDBY)
        return self._last_switch

    def is_armed(self) -> bool:
        self._request_if_due(MSP_STATUS)
        self._drain()
        return self._armed

    def release(self) -> None:
        # Best-effort handback: neutral sticks. BF failsafe keys off the RX link,
        # not MSP, so true handback needs the receiver — demo path only.
        if self._serial is not None:
            self._serial.write(encode(MSP_SET_RAW_RC, struct.pack("<HHHH", 1500, 1500, 1500, 1500)))

    def send_intent(self, intent: GuidanceIntent) -> None:
        a, e, t, r = intent_to_sticks(intent, self._mapping)
        # MSP_SET_RAW_RC takes channels as little-endian u16. AETR order matches default
        # Betaflight rcmap. The companion is responsible for any ordering remapping.
        payload = struct.pack("<HHHH", a, e, t, r)
        self._serial.write(encode(MSP_SET_RAW_RC, payload))
