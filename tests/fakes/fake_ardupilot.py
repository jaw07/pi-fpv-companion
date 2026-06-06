"""UDP loopback fake ArduCopter for integration testing the MAVLink backend.

Sends HEARTBEAT + RC_CHANNELS at 20 Hz toward the backend's bound port, and
captures any inbound RC_CHANNELS_OVERRIDE (the GPS-denied ALT_HOLD control path)
for later assertion.

This is NOT a physics simulator. It validates the wire protocol only.
Swap to real SITL or hardware by changing the backend's connection string —
no other code changes required.
"""
from __future__ import annotations
import threading
import time
from typing import List


class FakeArduCopter:
    def __init__(self, target_port: int) -> None:
        self._target_port = target_port
        self._mav = None
        self._mavutil = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.armed: bool = False
        self.rc_channels: List[int] = [1500] * 18
        self.pitch_rad: float = 0.0          # ATTITUDE.pitch to emit (+nose-up)
        self.alt: float = 0.0                # VFR_HUD.alt to emit (m)
        self.climb: float = 0.0              # VFR_HUD.climb to emit (m/s, +up)
        self.params: dict = {}               # FC parameter store (PARAM_REQUEST_READ/SET)
        self.captured_overrides: List = []   # inbound RC_CHANNELS_OVERRIDE messages
        self.custom_mode: int = 0            # current flight mode emitted in HEARTBEAT
        self.set_mode_cmds: List[int] = []   # custom_modes requested via DO_SET_MODE
        self.drop_first_mode_cmds: int = 0   # ignore the first N DO_SET_MODE (UART-drop sim)
        self.reject_mode: int | None = None  # if set, refuse to enter this mode (stays put)

    def start(self) -> None:
        from pymavlink import mavutil
        self._mavutil = mavutil
        self._mav = mavutil.mavlink_connection(
            f"udpout:127.0.0.1:{self._target_port}",
            source_system=1,
            source_component=1,
        )
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._mav is not None:
            self._mav.close()

    def _run(self) -> None:
        last_emit = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            if now - last_emit >= 0.05:
                self._emit_heartbeat()
                self._emit_rc_channels()
                self._emit_attitude()
                self._emit_vfr_hud()
                last_emit = now
            msg = self._mav.recv_match(blocking=False)
            if msg is not None:
                mt = msg.get_type()
                if mt == "RC_CHANNELS_OVERRIDE":
                    self.captured_overrides.append(msg)
                elif mt == "PARAM_REQUEST_READ":
                    self._emit_param(msg.param_id.strip("\x00"))
                elif mt == "PARAM_SET":
                    name = msg.param_id.strip("\x00")
                    self.params[name] = float(msg.param_value)   # store the write
                    self._emit_param(name)                       # echo back (ack)
                elif mt == "COMMAND_LONG" and \
                        msg.command == self._mavutil.mavlink.MAV_CMD_DO_SET_MODE:
                    requested = int(msg.param2)                  # custom_mode field
                    self.set_mode_cmds.append(requested)
                    if self.drop_first_mode_cmds > 0:
                        self.drop_first_mode_cmds -= 1           # simulate a dropped command
                    elif requested != self.reject_mode:
                        self.custom_mode = requested             # mode change takes effect
            time.sleep(0.005)

    def _emit_param(self, name: str) -> None:
        # Unknown params default to 0.0 (as a real FC would return them if they
        # exist; tests pre-seed self.params with the "wrong" current values).
        val = self.params.get(name, 0.0)
        self._mav.mav.param_value_send(name.encode(), float(val),
                                       self._mavutil.mavlink.MAV_PARAM_TYPE_REAL32, 1, 0)

    def _emit_heartbeat(self) -> None:
        base_mode = self._mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
        if self.armed:
            base_mode |= self._mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
        self._mav.mav.heartbeat_send(
            self._mavutil.mavlink.MAV_TYPE_QUADROTOR,
            self._mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA,
            base_mode,
            self.custom_mode,
            self._mavutil.mavlink.MAV_STATE_STANDBY,
        )

    def _emit_rc_channels(self) -> None:
        ch = self.rc_channels + [0] * (18 - len(self.rc_channels))
        self._mav.mav.rc_channels_send(
            0, 18,
            ch[0], ch[1], ch[2], ch[3], ch[4], ch[5], ch[6], ch[7],
            ch[8], ch[9], ch[10], ch[11], ch[12], ch[13], ch[14], ch[15],
            ch[16], ch[17],
            255,
        )

    def _emit_attitude(self) -> None:
        # roll, pitch, yaw (rad) + body rates; only pitch is consumed.
        self._mav.mav.attitude_send(0, 0.0, self.pitch_rad, 0.0, 0.0, 0.0, 0.0)

    def _emit_vfr_hud(self) -> None:
        # airspeed, groundspeed, heading, throttle%, alt, climb. Backend uses alt (AGL home),
        # climb (hover trim / flight-path angle) and groundspeed.
        self._mav.mav.vfr_hud_send(0.0, 0.0, 0, 0, float(self.alt), float(self.climb))
