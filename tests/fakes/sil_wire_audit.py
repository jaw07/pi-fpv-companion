"""Software-in-the-loop wire audit: census every MAVLink send the companion makes.

Drives the REAL Pipeline + REAL ArduPilotBackend against the UDP FakeArduCopter,
which — unlike a bench FC — can be armed, placed in an arbitrary flight mode, and
have its RC channels moved. That reaches the states that matter for the STANDBY
command contract and cannot otherwise be tested without flying.

Shared by `scripts/sil_standby_audit.py` (CLI report) and
`tests/test_standby_wire_contract.py` (assertions), so both observe exactly the
same thing.
"""
from __future__ import annotations

import socket
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from pi_fpv_companion.safety_contract import ContractChecker, ContractConfig

# Sends that can move the aircraft. Everything else (GCS heartbeat, stream
# subscriptions, param traffic) is link housekeeping and touches no control input.
CONTROL_AFFECTING = frozenset(
    {"rc_channels_override_nonzero", "set_attitude_target", "do_set_mode"}
)


@dataclass
class WireAudit:
    checker: ContractChecker
    census: Counter = field(default_factory=Counter)
    # (kind, switch mode name, armed) for every control-affecting send
    log: List[Tuple[str, str, bool]] = field(default_factory=list)

    def control_events(self, mode: Optional[str] = None) -> List[Tuple[str, str, bool]]:
        return [e for e in self.log if mode is None or e[1] == mode]

    def count(self, kind: str) -> int:
        return self.census[kind]


def free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def tap_backend(fc, audit: WireAudit) -> None:
    """Wrap every outbound MAVLink send on the backend's connection.

    CRITICAL: these wrappers run INSIDE the backend's `_send_lock` — every send site
    holds it. So the tap must NEVER call back into a backend method that drains or
    sends: `read_switch()`/`is_armed()` drain, `_drain()` calls `_service_mode()`, and
    that can call `_send_mode()`, which re-takes the non-reentrant `_send_lock` and
    deadlocks the calling thread. Read the backend's CACHED state attributes instead.

    This is not hypothetical — the first version of this tap deadlocked the control
    thread on the very first release() and then reported "nothing transmitted" for
    every scenario: a green result from an instrument that had stopped measuring.
    """
    from pymavlink import mavutil

    real = fc._mav.mav
    lock = threading.Lock()

    def observe(kind: str, a: tuple) -> None:
        sw = fc._last_switch            # cached by the last drain; no re-entry
        armed = bool(fc._armed)
        pwm = sw.pwm_us if sw is not None else 0
        mode = sw.mode.name if sw is not None else "UNKNOWN"
        t = time.time()
        with lock:
            # State first: the checker ignores events until switch + armed are known.
            audit.checker.on_rc_channels(t, pwm)
            audit.checker.on_heartbeat(t, armed)
            if kind.startswith("rc_channels_override"):
                audit.checker.on_rc_override(t, list(a[2:10]))
            elif kind == "set_attitude_target":
                audit.checker.on_attitude_target(t)
            elif kind == "do_set_mode":
                audit.checker.on_set_mode(t, int(a[5]))
            audit.census[kind] += 1
            if kind in CONTROL_AFFECTING:
                audit.log.append((kind, mode, armed))

    def wrap(orig, classify):
        def inner(*a, **k):
            kind = classify(a)
            if kind:
                observe(kind, a)
            return orig(*a, **k)
        return inner

    def classify_cmd(a):
        if len(a) > 2 and a[2] == mavutil.mavlink.MAV_CMD_DO_SET_MODE:
            return "do_set_mode"
        if len(a) > 2 and a[2] == mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL:
            return "set_message_interval"
        return "command_long_other"

    real.rc_channels_override_send = wrap(
        real.rc_channels_override_send,
        lambda a: ("rc_channels_override_nonzero" if any(a[2:10])
                   else "rc_channels_override_zero"))
    real.set_attitude_target_send = wrap(
        real.set_attitude_target_send, lambda a: "set_attitude_target")
    real.command_long_send = wrap(real.command_long_send, classify_cmd)
    real.heartbeat_send = wrap(real.heartbeat_send, lambda a: "gcs_heartbeat")
    real.request_data_stream_send = wrap(
        real.request_data_stream_send, lambda a: "request_data_stream")
    real.param_request_read_send = wrap(
        real.param_request_read_send, lambda a: "param_request_read")
    real.param_set_send = wrap(real.param_set_send, lambda a: "param_set")


def run_scenario(*, armed: bool, fc_mode: int, ch7_us: int, seconds: float = 2.0,
                 engage_us: Optional[int] = None, auto_guided: bool = True) -> WireAudit:
    """Run the real pipeline against the fake FC and return the wire census.

    `engage_us` (optional) drives a STANDBY -> engaged -> STANDBY cycle across the run,
    one third of `seconds` in each phase, to exercise the mode-restore edge.
    """
    from tests.fakes.fake_ardupilot import FakeArduCopter
    from pi_fpv_companion.camera.synthetic import SyntheticCamera
    from pi_fpv_companion.fc.ardupilot import ArduPilotBackend, ArduCopterRcMapping
    from pi_fpv_companion.guidance.rate_control import RateConfig
    from pi_fpv_companion.guidance.safety import SafetyConfig
    from pi_fpv_companion.guidance.visual_servo import ServoConfig
    from pi_fpv_companion.pipeline import Pipeline
    from pi_fpv_companion.track.multi_target import MultiObjectTracker

    port = free_port()
    fc = ArduPilotBackend(device=f"udpin:127.0.0.1:{port}", baud=0, switch_channel=7,
                          track_threshold_us=1300, dive_threshold_us=1700,
                          auto_guided=auto_guided,
                          mapping=ArduCopterRcMapping(control_mode="guided_nogps"))
    fake = FakeArduCopter(target_port=port)
    fake.armed = armed
    fake.custom_mode = fc_mode
    fake.rc_channels = [1500] * 18
    fake.rc_channels[6] = ch7_us              # ch7 (1-indexed) = engage switch
    fc.open()
    fake.start()
    fc.wait_ready(timeout=5)

    audit = WireAudit(checker=ContractChecker(cfg=ContractConfig(switch_channel=7)))
    tap_backend(fc, audit)

    cam = SyntheticCamera(width=720, height=576, fps=22)
    servo = ServoConfig(frame_width=720, frame_height=576, max_yaw_rate_dps=60.0,
                        max_pitch_deg=15.0, pixel_deadzone_px=10.0, yaw_p_gain=0.3,
                        yaw_ff_gain=0.0, desired_bbox_frac=0.30, closure_p_gain=50.0)
    pipe = Pipeline(cam, MultiObjectTracker(iou_threshold=0.3, max_lost_frames=8),
                    servo, SafetyConfig(watchdog_timeout_s=1.0, require_armed=True), fc,
                    display=lambda *a, **k: None,      # exercise the decoupled control loop
                    rate_cfg=RateConfig(720, 576))

    t = threading.Thread(target=pipe.run, daemon=True)
    t.start()
    try:
        if engage_us is not None:
            time.sleep(seconds / 3)
            fake.rc_channels[6] = engage_us       # pilot engages
            time.sleep(seconds / 3)
            fake.rc_channels[6] = ch7_us          # pilot disengages
            time.sleep(seconds / 3)
        else:
            time.sleep(seconds)
    finally:
        pipe.stop()
        t.join(timeout=3)
        fake.stop()
        fc.close()
    return audit
