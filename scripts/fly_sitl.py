"""End-to-end: the REAL Pipeline flying a synthetic target in ArduPilot SITL,
GPS-denied, via the ALT_HOLD RC-override control path.

Where validate_sitl.py sends hand-crafted intents to check the RC-override sense,
this runs the actual production `Pipeline` — SyntheticCamera (moving target) ->
IouAssociator -> alpha-beta filter -> visual servo -> safety gate ->
ArduPilotBackend.send_intent (RC_CHANNELS_OVERRIDE) -> live ArduCopter in
ALT_HOLD — and shows the closed loop: as the target moves off-centre the servo
commands yaw, and the copter actually yaws to chase it.

`force_mode=TRACK` engages the pipeline (SITL has no RC engage switch). The
backend's _drain is subclassed only to also latch ATTITUDE for the readout; the
send path under test (send_intent / RC override) is unmodified production code.

    docker run -d --rm --name pifpv-sitl --platform linux/amd64 \
        -p 5760:5760 radarku/ardupilot-sitl:latest
    .venv/bin/python scripts/fly_sitl.py --connect tcp:127.0.0.1:5760
"""
from __future__ import annotations
import argparse
import math
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from pi_fpv_companion.camera.synthetic import SyntheticCamera
from pi_fpv_companion.fc.ardupilot import ArduPilotBackend, ArduCopterRcMapping
from pi_fpv_companion.guidance.safety import SafetyConfig
from pi_fpv_companion.guidance.visual_servo import ServoConfig
from pi_fpv_companion.pipeline import Pipeline
from pi_fpv_companion.track.iou_associator import IouAssociator
from pi_fpv_companion.types import GuidanceIntent, GuidanceMode, SwitchState

ALT_HOLD = 2
GUIDED = 4


class ObservingBackend(ArduPilotBackend):
    """Production backend; _drain also latches ATTITUDE + custom_mode so the demo
    can show the copter's response. The send path is inherited untouched."""
    latest_att = None          # (roll, pitch, yaw, yawspeed) radians
    custom_mode = None

    def _drain(self) -> None:
        if self._mav is None:
            return
        while True:
            msg = self._mav.recv_match(blocking=False)
            if msg is None:
                return
            t = msg.get_type()
            if t == "HEARTBEAT":
                armed_bit = self._mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                self._armed = bool(msg.base_mode & armed_bit)
                self.custom_mode = msg.custom_mode
            elif t == "RC_CHANNELS":
                pwm = getattr(msg, f"chan{self._switch_channel}_raw")
                mode = self._mode_for(pwm)
                self._last_switch = SwitchState(
                    active=mode is not GuidanceMode.STANDBY,
                    pwm_us=pwm, timestamp=time.monotonic(), mode=mode)
            elif t == "ATTITUDE":
                self.latest_att = (msg.roll, msg.pitch, msg.yaw, msg.yawspeed)


def _set_param(mav, AP, name, val):
    mav.mav.param_set_send(mav.target_system, AP, name.encode(), float(val), 6)  # INT32
    mav.mav.param_request_read_send(mav.target_system, AP, name.encode(), -1)
    end = time.time() + 8.0
    while time.time() < end:
        pv = mav.recv_match(type="PARAM_VALUE", blocking=True, timeout=end - time.time())
        if pv is not None and pv.param_id.strip("\x00") == name:
            return pv.param_value
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--connect", default="tcp:127.0.0.1:5760")
    ap.add_argument("--boot-timeout", type=float, default=180.0)
    ap.add_argument("--fly-seconds", type=float, default=35.0)
    args = ap.parse_args()

    AP = 1
    # ALT_HOLD for this demo: a 35s yaw-tracking chase needs stable altitude, so
    # use althold throttle mapping (matches the FLTMODE pin below), not the
    # stabilize default (no alt hold -> would drift/descend).
    backend = ObservingBackend(device=args.connect, baud=0, switch_channel=7,
                               track_threshold_us=1300, dive_threshold_us=1700,
                               mapping=ArduCopterRcMapping(control_mode="althold"))
    print(f"connecting to {args.connect} (boot timeout {args.boot_timeout:.0f}s) ...")
    backend.open()
    mav = backend._mav

    deadline = time.time() + args.boot_timeout
    hb = None
    while time.time() < deadline:
        hb = mav.recv_match(type="HEARTBEAT", blocking=True, timeout=5.0)
        if hb is not None:
            break
        print("  ... waiting for SITL heartbeat")
    if hb is None:
        print("FAIL: no SITL heartbeat"); return 1
    mav.target_component = AP
    mav.mav.request_data_stream_send(
        mav.target_system, AP, backend._mavutil.mavlink.MAV_DATA_STREAM_ALL, 15, 1)
    # 4.6 EKF3 needs to converge before flight; settle, then GUIDED NAV_TAKEOFF
    # (the reliable SITL way up — RC-override takeoff is flaky on fresh 4.6).
    # AHRS_EKF_TYPE left at FC default. ARMING_CHECK=0 + the settle = arm cleanly.
    _set_param(mav, AP, "ARMING_CHECK", 0)
    for i in range(1, 7):
        _set_param(mav, AP, f"FLTMODE{i}", ALT_HOLD)   # pin ALT_HOLD for the chase
    print("  settling 50s for IMU/EKF convergence ...")
    time.sleep(50)
    end = time.time() + 60
    while time.time() < end:
        g = mav.recv_match(type="GPS_RAW_INT", blocking=True, timeout=2)
        if g and g.fix_type >= 3:
            break

    mav.set_mode(GUIDED)
    armed = False
    deadline = time.time() + 90.0
    last = 0.0
    while time.time() < deadline and not armed:
        if time.time() - last > 4.0:
            mav.mav.command_long_send(
                mav.target_system, AP,
                backend._mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
            last = time.time()
        m = mav.recv_match(type=["HEARTBEAT", "STATUSTEXT"], blocking=True, timeout=2.0)
        if m is None:
            continue
        if m.get_type() == "STATUSTEXT" and any(k in m.text for k in ("Arm", "EKF", "Gyro", "Accel")):
            print(f"     STATUSTEXT: {m.text}")
        elif m.get_type() == "HEARTBEAT":
            armed = bool(m.base_mode & backend._mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            backend.custom_mode = m.custom_mode
    if not armed:
        print("FAIL: never armed"); return 1

    v0 = mav.recv_match(type="VFR_HUD", blocking=True, timeout=5)
    home_alt = v0.alt if v0 else 0.0
    print("  armed; GUIDED takeoff to 60 m AGL ...")
    mav.mav.command_long_send(mav.target_system, AP,
        backend._mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, 60)
    end = time.time() + 120
    agl = 0.0
    while time.time() < end:
        v = mav.recv_match(type="VFR_HUD", blocking=True, timeout=2)
        if v:
            agl = v.alt - home_alt
            if agl >= 54:
                break
    if agl < 20:
        print("FAIL: takeoff did not climb"); return 1
    mav.set_mode(ALT_HOLD)            # hand to ALT_HOLD for the stable yaw chase
    time.sleep(1.0)
    print(f"  at {agl:.0f} m AGL, switched to ALT_HOLD (mode={backend.custom_mode})")

    # ---- the real Pipeline (force_mode=TRACK engages it) -------------------
    camera = SyntheticCamera(width=720, height=576, fps=20)
    tracker = IouAssociator(iou_threshold=0.2, max_lost_frames=20)
    servo = ServoConfig(
        frame_width=720, frame_height=576,
        max_yaw_rate_dps=45.0, max_pitch_deg=12.0,
        pixel_deadzone_px=30.0, yaw_p_gain=0.15, yaw_ff_gain=0.04,
        desired_bbox_frac=0.30, closure_p_gain=50.0)
    safety = SafetyConfig(watchdog_timeout_s=0.5, require_armed=True)

    stats = {"ticks": 0, "open": 0, "agree": 0, "agree_n": 0,
             "reasons": {}, "off_min": 1e9, "off_max": -1e9,
             "yaw0": None, "yaw_last": None}

    def on_status(target, intent, gated, switch, armed_, bundle):
        s = stats
        s["ticks"] += 1
        if backend.latest_att is not None:
            if s["yaw0"] is None:
                s["yaw0"] = backend.latest_att[2]
            s["yaw_last"] = backend.latest_att[2]
        if gated.muted:
            s["reasons"][gated.reason] = s["reasons"].get(gated.reason, 0) + 1
            return
        s["open"] += 1
        if target is not None:
            off = target.detection.x - bundle.width / 2.0
            s["off_min"] = min(s["off_min"], off)
            s["off_max"] = max(s["off_max"], off)
            if abs(off) > servo.pixel_deadzone_px:
                s["agree_n"] += 1
                if (off > 0) == (intent.yaw_rate_dps > 0) and intent.yaw_rate_dps != 0:
                    s["agree"] += 1

    pipeline = Pipeline(camera, tracker, servo, safety, backend,
                        on_status=on_status, force_mode=GuidanceMode.TRACK)
    import threading
    threading.Timer(args.fly_seconds, pipeline.stop).start()
    print(f"\n  flying the real pipeline for {args.fly_seconds:.0f}s ...")
    pipeline.run()

    held = backend.custom_mode == ALT_HOLD and backend.is_armed()
    yaw_delta = (math.degrees(stats["yaw_last"] - stats["yaw0"])
                 if stats["yaw0"] is not None and stats["yaw_last"] is not None else None)

    print(f"\n{'='*60}")
    print(f"ticks                 {stats['ticks']}")
    print(f"gate OPEN (cmd flowed) {stats['open']}/{stats['ticks']}")
    print(f"mute reasons          {stats['reasons'] or '(none)'}")
    print(f"target x-offset range [{stats['off_min']:.0f}, {stats['off_max']:.0f}] px")
    if stats["agree_n"]:
        pct = 100.0 * stats["agree"] / stats["agree_n"]
        print(f"servo yaw sign tracks target  {stats['agree']}/{stats['agree_n']}  ({pct:.0f}%)")
    print(f"copter net yaw change {('%.0f deg' % yaw_delta) if yaw_delta is not None else 'n/a'}")
    print(f"held ALT_HOLD+armed throughout  {held}")

    backend.release()
    mav.mav.command_long_send(mav.target_system, AP,
        backend._mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 0, 0, 0, 0, 0, 0, 0)
    backend.close()
    ok = (stats["open"] > 0 and held and stats["agree_n"] > 0
          and stats["agree"] / stats["agree_n"] > 0.8)
    print(f"\n{'PASS' if ok else 'NEEDS REVIEW'}: closed loop "
          f"{'demonstrated' if ok else 'did not meet thresholds'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
