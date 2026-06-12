"""Closed-loop camera-in-the-loop demo: Webots camera -> our pipeline -> SITL.

This closes the loop the kinematic sim and the control-surface SITL each only
cover half of: a REAL rendered camera (Webots, driving ArduPilot SITL flight
dynamics) feeds the PRODUCTION perception+guidance stack, which steers the same
vehicle by RC override — so the rendered view updates and the loop closes.

    camera frame (Webots :5599)
        -> ArucoDetector            (real detector, swappable)
        -> IouAssociator            (real tracker)
        -> AlphaBetaTargetFilter    (real filter + quality gate)
        -> compute_intent (TRACK)   (real visual servo, PI closure)
        -> safety.gate              (real safety gate)
        -> ArduPilotBackend         (real RC_CHANNELS_OVERRIDE into STABILIZE)
        -> ArduCopter SITL          (drives the Webots vehicle) -> new frame

Everything between the camera and the FC is the shipping code, unchanged. An
annotated copy of the camera view is written to a video so the run is watchable.

Bring-up (three processes):
  1. Webots:  open libraries/SITL/examples/Webots_Python/worlds/iris_camera.wbt
              (controller binds SITL on udp:9002, streams camera on tcp:5599)
  2. SITL:    cd ~/ardupilot && Tools/autotest/sim_vehicle.py -v ArduCopter \\
                  --model JSON --add-param-file=\\
                  libraries/SITL/examples/Webots_Python/params/iris.parm \\
                  -I0 --no-mavproxy
  3. this:    .venv/bin/python scripts/sitl_webots_track.py
"""
from __future__ import annotations
import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from pi_fpv_companion.config import load
from pi_fpv_companion.camera.webots import WebotsCamera
from pi_fpv_companion.detect.aruco import ArucoDetector
from pi_fpv_companion.track.iou_associator import IouAssociator
from pi_fpv_companion.track.target_filter import AlphaBetaTargetFilter
from pi_fpv_companion.guidance.visual_servo import ClosureState, compute_intent
from pi_fpv_companion.guidance.safety import gate
from pi_fpv_companion.fc.ardupilot import ArduPilotBackend, ArduCopterRcMapping
from pi_fpv_companion.types import GuidanceMode, SwitchState, GuidanceIntent

STABILIZE, GUIDED = 0, 4
AP = 1


def arm_and_takeoff(mav, M, alt_m: float, home_alt: float) -> bool:
    """GUIDED arm + NAV_TAKEOFF (reliable), then hand to STABILIZE for RC override."""
    mav.set_mode(GUIDED)
    armed, end, last = False, time.time() + 90, 0.0
    while time.time() < end and not armed:
        if time.time() - last > 4:
            mav.mav.command_long_send(mav.target_system, AP,
                                      M.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
            last = time.time()
        hb = mav.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
        if hb:
            armed = bool(hb.base_mode & M.MAV_MODE_FLAG_SAFETY_ARMED)
    if not armed:
        return False
    mav.mav.command_long_send(mav.target_system, AP, M.MAV_CMD_NAV_TAKEOFF,
                              0, 0, 0, 0, 0, 0, 0, alt_m)
    end = time.time() + 60
    while time.time() < end:
        v = mav.recv_match(type="VFR_HUD", blocking=True, timeout=2)
        if v and v.alt - home_alt >= alt_m - 0.5:
            break
    for i in range(1, 7):                       # make a STABILIZE flight-mode slot
        mav.mav.param_set_send(mav.target_system, AP, f"FLTMODE{i}".encode(),
                               float(STABILIZE), M.MAV_PARAM_TYPE_INT32)
    mav.set_mode(STABILIZE)
    time.sleep(1.0)
    return True


def draw_overlay(img, dets, target, muted, reason, intent) -> None:
    for d in dets:                              # all detections, faint
        cv2.rectangle(img, (int(d.x - d.w / 2), int(d.y - d.h / 2)),
                      (int(d.x + d.w / 2), int(d.y + d.h / 2)), (120, 120, 120), 1)
    if target is not None:                      # the locked target, bold
        t = target.detection
        c = (0, 0, 255) if muted else (0, 255, 0)
        cv2.rectangle(img, (int(t.x - t.w / 2), int(t.y - t.h / 2)),
                      (int(t.x + t.w / 2), int(t.y + t.h / 2)), c, 2)
        cv2.circle(img, (int(t.x), int(t.y)), 4, c, -1)
    h, w = img.shape[:2]
    cv2.line(img, (w // 2, h // 2 - 10), (w // 2, h // 2 + 10), (255, 255, 0), 1)
    cv2.line(img, (w // 2 - 10, h // 2), (w // 2 + 10, h // 2), (255, 255, 0), 1)
    status = f"TRACK {'MUTED:' + reason if muted else 'ACTIVE'}  " \
             f"yaw={intent.yaw_rate_dps:+.0f}dps pitch={intent.pitch_deg:+.1f}deg"
    cv2.putText(img, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)


def main() -> int:
    from pymavlink import mavutil
    M = mavutil.mavlink
    ap = argparse.ArgumentParser()
    ap.add_argument("--connect", default="tcp:127.0.0.1:5760")
    ap.add_argument("--camera-host", default="127.0.0.1")
    ap.add_argument("--camera-port", type=int, default=5599)
    ap.add_argument("--alt", type=float, default=2.0, help="takeoff alt (low: forward cam must see the low markers)")
    ap.add_argument("--marker-id", type=int, default=0)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--settle", type=float, default=20.0, help="EKF settle seconds")
    ap.add_argument("--out", default=str(_root / "var" / "webots_track.mp4"))
    args = ap.parse_args()

    cfg = load(_root / "config" / "imx500.yaml")
    # Servo gains from the flight config, but the FRAME size must match the Webots
    # camera (640x480), not the IMX500 frame — pixel errors are resolution-relative.
    servo = replace(cfg.servo, frame_width=640, frame_height=480)

    backend = ArduPilotBackend(
        device=args.connect, baud=0, switch_channel=7,
        track_threshold_us=1300, dive_threshold_us=1700,
        mapping=ArduCopterRcMapping(control_mode="stabilize", hover_learn=True,
                                    hover_learn_band=0.05),
    )
    backend.open()
    mav = backend._mav
    print(f"connecting MAVLink {args.connect} ...")
    mav.wait_heartbeat(timeout=90)
    mav.target_component = AP
    mav.mav.request_data_stream_send(mav.target_system, AP, M.MAV_DATA_STREAM_ALL, 10, 1)
    backend._request_streams()
    print(f"  settling {args.settle:.0f}s ...")
    time.sleep(args.settle)
    v0 = mav.recv_match(type="VFR_HUD", blocking=True, timeout=5)
    home_alt = v0.alt if v0 else 0.0

    print(f"  arming + takeoff to {args.alt:.1f} m ...")
    if not arm_and_takeoff(mav, M, args.alt, home_alt):
        print("FAIL: never armed/took off"); backend.close(); return 1
    print("  airborne; STABILIZE. Opening camera stream ...")

    cam = WebotsCamera(args.camera_host, args.camera_port)
    cam.open()
    detector = ArucoDetector(only_id=args.marker_id)
    tracker = IouAssociator(max_lost_frames=15, max_match_dist_px=80.0)
    flt = AlphaBetaTargetFilter()
    closure = ClosureState()
    switch = SwitchState(active=True, pwm_us=1500, timestamp=0.0, mode=GuidanceMode.TRACK)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (640, 480))

    print(f"  tracking marker {args.marker_id}; recording to {args.out}")
    t_end = time.monotonic() + args.duration
    frames = detected = active = 0
    try:
        for fb in cam.frames():
            now = fb.timestamp
            backend._drain()                    # feed ATTITUDE/VFR_HUD to the backend
            dets = detector.detect(fb.image)
            raw = tracker.consume(fb.image, dets, now)
            target = flt.update(raw, fb.width, fb.height, now)
            intent = (compute_intent(target, servo, GuidanceMode.TRACK, closure=closure)
                      if target is not None else GuidanceIntent(0, 0, 0, 0.5, now))
            res = gate(intent, target, switch, backend.is_armed(), now, cfg.safety)
            backend.send_intent(res.intent)

            draw_overlay(fb.image, dets, target, res.muted, res.reason, res.intent)
            writer.write(fb.image)
            frames += 1
            detected += bool(dets)
            active += (target is not None and not res.muted)
            if frames % 20 == 0:
                print(f"    f={frames} det={detected} active={active} "
                      f"reason='{res.reason}' yaw={res.intent.yaw_rate_dps:+.0f} "
                      f"pitch={res.intent.pitch_deg:+.1f}")
            if time.monotonic() > t_end:
                break
    finally:
        backend.send_intent(GuidanceIntent(0, 0, 0, 0.5, time.monotonic()))
        backend.release()
        writer.release()
        cam.close()
        mav.set_mode(GUIDED)
        mav.mav.command_long_send(mav.target_system, AP, M.MAV_CMD_NAV_LAND,
                                  0, 0, 0, 0, 0, 0, 0, 0)
        backend.close()

    print(f"\nframes={frames} with-detection={detected} guidance-active={active}")
    print(f"video: {args.out}")
    return 0 if detected > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
