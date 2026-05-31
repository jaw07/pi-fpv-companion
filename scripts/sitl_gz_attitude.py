#!/usr/bin/env python3
"""GUIDED_NOGPS comparison node: flies the FRAMING controller (attitude + thrust) via
GUIDED + SET_ATTITUDE_TARGET — the Peregrine-style quad law on the attitude-control
surface, to A/B against the STABILIZE + RC-override path (sitl_gz_track.py). Gets to
altitude first, noses down to ACQUIRE a below target, then frames + descends onto it.
The SITL has GPS so plain GUIDED accepts SET_ATTITUDE_TARGET; the real GPS-denied
aircraft uses GUIDED_NOGPS (same message). Records /work/gz_att.mp4."""
import sys, time, argparse, math
sys.path.insert(0, "/work/pi-fpv-companion/src")
import numpy as np, cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from pi_fpv_companion.detect.color import ColorBlobDetector
from pi_fpv_companion.track.iou_associator import IouAssociator
from pi_fpv_companion.track.target_filter import AlphaBetaTargetFilter
from pi_fpv_companion.guidance.framing_control import FramingConfig, FramingState, compute_framing_intent
from pi_fpv_companion.fc.ardupilot import ArduPilotBackend, ArduCopterRcMapping
from pi_fpv_companion.types import GuidanceIntent

STABILIZE, GUIDED, AP = 0, 4, 1


def euler_to_quat(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]


def arm_takeoff(mav, M, alt, home_alt, settle=18.0):
    for n, v in [("FRAME_CLASS", 1), ("FRAME_TYPE", 1), ("ARMING_CHECK", 0)]:
        mav.mav.param_set_send(mav.target_system, AP, n.encode(), float(v), M.MAV_PARAM_TYPE_INT32); time.sleep(0.4)
    print("  settling %.0fs..." % settle, flush=True); time.sleep(settle)
    mav.set_mode(GUIDED); time.sleep(1); t0 = time.time(); armed = False; last = 0
    while time.time() - t0 < 75 and not armed:
        if time.time() - last > 5:
            force = 21196 if (time.time() - t0 > 30) else 0
            mav.mav.command_long_send(mav.target_system, AP, M.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, force, 0, 0, 0, 0, 0); last = time.time()
        hb = mav.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
        if hb: armed = bool(hb.base_mode & M.MAV_MODE_FLAG_SAFETY_ARMED)
    if not armed: return False
    print("  armed; NAV_TAKEOFF", flush=True)
    mav.mav.command_long_send(mav.target_system, AP, M.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, alt)
    end = time.time() + 40
    while time.time() < end:
        v = mav.recv_match(type="VFR_HUD", blocking=True, timeout=2)
        if v and v.alt - home_alt >= alt - 0.5: break       # fully at altitude before engaging
    mav.set_mode(GUIDED); time.sleep(1.0); return True


class Att(Node):
    def __init__(self, backend, args):
        super().__init__("pifpv_att")
        self.backend, self.a = backend, args
        self.mav = backend._mav
        self.det = ColorBlobDetector(min_area_px=50)
        self.tracker = IouAssociator(max_lost_frames=25, max_match_dist_px=160.0)
        self.flt = AlphaBetaTargetFilter()
        self.fcfg = FramingConfig(720, 576, descent_pitch_fold=0.5, vert_goal=0.30, max_thrust_off=0.25, thrust_kp=0.45); self.fstate = FramingState()
        self.writer = cv2.VideoWriter("/work/gz_att.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (720, 576))
        self.frames = self.detected = 0
        self.yaw_sp = None; self.last_t = None; self.acquired = False
        self.create_subscription(Image, "/imx500/image", self.on_image, 10)

    def send_attitude(self, roll_deg, pitch_deg, yaw_rate_dps, thrust, now):
        if self.yaw_sp is None:
            self.yaw_sp = math.radians(self.backend.yaw_deg())
        dt = (now - self.last_t) if self.last_t else 0.0
        self.last_t = now
        self.yaw_sp -= math.radians(yaw_rate_dps) * dt
        q = euler_to_quat(math.radians(roll_deg), math.radians(pitch_deg), self.yaw_sp)
        self.mav.mav.set_attitude_target_send(0, self.mav.target_system, AP, 0b00000000, q, 0.0, 0.0, 0.0, thrust)

    def on_image(self, msg):
        now = time.monotonic()
        rgb = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.width, 3)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self.backend._drain()
        dets = self.det.detect(bgr)
        raw = self.tracker.consume(bgr, dets, now)
        target = self.flt.update(raw, msg.width, msg.height, now)
        th = target.detection.h if target is not None else 0.0
        if target is not None:
            self.acquired = True
            it = compute_framing_intent(target, self.fcfg, self.fstate, now, pitch_deg_measured=self.backend.pitch_deg())
            roll, pitch, yawr, thr = it.roll_deg, it.pitch_deg, it.yaw_rate_dps, it.thrust
            phase = "FRAME"
        else:
            # ACQUIRE: nose down at a steady hover to bring a below target into the FOV.
            roll, pitch, yawr, thr = 0.0, -22.0, 0.0, 0.5
            phase = "SEARCH"
        self.send_attitude(roll, pitch, yawr, thr, now)
        for d in dets:
            cv2.rectangle(bgr, (int(d.x-d.w/2), int(d.y-d.h/2)), (int(d.x+d.w/2), int(d.y+d.h/2)), (120,120,120), 1)
        if target is not None:
            t = target.detection
            cv2.rectangle(bgr, (int(t.x-t.w/2), int(t.y-t.h/2)), (int(t.x+t.w/2), int(t.y+t.h/2)), (0,255,0), 2)
        cv2.line(bgr,(360,278),(360,298),(255,255,0),1); cv2.line(bgr,(350,288),(370,288),(255,255,0),1)
        s = "ATT %s roll=%+.1f pitch=%+.1f yaw=%+.0f thr=%.2f h=%.0f" % (phase, roll, pitch, yawr, thr, th)
        cv2.putText(bgr, s, (8,22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(bgr, s, (8,22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        self.writer.write(bgr); self.frames += 1; self.detected += bool(dets)
        if self.frames % 20 == 0:
            self.get_logger().info("ATT %s f=%d det=%d tgt=%s h=%.0f roll=%+.1f pitch=%+.1f thr=%.2f" % (phase, self.frames, self.detected, target is not None, th, roll, pitch, thr))


def main():
    from pymavlink import mavutil
    M = mavutil.mavlink
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt", type=float, default=30.0); ap.add_argument("--duration", type=float, default=55.0)
    a = ap.parse_args()
    backend = ArduPilotBackend(device="tcp:127.0.0.1:5760", baud=0, switch_channel=7, track_threshold_us=1300, dive_threshold_us=1700,
                               mapping=ArduCopterRcMapping(control_mode="stabilize", hover_learn=True, hover_learn_band=0.05))
    backend.open(); mav = backend._mav; mav.wait_heartbeat(timeout=60); mav.target_component = AP
    mav.mav.request_data_stream_send(mav.target_system, AP, M.MAV_DATA_STREAM_ALL, 15, 1); backend._request_streams()
    v0 = mav.recv_match(type="VFR_HUD", blocking=True, timeout=5); home = v0.alt if v0 else 0.0
    print("arming+takeoff to %.1fm..." % a.alt, flush=True)
    if not arm_takeoff(mav, M, a.alt, home): print("FAIL takeoff"); return 1
    print("airborne (GUIDED) at altitude; settling 4s...", flush=True)
    t_settle = time.monotonic() + 4.0
    yaw0 = math.radians(backend.yaw_deg())
    while time.monotonic() < t_settle:
        q = euler_to_quat(0, 0, yaw0); mav.mav.set_attitude_target_send(0, mav.target_system, AP, 0b00000000, q, 0,0,0, 0.5); time.sleep(0.05)
    print("airborne; FRAMING (acquire -> dive).", flush=True)
    rclpy.init(); node = Att(backend, a)
    end = time.monotonic() + a.duration
    while rclpy.ok() and time.monotonic() < end: rclpy.spin_once(node, timeout_sec=0.5)
    node.writer.release(); backend.release()
    mav.set_mode(GUIDED); mav.mav.command_long_send(mav.target_system, AP, M.MAV_CMD_NAV_LAND, 0,0,0,0,0,0,0,0)
    print("frames=%d det=%d -> /work/gz_att.mp4" % (node.frames, node.detected), flush=True)
    rclpy.shutdown(); return 0


if __name__ == "__main__": sys.exit(main())
