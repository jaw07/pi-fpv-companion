#!/usr/bin/env python3
"""GUIDED_NOGPS body-rate node — drives the PRODUCTION rate law.

This node is a thin Gazebo/ROS harness around the shipped production controller
`pi_fpv_companion.guidance.rate_control.compute_rate_intent` (NOT a private copy of the
law). It exercises the exact code path the aircraft flies: ColorBlob detect -> IoU track
-> alpha-beta filter -> compute_rate_intent(mode) -> backend.send_body_rates. It runs a
TRACK phase (follow + hold range, no descent) for `--track` seconds, then switches to DIVE
(commit) so both production branches are validated. Records /work/gz_rate.mp4."""
import sys, time, argparse, math
sys.path.insert(0, "/work/pi-fpv-companion/src")
import numpy as np, cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from pi_fpv_companion.detect.color import ColorBlobDetector
from pi_fpv_companion.track.iou_associator import IouAssociator
from pi_fpv_companion.track.target_filter import AlphaBetaTargetFilter
from pi_fpv_companion.fc.ardupilot import ArduPilotBackend, ArduCopterRcMapping
from pi_fpv_companion.guidance.rate_control import RateConfig, RateState, compute_rate_intent
from pi_fpv_companion.types import GuidanceMode

STABILIZE, GUIDED, GUIDED_NOGPS, AP = 0, 4, 20, 1
W, H = 720, 576


def clamp(v, lo, hi): return lo if v < lo else hi if v > hi else v


def drain_for(backend, secs):
    # Continuously drain the backend for `secs` (mirrors the production pipeline, which ticks
    # the FC link every frame). Keeps _armed / alt fresh so the disarmed-only AGL home capture
    # freezes at the true ground/arming altitude instead of re-homing at altitude.
    t = time.monotonic()
    while time.monotonic() - t < secs:
        backend._drain(); time.sleep(0.02)


def arm_takeoff(backend, mav, M, alt, settle=18.0):
    # GUID_OPTIONS bit3 (=8) SetAttitudeTarget_ThrustAsThrust: make the thrust field REAL throttle.
    for n, v in [("FRAME_CLASS", 1), ("FRAME_TYPE", 1), ("ARMING_CHECK", 0), ("GUID_OPTIONS", 8)]:
        mav.mav.param_set_send(mav.target_system, AP, n.encode(), float(v), M.MAV_PARAM_TYPE_INT32); drain_for(backend, 0.4)
    print("  settling %.0fs..." % settle, flush=True); drain_for(backend, settle)
    mav.set_mode(GUIDED); drain_for(backend, 1); t0 = time.time(); last = 0
    while time.time() - t0 < 75 and not backend.is_armed():
        if time.time() - last > 5:
            force = 21196 if (time.time() - t0 > 30) else 0
            mav.mav.command_long_send(mav.target_system, AP, M.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, force, 0, 0, 0, 0, 0); last = time.time()
        drain_for(backend, 0.3)
    if not backend.is_armed(): return False
    print("  armed; NAV_TAKEOFF", flush=True)
    mav.mav.command_long_send(mav.target_system, AP, M.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, alt)
    end = time.time() + 40
    while time.time() < end:
        drain_for(backend, 0.1)
        if backend.agl_m() >= alt - 0.5: break   # AGL now valid (home frozen at ground)
    mav.set_mode(GUIDED_NOGPS); drain_for(backend, 1.0); return True   # rate control in GUIDED_NOGPS


class Rate(Node):
    def __init__(self, backend, args):
        super().__init__("pifpv_rate")
        self.backend, self.a = backend, args; self.mav = backend._mav
        self.det = ColorBlobDetector(min_area_px=50)
        self.tracker = IouAssociator(max_lost_frames=25, max_match_dist_px=160.0)
        self.flt = AlphaBetaTargetFilter()
        self.cfg = RateConfig(frame_width=W, frame_height=H)
        self.state = RateState()
        self.track_secs = args.track          # TRACK (range-hold) before committing to DIVE
        self.t0 = None                        # first control-frame time
        self.dive_announced = False
        self.writer = cv2.VideoWriter("/work/gz_rate.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (W, H))
        self.frames = self.detected = 0
        self.create_subscription(Image, "/imx500/image", self.on_image, 10)

    def on_image(self, msg):
        now = time.monotonic()
        if self.t0 is None: self.t0 = now
        mode = GuidanceMode.TRACK if (now - self.t0) < self.track_secs else GuidanceMode.DIVE
        if mode is GuidanceMode.DIVE and not self.dive_announced:
            self.get_logger().info("TRACK then DIVE: committing"); self.dive_announced = True
        rgb = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.width, 3)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self.backend._drain()
        dets = self.det.detect(bgr)
        raw = self.tracker.consume(bgr, dets, now)
        target = self.flt.update(raw, msg.width, msg.height, now)
        pitch = math.radians(self.backend.pitch_deg())
        roll = math.radians(self.backend.roll_deg())
        gamma = self.backend.flight_path_angle_rad()
        agl = self.backend.agl_m()
        # Online hover trim during TRACK (hold altitude -> trim hover toward null climb).
        if mode is GuidanceMode.TRACK and target is not None:
            self.state.hover = clamp(self.state.hover - 0.01 * self.backend.climb_mps(), 0.05, 0.6)
        ri = compute_rate_intent(target, self.cfg, self.state, now, mode=mode,
                                 pitch_rad=pitch, roll_rad=roll, gamma_rad=gamma, agl_m=agl)
        self.backend.send_body_rates(ri.roll_rate, ri.pitch_rate, ri.yaw_rate, ri.thrust)
        # --- overlay / video ---
        th = target.detection.h if target is not None else 0.0
        tpy = target.detection.y if target is not None else -1
        for d in dets:
            cv2.rectangle(bgr, (int(d.x-d.w/2), int(d.y-d.h/2)), (int(d.x+d.w/2), int(d.y+d.h/2)), (120,120,120), 1)
        if target is not None:
            t = target.detection
            cv2.rectangle(bgr, (int(t.x-t.w/2), int(t.y-t.h/2)), (int(t.x+t.w/2), int(t.y+t.h/2)), (0,255,0), 2)
        cv2.line(bgr,(360,278),(360,298),(255,255,0),1); cv2.line(bgr,(350,288),(370,288),(255,255,0),1)
        s = "%s alt=%.0f pitchR=%+.0f rollR=%+.0f thr=%.2f | pitch_m=%+.0f h=%.0f py=%.0f" % (
            ri.phase, agl, math.degrees(ri.pitch_rate), math.degrees(ri.roll_rate), ri.thrust,
            math.degrees(pitch), th, tpy)
        cv2.putText(bgr, s, (8,22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(bgr, s, (8,22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        self.writer.write(bgr); self.frames += 1; self.detected += bool(dets)
        if self.frames % 15 == 0:
            px, py_pos = self.backend.pos_xy()
            self.get_logger().info("%s f=%d alt=%.0f x=%.0f y=%.0f det=%d h=%.0f py=%.0f pitch_m=%+.0f thr=%.2f" % (
                ri.phase, self.frames, agl, px, py_pos, self.detected, th, tpy, math.degrees(pitch), ri.thrust))


def main():
    from pymavlink import mavutil
    M = mavutil.mavlink
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt", type=float, default=40.0); ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--track", type=float, default=6.0, help="seconds of TRACK range-hold before DIVE")
    a = ap.parse_args()
    backend = ArduPilotBackend(device="tcp:127.0.0.1:5760", baud=0, switch_channel=7, track_threshold_us=1300, dive_threshold_us=1700,
                               mapping=ArduCopterRcMapping(control_mode="stabilize", hover_learn=True, hover_learn_band=0.05))
    backend.open(); mav = backend._mav; mav.wait_heartbeat(timeout=60); mav.target_component = AP
    mav.mav.request_data_stream_send(mav.target_system, AP, M.MAV_DATA_STREAM_ALL, 15, 1); backend._request_streams()
    # Drain continuously from here (as the production pipeline ticks the FC every frame) so the
    # backend captures its ground home (agl_m) while DISARMED and freezes it at the arm instant.
    print("capturing ground home (disarmed)...", flush=True)
    drain_for(backend, 3.0)
    print("arming+takeoff to %.1fm..." % a.alt, flush=True)
    if not arm_takeoff(backend, mav, M, a.alt): print("FAIL takeoff"); return 1
    print("airborne; learning hover thrust (null climb)...", flush=True)
    hover = 0.30; t0 = time.monotonic(); samples = []
    while time.monotonic() - t0 < 16.0:
        backend._drain()                                   # keep _armed/alt fresh + read climb
        hover = min(0.55, max(0.05, hover - 0.010 * backend.climb_mps()))
        if time.monotonic() - t0 > 10.0:
            samples.append(hover)
        mav.mav.set_attitude_target_send(0, mav.target_system, AP, 0b10000000, [1,0,0,0], 0,0,0, hover); time.sleep(0.05)
    if samples: hover = sum(samples) / len(samples)
    print("learned hover=%.3f; home_agl=%.1f (ground); production rate law (TRACK -> DIVE)." % (
        hover, backend._home_alt or 0.0), flush=True)
    rclpy.init(); node = Rate(backend, a); node.state.hover = hover; node.state.sm_thr = hover
    end = time.monotonic() + a.duration
    while rclpy.ok() and time.monotonic() < end: rclpy.spin_once(node, timeout_sec=0.5)
    node.writer.release(); backend.release()
    mav.set_mode(GUIDED); mav.mav.command_long_send(mav.target_system, AP, M.MAV_CMD_NAV_LAND, 0,0,0,0,0,0,0,0)
    print("frames=%d det=%d -> /work/gz_rate.mp4" % (node.frames, node.detected), flush=True)
    rclpy.shutdown(); return 0


if __name__ == "__main__": sys.exit(main())
