#!/usr/bin/env python3
"""GUIDED_NOGPS rate-control node (faithful reference-style quad law).

Unlike sitl_gz_attitude.py (which sent an attitude quaternion), this commands BODY RATES
+ thrust via SET_ATTITUDE_TARGET with the attitude-ignore mask — the reference quad
interceptor's surface. Control law:
  * pitch RATE from a framing PID on the vertical angle error (target -> vert_goal, near top)
  * thrust from a PID on the TRUE angle below the horizon (in-frame elevation + measured pitch)
  * yaw RATE + roll RATE from a horizontal-error PID, blended (yaw far, roll near), with a
    roll-RETURN term (-k*current_roll) so the bank settles back to level.
Logs the target's frame position + commanded rates + measured attitude every frame so the
dive can be diagnosed (control-tracking vs framing-law). Records /work/gz_rate.mp4."""
import sys, time, argparse, math
sys.path.insert(0, "/work/pi-fpv-companion/src")
import numpy as np, cv2
from collections import deque
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from pi_fpv_companion.detect.color import ColorBlobDetector
from pi_fpv_companion.track.iou_associator import IouAssociator
from pi_fpv_companion.track.target_filter import AlphaBetaTargetFilter
from pi_fpv_companion.fc.ardupilot import ArduPilotBackend, ArduCopterRcMapping

STABILIZE, GUIDED, GUIDED_NOGPS, AP = 0, 4, 20, 1
W, H = 720, 576
HFOV, VFOV = math.radians(66.3), math.radians(52.3)
VERT_GOAL, HORI_GOAL = 0.40, 0.5   # target near CENTRE (on the velocity vector): fly INTO a ground target,
                                   # not under it. (Reference's 0.15/top is for chasing a forward AIR target.)


def clamp(v, lo, hi): return lo if v < lo else hi if v > hi else v


class PID:
    def __init__(self, kp, ki=0.0, kd=0.0, out=1e9, ilim=1e9):
        self.kp, self.ki, self.kd, self.out, self.ilim = kp, ki, kd, out, ilim
        self.i = 0.0; self.hist = deque(maxlen=5)
    def update(self, e, dt):
        self.i = clamp(self.i + e * dt, -self.ilim, self.ilim)
        self.hist.append(e); d = 0.0
        if len(self.hist) > 1 and dt > 0:
            d = (self.hist[-1] - self.hist[0]) / (dt * (len(self.hist) - 1))
        return clamp(self.kp * e + self.ki * self.i + self.kd * d, -self.out, self.out)


def arm_takeoff(mav, M, alt, home_alt, settle=18.0):
    # GUID_OPTIONS bit3 (=8) SetAttitudeTarget_ThrustAsThrust: make the thrust field REAL throttle.
    # Without it ArduCopter reads thrust as a CLIMB RATE (0 = hold altitude) -> the dive planes.
    for n, v in [("FRAME_CLASS", 1), ("FRAME_TYPE", 1), ("ARMING_CHECK", 0), ("GUID_OPTIONS", 8)]:
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
        if v and v.alt - home_alt >= alt - 0.5: break
    mav.set_mode(GUIDED_NOGPS); time.sleep(1.0); return True   # rate control in GUIDED_NOGPS


class Rate(Node):
    def __init__(self, backend, args):
        super().__init__("pifpv_rate")
        self.backend, self.a = backend, args; self.mav = backend._mav
        self.det = ColorBlobDetector(min_area_px=50)
        self.tracker = IouAssociator(max_lost_frames=25, max_match_dist_px=160.0)
        self.flt = AlphaBetaTargetFilter()
        HALFPI = math.pi / 2
        # Faithful reference gains: pitch/yaw/roll are RATE PIDs (rad/s); thrust is hover-relative [0,1].
        self.pitch_pid = PID(1.0, 0.0, 0.2, out=HALFPI)   # Kd trimmed from ref 0.3: our ColorBlob box is noisier
        self.thrust_pid = PID(0.85, 0.30, 0.1, out=0.3, ilim=0.3)  # strong integral: drives the velocity onto the
                                                                    # line-of-sight regardless of hover-feedforward error (robust to hover-learn scatter)
        self.hover = 0.30                # learned in main() (this fast quad hovers well below 0.5)
        self.aim_bias = -0.06            # rad (~3.4deg steeper): put the VELOCITY VECTOR on the target (it rides
                                         # ~one mush-angle above the bore, so aiming on the LOS passes a few m high)
        self.yaw_pid = PID(4.0, 0.0, 0.1, out=HALFPI, ilim=0.5)
        self.roll_pid = PID(3.0, 0.01, 0.1, out=HALFPI, ilim=0.5)
        self.roll_return = 5.0           # roll_position_p: rad/s per rad of current roll -> level
        self.base_yaw_p, self.base_roll_p = 4.0, 3.0
        self.max_pitch = 0.70            # rad (~40deg): moderate nose; with forward drag this yields a ~25-30deg
                                         # FLIGHT-PATH dive (the path is shallower than the nose on a powered multirotor)
        self.camera_pitch = 0.0          # fixed bore-sight level with the airframe
        self.max_horiz_err = 0.4; self.horiz_thresh = 0.05
        self.sm_yr = 0.0; self.sm_rr = 0.0; self.sm_pr = 0.0; self.sm_thr = 0.30   # EMA state (rates + thrust: anti-jitter/oscillation)
        self.writer = cv2.VideoWriter("/work/gz_rate.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (W, H))
        self.frames = self.detected = 0; self.last_t = None
        self.create_subscription(Image, "/imx500/image", self.on_image, 10)

    def send_rates(self, rr, pr, yr, thrust):
        # type_mask 0b10000000 + identity quaternion + body rates + thrust (reference form)
        tb = int(time.time() * 1000) & 0xFFFFFFFF
        self.mav.mav.set_attitude_target_send(tb, self.mav.target_system, AP, 0b10000000,
                                              [1.0, 0.0, 0.0, 0.0], rr, pr, yr, thrust)

    def _preprocess(self, cxn, cyn, roll_m):
        # De-rotate the target pixel about frame centre by the airframe roll, and widen the
        # effective FOV for that roll, so a banked airframe still computes correct angle errors.
        dx, dy = cxn - 0.5, cyn - 0.5
        c, s = math.cos(roll_m), math.sin(roll_m)
        rx, ry = c * dx - s * dy, s * dx + c * dy
        cxn, cyn = rx + 0.5, ry + 0.5
        hfov = abs(HFOV * c) + abs(VFOV * s)
        vfov = abs(VFOV * c) + abs(HFOV * s)
        horiz_err = (cxn - HORI_GOAL) * hfov                       # + = target right of centre
        vert_err = (VERT_GOAL - cyn) * vfov                        # + = target above the top goal
        ang_to_tgt = (0.5 - cyn) * vfov + (self.pitch_m + self.camera_pitch)  # true angle below horizon
        return horiz_err, vert_err, ang_to_tgt

    def on_image(self, msg):
        now = time.monotonic()
        dt = clamp((now - self.last_t) if self.last_t else 0.0, 0.0, 0.2); self.last_t = now
        rgb = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.width, 3)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self.backend._drain()
        dets = self.det.detect(bgr)
        raw = self.tracker.consume(bgr, dets, now)
        target = self.flt.update(raw, msg.width, msg.height, now)
        pitch_m = math.radians(self.backend.pitch_deg()); roll_m = math.radians(self.backend.roll_deg())
        self.pitch_m = pitch_m
        th = target.detection.h if target is not None else 0.0
        alt = self.backend.alt_m()
        if target is not None:
            det = target.detection; cxn, cyn = det.x / W, det.y / H
            horiz_err, vert_err, ang_to_tgt = self._preprocess(cxn, cyn, roll_m)
            # PITCH rate frames the target to the top goal; only zero the rate at the attitude limit.
            pr = self.pitch_pid.update(vert_err, dt)
            if (pitch_m <= -self.max_pitch and pr < 0) or (pitch_m >= self.max_pitch and pr > 0): pr = 0.0
            # THRUST = PURSUIT (now that GUID_OPTIONS bit3 gives REAL throttle authority): descend just
            # enough to drive the flight-path angle onto the line-of-sight, so the velocity vector points
            # AT the target -> a straight-line dive whose angle == the target's depression. gamma>los
            # (too shallow) -> error<0 -> thrust<hover -> descend until the velocity aims at the target.
            gamma = self.backend.flight_path_angle_rad()
            thrust = clamp(self.hover + self.thrust_pid.update(ang_to_tgt + self.aim_bias - gamma, dt), 0.0, 1.0)
            # YAW/ROLL blend: yaw dominates far off-axis, roll banks near centre; roll returns to level.
            ae = abs(horiz_err)
            alpha = clamp((ae - self.horiz_thresh) / max(self.max_horiz_err, ae - self.horiz_thresh), 0.0, 1.0)
            self.yaw_pid.kp = alpha * self.base_yaw_p
            self.roll_pid.kp = (1.0 - alpha) * self.base_roll_p
            yr = self.yaw_pid.update(horiz_err, dt)
            rr = self.roll_pid.update(horiz_err, dt) - self.roll_return * roll_m
            # Scale horizontal authority by target SIZE: a small far target gives a noisy box, and the
            # high-gain yaw/roll turn that into gyration early in the run. Reduce it when far; full
            # authority once the target is large (deep in the dive), where the box is stable.
            sg = clamp((th - 8.0) / 18.0, 0.3, 1.0)
            yr *= sg; rr *= sg
            phase = "RATE"; tpx, tpy = det.x, det.y
        else:
            # SEARCH: nose down to ~-25deg to bring a below ground target into the FOV. Our targets sit
            # below the airframe; from a steep engagement a level hover never sees the target, so the
            # craft must look down to acquire. Hold level roll, hover thrust.
            pr = clamp(2.0 * (math.radians(-25) - pitch_m), -0.6, 0.6)
            rr, yr, thrust = -self.roll_return * roll_m, 0.0, self.hover
            phase = "SRCH"; tpx, tpy = -1, -1
        # Low-pass ALL commanded rates: the ColorBlob box jitters in size/centre (worst when the
        # target is small/far and again as it whips through the frame in the terminal), and the
        # high-gain rate PIDs turn that into pitch/yaw/roll twitch. EMA-smoothing the rate the
        # airframe is asked to fly removes the visible jitter/oscillation while the rate surface
        # (airframe integrates the command) keeps the dive itself smooth.
        b = 0.35
        self.sm_pr += b * (pr - self.sm_pr); self.sm_yr += b * (yr - self.sm_yr); self.sm_rr += b * (rr - self.sm_rr)
        self.sm_thr += b * (thrust - self.sm_thr)            # smooth the throttle too -> less descent oscillation
        pr, yr, rr, thrust = self.sm_pr, self.sm_yr, self.sm_rr, self.sm_thr
        self.send_rates(rr, pr, yr, thrust)
        for d in dets:
            cv2.rectangle(bgr, (int(d.x-d.w/2), int(d.y-d.h/2)), (int(d.x+d.w/2), int(d.y+d.h/2)), (120,120,120), 1)
        if target is not None:
            t = target.detection
            cv2.rectangle(bgr, (int(t.x-t.w/2), int(t.y-t.h/2)), (int(t.x+t.w/2), int(t.y+t.h/2)), (0,255,0), 2)
        cv2.line(bgr,(360,278),(360,298),(255,255,0),1); cv2.line(bgr,(350,288),(370,288),(255,255,0),1)
        s = "%s alt=%.0f pitchR=%+.0f rollR=%+.0f thr=%.2f | pitch_m=%+.0f h=%.0f py=%.0f" % (
            phase, alt, math.degrees(pr), math.degrees(rr), thrust, math.degrees(pitch_m), th, tpy)
        cv2.putText(bgr, s, (8,22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(bgr, s, (8,22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        self.writer.write(bgr); self.frames += 1; self.detected += bool(dets)
        if self.frames % 15 == 0:
            px, py_pos = self.backend.pos_xy()
            self.get_logger().info("%s f=%d alt=%.0f x=%.0f y=%.0f det=%d h=%.0f py=%.0f pitch_m=%+.0f thr=%.2f" % (
                phase, self.frames, alt, px, py_pos, self.detected, th, tpy, math.degrees(pitch_m), thrust))


def main():
    from pymavlink import mavutil
    M = mavutil.mavlink
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt", type=float, default=40.0); ap.add_argument("--duration", type=float, default=60.0)
    a = ap.parse_args()
    backend = ArduPilotBackend(device="tcp:127.0.0.1:5760", baud=0, switch_channel=7, track_threshold_us=1300, dive_threshold_us=1700,
                               mapping=ArduCopterRcMapping(control_mode="stabilize", hover_learn=True, hover_learn_band=0.05))
    backend.open(); mav = backend._mav; mav.wait_heartbeat(timeout=60); mav.target_component = AP
    mav.mav.request_data_stream_send(mav.target_system, AP, M.MAV_DATA_STREAM_ALL, 15, 1); backend._request_streams()
    v0 = mav.recv_match(type="VFR_HUD", blocking=True, timeout=5); home = v0.alt if v0 else 0.0
    print("arming+takeoff to %.1fm..." % a.alt, flush=True)
    if not arm_takeoff(mav, M, a.alt, home): print("FAIL takeoff"); return 1
    print("airborne; learning hover thrust (null climb)...", flush=True)
    hover = 0.30; t0 = time.monotonic(); samples = []
    while time.monotonic() - t0 < 16.0:
        v = mav.recv_match(type="VFR_HUD", blocking=False)
        if v is not None:
            hover = min(0.55, max(0.05, hover - 0.010 * float(v.climb)))  # climbing -> lower hover; sinking -> raise
            if time.monotonic() - t0 > 10.0:                 # collect once converged; average for a stable value
                samples.append(hover)
        mav.mav.set_attitude_target_send(0, mav.target_system, AP, 0b10000000, [1,0,0,0], 0,0,0, hover); time.sleep(0.05)
    if samples: hover = sum(samples) / len(samples)
    print("learned hover thrust = %.3f (avg of %d); RATE control (search -> dive)." % (hover, len(samples)), flush=True)
    rclpy.init(); node = Rate(backend, a); node.hover = hover
    end = time.monotonic() + a.duration
    while rclpy.ok() and time.monotonic() < end: rclpy.spin_once(node, timeout_sec=0.5)
    node.writer.release(); backend.release()
    mav.set_mode(GUIDED); mav.mav.command_long_send(mav.target_system, AP, M.MAV_CMD_NAV_LAND, 0,0,0,0,0,0,0,0)
    print("frames=%d det=%d -> /work/gz_rate.mp4" % (node.frames, node.detected), flush=True)
    rclpy.shutdown(); return 0


if __name__ == "__main__": sys.exit(main())
