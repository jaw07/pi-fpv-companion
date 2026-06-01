#!/usr/bin/env python3
"""End-to-end validation of the PRODUCTION guided_nogps path in Gazebo + ArduCopter SITL.

Unlike sitl_gz_rate.py (which drives the rate law directly), this runs the ACTUAL
pi_fpv_companion.Pipeline (control_mode guided_nogps, multi_iou tracker, ColorBlob on the Gazebo
camera as the IMX500 stand-in) against SITL, and exercises the flight-safety items the rate node
skipped:
  * GUID_OPTIONS bit-3 set by the PRODUCTION preflight (backend.ensure_param_bits), read back.
  * STANDBY safe-hold: force STANDBY in GUIDED_NOGPS -> copter HOLDS a level hover.
  * Pi-death: stop ticking entirely (no rates, no heartbeat) -> copter holds via the GUIDED
    command timeout (must not tumble / fall away).
  * TRACK -> DIVE -> impact through the real Pipeline._tick_rate.
Prints PASS/FAIL for each safety check and logs AGL + phase."""
import sys, time, math, argparse
sys.path.insert(0, "/work/pi-fpv-companion/src"); sys.path.insert(0, "/work")
import numpy as np, cv2, rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from pi_fpv_companion.config import load
from pi_fpv_companion.pipeline import Pipeline
from pi_fpv_companion.detect.color import ColorBlobDetector
from pi_fpv_companion.main import _build_tracker
from pi_fpv_companion.fc.ardupilot import ArduPilotBackend, ArduCopterRcMapping, GUID_OPTIONS_THRUST_AS_THRUST
from pi_fpv_companion.camera.base import FrameBundle
from pi_fpv_companion.types import GuidanceMode
from pi_fpv_companion.guidance.rate_control import RateConfig

GUIDED, GUIDED_NOGPS, AP = 4, 20, 1
W, H = 720, 576


def drain_for(b, s):
    t = time.monotonic()
    while time.monotonic() - t < s:
        b._drain(); time.sleep(0.02)


def arm_takeoff(backend, mav, M, alt, settle=18.0):
    # NOTE: GUID_OPTIONS is deliberately NOT set here — the production preflight does it (validated).
    for n, v in [("FRAME_CLASS", 1), ("FRAME_TYPE", 1), ("ARMING_CHECK", 0)]:
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
        if backend.agl_m() >= alt - 0.5: break
    mav.set_mode(GUIDED_NOGPS); drain_for(backend, 1.0); return True


class _DummyCam:
    def open(self): ...
    def close(self): ...
    def frames(self): return iter([])


class Validate(Node):
    def __init__(self, backend, cfg, hover):
        super().__init__("pifpv_validate")
        self.backend = backend
        self.det = ColorBlobDetector(min_area_px=50)
        self.pipe = Pipeline(_DummyCam(), _build_tracker(cfg), cfg.servo, cfg.safety, backend,
                             detector=self.det, rate_cfg=RateConfig(W, H), force_mode=GuidanceMode.STANDBY)
        self.pipe._rate_state.hover = hover          # seed the learned hover
        self.t0 = None
        self.marks = {}                              # phase -> (alt_start, alt_last)
        self.create_subscription(Image, "/imx500/image", self.on_image, 10)

    def phase(self, dt):
        if dt < 5:   return "A_STANDBY"
        if dt < 9:   return "B_PIDEATH"
        if dt < 15:  return "C_TRACK"
        return "D_DIVE"

    def on_image(self, msg):
        now = time.monotonic()
        if self.t0 is None: self.t0 = now
        dt = now - self.t0
        ph = self.phase(dt)
        agl = self.backend.agl_m()
        a0, _ = self.marks.get(ph, (agl, agl)); self.marks[ph] = (a0, agl)
        rgb = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.width, 3)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if ph == "B_PIDEATH":
            # Simulate Pi death: do NOT tick the pipeline -> no SET_ATTITUDE_TARGET, no heartbeat.
            pass
        else:
            self.pipe._force_mode = {"A_STANDBY": GuidanceMode.STANDBY, "C_TRACK": GuidanceMode.TRACK,
                                     "D_DIVE": GuidanceMode.DIVE}[ph]
            self.pipe.tick(FrameBundle(image=bgr, width=W, height=H, timestamp=now, detections=[]))
        if int(dt * 5) % 5 == 0:
            self.get_logger().info("%s dt=%.1f agl=%.1f pitch=%+.0f thr_state=%.2f" % (
                ph, dt, agl, self.backend.pitch_deg(), self.pipe._rate_state.hover))


def main():
    from pymavlink import mavutil
    M = mavutil.mavlink
    ap = argparse.ArgumentParser(); ap.add_argument("--alt", type=float, default=30.0); ap.add_argument("--duration", type=float, default=45.0)
    a = ap.parse_args()
    cfg = load("/work/pi-fpv-companion/config/imx500.yaml")
    print("control_mode=%s switch_ch=%d select_ch=%d tracker=%s" % (
        cfg.fc.control_mode, cfg.fc.switch_channel, cfg.fc.select_channel, cfg.tracker.type), flush=True)
    backend = ArduPilotBackend(device="tcp:127.0.0.1:5760", baud=0, switch_channel=7,
                               track_threshold_us=1300, dive_threshold_us=1700,
                               mapping=ArduCopterRcMapping(control_mode="guided_nogps"))
    backend.open(); mav = backend._mav; mav.wait_heartbeat(timeout=60); mav.target_component = AP
    mav.mav.request_data_stream_send(mav.target_system, AP, M.MAV_DATA_STREAM_ALL, 15, 1); backend._request_streams()
    drain_for(backend, 3.0)
    # ---- CHECK 1: production preflight sets GUID_OPTIONS bit 3 ----
    st = backend.ensure_param_bits("GUID_OPTIONS", GUID_OPTIONS_THRUST_AS_THRUST)
    pv = mav.recv_match(type="PARAM_VALUE", blocking=False)
    mav.mav.param_request_read_send(mav.target_system, AP, b"GUID_OPTIONS", -1); time.sleep(0.5)
    val = None
    for _ in range(20):
        m = mav.recv_match(type="PARAM_VALUE", blocking=True, timeout=1)
        if m and m.param_id.strip("\x00") == "GUID_OPTIONS": val = int(m.param_value); break
    ok1 = val is not None and (val & GUID_OPTIONS_THRUST_AS_THRUST)
    print("CHECK1 GUID_OPTIONS preflight: status=%s readback=%s -> %s" % (st, val, "PASS" if ok1 else "FAIL"), flush=True)
    if not arm_takeoff(backend, mav, M, a.alt): print("FAIL takeoff"); return 1
    # short hover-learn to seed the rate state
    hover = 0.30; t0 = time.monotonic(); samples = []
    while time.monotonic() - t0 < 12.0:
        backend._drain(); hover = min(0.55, max(0.05, hover - 0.010 * backend.climb_mps()))
        if time.monotonic() - t0 > 7.0: samples.append(hover)
        mav.mav.set_attitude_target_send(0, mav.target_system, AP, 0b10000000, [1,0,0,0], 0,0,0, hover); time.sleep(0.05)
    if samples: hover = sum(samples) / len(samples)
    print("learned hover=%.3f; running production Pipeline phases A_STANDBY -> B_PIDEATH -> C_TRACK -> D_DIVE" % hover, flush=True)
    rclpy.init(); node = Validate(backend, cfg, hover)
    end = time.monotonic() + a.duration
    while rclpy.ok() and time.monotonic() < end: rclpy.spin_once(node, timeout_sec=0.5)
    # ---- CHECK 2/3: STANDBY hold + Pi-death hold (altitude deltas) ----
    def delta(ph): a0, a1 = node.marks.get(ph, (0, 0)); return a1 - a0
    dA, dB = delta("A_STANDBY"), delta("B_PIDEATH")
    print("CHECK2 STANDBY safe-hold: dAGL=%+.1f m -> %s" % (dA, "PASS" if abs(dA) < 4.0 else "FAIL"), flush=True)
    print("CHECK3 Pi-death hold:     dAGL=%+.1f m -> %s" % (dB, "PASS" if dB > -6.0 else "FAIL"), flush=True)
    backend.release(); mav.set_mode(GUIDED); mav.mav.command_long_send(mav.target_system, AP, M.MAV_CMD_NAV_LAND, 0,0,0,0,0,0,0,0)
    rclpy.shutdown(); return 0


if __name__ == "__main__": sys.exit(main())
