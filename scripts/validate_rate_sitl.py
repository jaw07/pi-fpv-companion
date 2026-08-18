#!/usr/bin/env python3
"""SITL validation of the PRODUCTION guided_nogps body-rate law — no Gazebo needed.

sitl_gz_rate.py exercises this same law but needs ROS 2 + a Gazebo camera topic. That
left the rate path — the one the aircraft actually flies with control_mode:
guided_nogps — with NO runnable SITL check. This drives
`rate_control.compute_rate_intent` (the shipped controller, not a copy) against real
ArduCopter physics using a SYNTHETIC target, and asserts the output-conditioning
properties: bounded throttle, bounded attitude, incremental commands, no oscillation.

    docker run -d --rm --name pifpv-sitl -p 5760:5760 pifpv-sitl:4.6
    .venv/bin/python scripts/validate_rate_sitl.py --connect tcp:127.0.0.1:5760

SAFETY: this ARMS the vehicle, so it refuses anything that is not a tcp/udp SITL
connection. It must never be pointed at a serial device / real flight controller.
"""
from __future__ import annotations
import argparse, math, statistics, sys, time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from pymavlink import mavutil
from pi_fpv_companion.guidance.rate_control import RateConfig, RateState, compute_rate_intent
from pi_fpv_companion.types import Detection, FilteredTarget, GuidanceMode

W, H = 720, 576
GUIDED, GUIDED_NOGPS = 4, 20


def _ft(cx_n, cy_n, h_px, t):
    return FilteredTarget(
        detection=Detection(x=cx_n * W, y=cy_n * H, w=h_px, h=h_px, confidence=0.9, class_id=0),
        track_id=1, vx_px_s=0.0, vy_px_s=0.0, quality=0.9, timestamp=t)


def set_mode(m, mode, timeout=15):
    end = time.time() + timeout
    while time.time() < end:
        m.mav.set_mode_send(m.target_system,
                            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode)
        h = m.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
        if h and h.custom_mode == mode:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--connect", default="tcp:127.0.0.1:5760")
    ap.add_argument("--alt", type=float, default=40.0)
    ap.add_argument("--track-s", type=float, default=8.0)
    ap.add_argument("--dive-s", type=float, default=12.0)
    ap.add_argument("--hz", type=float, default=22.0)
    args = ap.parse_args()

    if not args.connect.startswith(("tcp:", "udp:", "udpin:", "udpout:")):
        print(f"REFUSING: --connect {args.connect!r} is not a SITL socket. "
              "This script ARMS the vehicle and must never touch a real FC.")
        return 2

    m = mavutil.mavlink_connection(args.connect)
    m.wait_heartbeat(timeout=40)
    print("connected to SITL")
    for msg in ("ATTITUDE", "VFR_HUD", "GLOBAL_POSITION_INT"):
        m.mav.command_long_send(m.target_system, m.target_component,
                                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                                getattr(mavutil.mavlink, f"MAVLINK_MSG_ID_{msg}"),
                                50000, 0, 0, 0, 0, 0)
    for name, val in (("GUID_OPTIONS", 8), ("ARMING_CHECK", 0)):
        m.mav.param_set_send(m.target_system, m.target_component, name.encode(),
                             float(val), mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
        time.sleep(0.3)
    print("  GUID_OPTIONS=8 (ThrustAsThrust), ARMING_CHECK=0")

    # Climb to altitude in GUIDED (GPS available in SITL), then hand over to the
    # GPS-denied rate law for the actual test.
    if not set_mode(m, GUIDED):
        print("could not enter GUIDED"); return 1
    # Wait for the EKF to settle before arming, or the arm is silently refused and the
    # whole run proceeds on a vehicle sitting on the ground — where every "bounded
    # attitude / bounded descent" assertion passes trivially and the result is a lie.
    # A fresh SITL needs ~60-90s before it will arm: the IMU has to settle ("Arm: Accels
    # inconsistent") and the EKF has to set home ("Arm: AHRS: waiting for home").
    # Arm on the COMMAND_ACK, not on a heartbeat flag — a stale heartbeat read as armed
    # once and the run proceeded on a vehicle still sitting on the ground.
    print("  waiting for EKF / IMU / home (fresh SITL needs ~60-90s) ...")
    armed = False
    reasons = set()
    end = time.time() + 180
    while time.time() < end and not armed:
        m.mav.command_long_send(m.target_system, m.target_component,
                                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
                                1, 0, 0, 0, 0, 0, 0)
        t0 = time.time()
        while time.time() - t0 < 4:
            a = m.recv_match(type=["COMMAND_ACK", "STATUSTEXT"], blocking=True, timeout=1)
            if a is None:
                continue
            if a.get_type() == "STATUSTEXT":
                if a.text not in reasons:
                    reasons.add(a.text)
                    print(f"    FC says: {a.text}")
            elif a.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
                if a.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    armed = True
                break
    if not armed:
        print("ABORT: SITL never armed — cannot validate anything."); return 1
    print("  armed (ACK accepted)")
    m.mav.command_long_send(m.target_system, m.target_component,
                            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0,0,0,0,0,0, args.alt)
    print(f"  climbing to {args.alt:.0f}m ...")
    end = time.time() + 90
    alt = 0.0
    while time.time() < end:
        v = m.recv_match(type="VFR_HUD", blocking=True, timeout=2)
        if v:
            alt = v.alt
            if alt >= args.alt * 0.92:
                break
    print(f"  at {alt:.1f}m")
    if alt < args.alt * 0.7:
        print(f"ABORT: only reached {alt:.1f}m of {args.alt:.0f}m. A grounded vehicle "
              "passes every bounds check vacuously — refusing to report a result.")
        return 1
    start_alt = alt
    if not set_mode(m, GUIDED_NOGPS):
        print("could not enter GUIDED_NOGPS"); return 1
    print("  in GUIDED_NOGPS — driving the production rate law\n")

    cfg = RateConfig(W, H)
    st = RateState()
    st.hover = 0.35
    st.sm_thr = st.hover
    dt = 1.0 / args.hz

    rec = {"thrust": [], "climb": [], "pitch": [], "roll": [], "alt": [],
           "yaw_rate": [], "d_thrust": []}
    prev_thr = st.hover
    prev_t = None
    t0 = time.monotonic()
    phase_end = t0 + args.track_s
    mode = GuidanceMode.TRACK
    pitch = roll = climb = 0.0
    alt_now = alt

    while True:
        now = time.monotonic()
        if mode is GuidanceMode.TRACK and now >= phase_end:
            mode = GuidanceMode.DIVE
            st.reset(); st.sm_thr = st.hover
            phase_end = now + args.dive_s
            print(f"  -> DIVE at t={now-t0:.1f}s alt={alt_now:.1f}m")
        elif mode is GuidanceMode.DIVE and now >= phase_end:
            break

        while True:
            msg = m.recv_match(type=["ATTITUDE", "VFR_HUD"], blocking=False)
            if msg is None:
                break
            if msg.get_type() == "ATTITUDE":
                pitch, roll = msg.pitch, msg.roll
            else:
                climb, alt_now = msg.climb, msg.alt

        # Synthetic target: centred horizontally, LOW in frame (a ground target below).
        tgt = _ft(0.52, 0.80, 40, now)
        gamma = math.atan2(climb, 5.0)
        ri = compute_rate_intent(tgt, cfg, st, now=now, mode=mode, pitch_rad=pitch,
                                 roll_rad=roll, gamma_rad=gamma, agl_m=max(alt_now, 0.0))
        m.mav.set_attitude_target_send(
            int((now - t0) * 1000) & 0xFFFFFFFF, m.target_system, m.target_component,
            0b10000000, [1.0, 0.0, 0.0, 0.0],
            ri.roll_rate, ri.pitch_rate, ri.yaw_rate, ri.thrust)

        rec["thrust"].append(ri.thrust); rec["climb"].append(climb)
        rec["pitch"].append(math.degrees(pitch)); rec["roll"].append(math.degrees(roll))
        rec["alt"].append(alt_now); rec["yaw_rate"].append(ri.yaw_rate)
        # Use the ACTUAL elapsed time, not the nominal period: time.sleep() jitters and
        # dividing by the nominal dt inflates the apparent slew rate by 30%+.
        if prev_t is not None:
            real_dt = now - prev_t
            if real_dt > 1e-4:
                rec["d_thrust"].append(abs(ri.thrust - prev_thr) / real_dt)
        prev_t = now
        prev_thr = ri.thrust
        time.sleep(dt)

    # Land the sim vehicle and disarm.
    set_mode(m, 9)   # LAND
    print("\n=== results ===")
    ok = True

    # GUARD FIRST: prove the run actually exercised the controller. Without this a
    # vehicle that never left the ground reports six green PASSes.
    alt_span = max(rec["alt"]) - min(rec["alt"])
    thr_span = max(rec["thrust"]) - min(rec["thrust"])
    if alt_span < 2.0 and thr_span < 0.02:
        print(f"  [INVALID] the vehicle never moved (alt span {alt_span:.1f}m, "
              f"thrust span {thr_span:.3f}). Nothing was validated.")
        return 1

    def chk(label, cond, detail):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}: {detail}")
        if not cond:
            ok = False

    min_thr, max_thr = min(rec["thrust"]), max(rec["thrust"])
    floor = cfg.min_thrust_frac * st.hover
    chk("throttle never idles", min_thr >= floor - 1e-6,
        f"min={min_thr:.3f} floor={floor:.3f} max={max_thr:.3f}")
    chk("throttle incremental", max(rec["d_thrust"]) <= cfg.slew_thrust + 0.05,
        f"worst {max(rec['d_thrust']):.2f}/s limit {cfg.slew_thrust}/s")
    chk("pitch bounded", max(abs(p) for p in rec["pitch"]) < 60.0,
        f"max |pitch| {max(abs(p) for p in rec['pitch']):.1f} deg")
    chk("roll bounded", max(abs(r) for r in rec["roll"]) < 45.0,
        f"max |roll| {max(abs(r) for r in rec['roll']):.1f} deg")
    chk("descent bounded", min(rec["climb"]) > -25.0,
        f"peak descent {min(rec['climb']):.1f} m/s")
    # Oscillation: sign changes per second in the commanded yaw rate.
    sc = sum(1 for a, b in zip(rec["yaw_rate"], rec["yaw_rate"][1:]) if a * b < 0)
    per_s = sc / max(1e-6, len(rec["yaw_rate"]) * dt)
    chk("no yaw oscillation", per_s < 3.0, f"{per_s:.2f} sign changes/s")
    print(f"\n  altitude {rec['alt'][0]:.1f} -> {rec['alt'][-1]:.1f} m over "
          f"{len(rec['alt'])*dt:.0f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
