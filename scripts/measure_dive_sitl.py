"""DIVE capability side-by-side: ALT_HOLD vs STABILIZE, via RC override (SITL).

ALT_HOLD holds altitude — descent is rate-limited to PILOT_SPEED_DN and the
altitude controller resists altitude loss. STABILIZE has NO altitude hold:
throttle is direct, so cutting it + nosing down is a real (physics-limited) dive,
at the cost of no baro altitude floor.

Method: GUIDED NAV_TAKEOFF to a high altitude (the reliable SITL way up), then dive
in each mode in sequence (no re-climb — ALT_HOLD first since it barely loses
altitude, STABILIZE last) commanding a DIVE-like intent (forward lean + full
throttle-down) via the production ArduPilotBackend.send_intent (RC override),
measuring real altitude loss (VFR_HUD.alt) + ground speed + dive-path angle.

    docker run -d --rm --name pifpv-sitl -p 127.0.0.1:5760:5760 pifpv-sitl:4.6
    .venv/bin/python scripts/measure_dive_sitl.py --connect tcp:127.0.0.1:5760
"""
from __future__ import annotations
import argparse
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from pi_fpv_companion.fc.ardupilot import ArduPilotBackend, ArduCopterRcMapping
from pi_fpv_companion.types import GuidanceIntent

STABILIZE, ALT_HOLD, GUIDED = 0, 2, 4
AP = 1


def set_param(mav, name, val, ptype="INT32"):
    from pymavlink import mavutil
    mav.mav.param_set_send(mav.target_system, AP, name.encode(), float(val),
                           getattr(mavutil.mavlink, f"MAV_PARAM_TYPE_{ptype}"))
    return read_param(mav, name)


def read_param(mav, name, default=0):
    mav.mav.param_request_read_send(mav.target_system, AP, name.encode(), -1)
    end = time.time() + 6.0
    while time.time() < end:
        pv = mav.recv_match(type="PARAM_VALUE", blocking=True, timeout=end - time.time())
        if pv is not None and pv.param_id.strip("\x00") == name:   # match the NAME
            return pv.param_value
    return default


def vfr(mav, timeout=3.0):
    return mav.recv_match(type="VFR_HUD", blocking=True, timeout=timeout)


def main():
    from pymavlink import mavutil
    M = mavutil.mavlink
    ap = argparse.ArgumentParser()
    ap.add_argument("--connect", default="tcp:127.0.0.1:5760")
    ap.add_argument("--alt", type=float, default=150.0, help="takeoff altitude AGL (m)")
    ap.add_argument("--pitch-deg", type=float, default=30.0)
    ap.add_argument("--dive-secs", type=float, default=4.0)
    args = ap.parse_args()

    base_map = ArduCopterRcMapping()
    backend = ArduPilotBackend(device=args.connect, baud=0, switch_channel=7,
                               track_threshold_us=1300, dive_threshold_us=1700, mapping=base_map)
    backend.open()
    mav = backend._mav
    print(f"connecting to {args.connect} ...")
    mav.wait_heartbeat(timeout=90)
    mav.target_component = AP
    mav.mav.request_data_stream_send(mav.target_system, AP, M.MAV_DATA_STREAM_ALL, 10, 1)

    print("  settling 60s for IMU/EKF convergence ...")
    time.sleep(60)
    end = time.time() + 60
    while time.time() < end:
        g = mav.recv_match(type="GPS_RAW_INT", blocking=True, timeout=2)
        if g and g.fix_type >= 3:
            break
    default_dn = int(read_param(mav, "PILOT_SPEED_DN", 0))
    v0 = vfr(mav, 5)
    home_alt = v0.alt if v0 else 0.0
    print(f"  home_alt={home_alt:.1f} m, PILOT_SPEED_DN default={default_dn} cm/s")

    # ---- GUIDED takeoff (reliable; pure NAV_TAKEOFF + poll) -------------
    mav.set_mode(GUIDED)
    armed = False
    end = time.time() + 90
    last = 0.0
    while time.time() < end and not armed:
        if time.time() - last > 4:
            mav.mav.command_long_send(mav.target_system, AP, M.MAV_CMD_COMPONENT_ARM_DISARM,
                                      0, 1, 0, 0, 0, 0, 0, 0)
            last = time.time()
        x = mav.recv_match(type=["HEARTBEAT", "STATUSTEXT"], blocking=True, timeout=2)
        if x and x.get_type() == "STATUSTEXT" and any(k in x.text for k in ("Arm", "EKF", "Gyro", "Accel")):
            print("   ST:", x.text)
        if x and x.get_type() == "HEARTBEAT":
            armed = bool(x.base_mode & M.MAV_MODE_FLAG_SAFETY_ARMED)
    if not armed:
        print("FAIL: never armed"); return 1

    print(f"\n  GUIDED takeoff to {args.alt:.0f} m AGL ...")
    mav.mav.command_long_send(mav.target_system, AP, M.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, args.alt)
    end = time.time() + 150
    agl = 0.0
    while time.time() < end:
        v = vfr(mav, 2)
        if v:
            agl = v.alt - home_alt
            if agl >= args.alt - 6:
                break
    print(f"  reached {agl:.0f} m AGL")
    if agl < 20:
        print("FAIL: takeoff did not climb"); return 1

    def dive(label, mode, control_mode, dn=None):
        backend._mapping = replace(base_map, control_mode=control_mode)
        for i in range(1, 7):
            set_param(mav, f"FLTMODE{i}", mode)
        mav.set_mode(mode)
        if dn is not None:
            set_param(mav, "PILOT_SPEED_DN", int(dn))
        time.sleep(0.5)
        a0 = (vfr(mav, 3) or v0).alt
        dive_i = GuidanceIntent(0.0, -args.pitch_deg, 0.0, 0.0, time.monotonic())
        end = time.time() + args.dive_secs
        gss = []
        while time.time() < end:
            backend.send_intent(dive_i)
            v = mav.recv_match(type="VFR_HUD", blocking=False)
            if v:
                gss.append(v.groundspeed)
            time.sleep(0.05)
        a1 = (vfr(mav, 3) or v0).alt
        descent = (a0 - a1) / args.dive_secs
        gs = sum(gss) / len(gss) if gss else 0.0
        ang = math.degrees(math.atan2(descent, gs)) if gs > 0.2 else 90.0
        # arrest: hand back to level/hold so we don't crash before the next case
        backend.send_intent(GuidanceIntent(0, 0, 0, 0.5, time.monotonic()))
        print(f"  [{label}] alt {a0-home_alt:.0f}->{a1-home_alt:.0f} AGL | "
              f"descent {descent:4.1f} m/s | gs {gs:4.1f} m/s | path {ang:3.0f} deg")
        return descent, gs, ang

    print(f"\n{'='*64}\nDIVE side-by-side (forward lean {args.pitch_deg:.0f} deg, full throttle-down, {args.dive_secs:.0f}s)")
    a_def = dive(f"ALT_HOLD DN={default_dn}", ALT_HOLD, "althold", default_dn)
    a_hi = dive("ALT_HOLD DN=1000", ALT_HOLD, "althold", 1000)
    stab = dive("STABILIZE", STABILIZE, "stabilize")

    print(f"\n{'='*64}")
    print(f"{'mode':<28}{'descent m/s':>12}{'path deg':>10}")
    print(f"{'ALT_HOLD DN='+str(default_dn):<28}{a_def[0]:>12.1f}{a_def[2]:>10.0f}")
    print(f"{'ALT_HOLD DN=1000 (10 m/s)':<28}{a_hi[0]:>12.1f}{a_hi[2]:>10.0f}")
    print(f"{'STABILIZE (direct throttle)':<28}{stab[0]:>12.1f}{stab[2]:>10.0f}")

    backend.release()
    mav.set_mode(GUIDED)
    mav.mav.command_long_send(mav.target_system, AP, M.MAV_CMD_NAV_LAND, 0, 0, 0, 0, 0, 0, 0, 0)
    backend.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
