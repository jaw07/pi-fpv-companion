"""SITL check: the backend's closed-loop tracks a commanded vertical RATE.

The closed-loop DIVE (constant-bearing homing) commands a vertical rate
(`GuidanceIntent.vertical_rate_mps`) that the backend tracks against VFR_HUD.climb.
This confirms, on ArduCopter 4.6.3, that commanding e.g. -5 m/s actually produces
~-5 m/s and that 0 holds — the inner loop the servo's framing loop sits on top of.

    docker run -d --rm --name pifpv-sitl -p 127.0.0.1:5760:5760 pifpv-sitl:4.6
    .venv/bin/python scripts/validate_vrate_sitl.py --connect tcp:127.0.0.1:5760
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from pi_fpv_companion.fc.ardupilot import ArduPilotBackend, ArduCopterRcMapping
from pi_fpv_companion.types import GuidanceIntent

STABILIZE, GUIDED = 0, 4
AP = 1


def vfr(mav, t=3.0):
    return mav.recv_match(type="VFR_HUD", blocking=True, timeout=t)


def track(backend, mav, label, rate, secs, home_alt):
    a0 = (vfr(mav, 3) or None); a0 = a0.alt if a0 else home_alt
    climbs = []
    intent = GuidanceIntent(0.0, 0.0, 0.0, 0.5, time.monotonic(), vertical_rate_mps=rate)
    end = time.time() + secs
    while time.time() < end:
        backend._drain()
        backend.send_intent(intent)
        c = backend._climb_mps
        if c is not None:
            climbs.append(c)
        time.sleep(0.05)
    a1 = (vfr(mav, 3) or None); a1 = a1.alt if a1 else home_alt
    measured = (a1 - a0) / secs
    settled = sum(climbs[-20:]) / max(1, len(climbs[-20:]))
    print(f"  [{label}] cmd {rate:+.1f}  measured {measured:+5.2f} m/s  "
          f"(climb sensor {settled:+5.2f})  hover={backend._hover_pwm:.0f}")
    return measured


def main():
    from pymavlink import mavutil
    M = mavutil.mavlink
    ap = argparse.ArgumentParser()
    ap.add_argument("--connect", default="tcp:127.0.0.1:5760")
    ap.add_argument("--alt", type=float, default=120.0)
    args = ap.parse_args()
    backend = ArduPilotBackend(device=args.connect, baud=0, switch_channel=7,
                               track_threshold_us=1300, dive_threshold_us=1700,
                               mapping=ArduCopterRcMapping(control_mode="stabilize", hover_learn=True))
    backend.open()
    mav = backend._mav
    print(f"connecting {args.connect} ...")
    mav.wait_heartbeat(timeout=90)
    mav.target_component = AP
    mav.mav.request_data_stream_send(mav.target_system, AP, M.MAV_DATA_STREAM_ALL, 10, 1)
    backend._request_streams()
    print("  settling 60s ..."); time.sleep(60)
    v0 = vfr(mav, 5); home_alt = v0.alt if v0 else 0.0
    mav.set_mode(GUIDED)
    end = time.time() + 90; armed = False; last = 0.0
    while time.time() < end and not armed:
        if time.time() - last > 4:
            mav.mav.command_long_send(mav.target_system, AP, M.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
            last = time.time()
        x = mav.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
        if x:
            armed = bool(x.base_mode & M.MAV_MODE_FLAG_SAFETY_ARMED)
    if not armed:
        print("FAIL: never armed"); return 1
    mav.mav.command_long_send(mav.target_system, AP, M.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, args.alt)
    end = time.time() + 150
    while time.time() < end:
        v = vfr(mav, 2)
        if v and v.alt - home_alt >= args.alt - 6:
            break
    for i in range(1, 7):
        mav.mav.param_set_send(mav.target_system, AP, f"FLTMODE{i}".encode(), float(STABILIZE), M.MAV_PARAM_TYPE_INT32)
    mav.set_mode(STABILIZE); time.sleep(1)
    print("  converging hover 20s ...")
    end = time.time() + 20
    while time.time() < end:
        backend._drain(); backend.send_intent(GuidanceIntent(0, 0, 0, 0.5, time.monotonic(), vertical_rate_mps=0.0)); time.sleep(0.05)
    print(f"  hover={backend._hover_pwm:.0f}\n\nClosed-loop vertical-rate tracking (alt {args.alt:.0f} m):")
    hold = track(backend, mav, "HOLD  0.0", 0.0, 5.0, home_alt)
    d3 = track(backend, mav, "DESC -3.0", -3.0, 6.0, home_alt)
    c2 = track(backend, mav, "CLIMB+2.0", 2.0, 5.0, home_alt)
    backend.send_intent(GuidanceIntent(0, 0, 0, 0.5, time.monotonic(), vertical_rate_mps=0.0))
    backend.release()
    ok = abs(hold) < 1.0 and d3 < -1.5 and c2 > 0.8
    print(f"\n{'PASS' if ok else 'FAIL'}: hold≈0 ({hold:+.2f}), descent tracks ({d3:+.2f}), "
          f"climb tracks ({c2:+.2f}).")
    mav.set_mode(GUIDED)
    mav.mav.command_long_send(mav.target_system, AP, M.MAV_CMD_NAV_LAND, 0, 0, 0, 0, 0, 0, 0, 0)
    backend.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
