"""Adaptive-hover proof (SITL): seed a WRONG hover throttle and watch the
companion learn it.

STABILIZE has no altitude hold, so the backend runs a companion vertical-velocity
hold: it trims the hover throttle from measured climb rate (VFR_HUD) until the
craft levels out. This GUIDED-takes-off to altitude, switches to STABILIZE with
the hover seed set deliberately LOW (so it starts descending), commands "hold"
(thrust 0.5), and logs altitude / climb rate / learned hover PWM over time —
showing climb rate converge to ~0 and the learned hover settle at the true value.

    docker run -d --rm --name pifpv-sitl -p 127.0.0.1:5760:5760 pifpv-sitl:4.6
    .venv/bin/python scripts/learn_hover_sitl.py --connect tcp:127.0.0.1:5760
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

STABILIZE, ALT_HOLD, GUIDED = 0, 2, 4
AP = 1


def set_param(mav, name, val, ptype="INT32"):
    from pymavlink import mavutil
    mav.mav.param_set_send(mav.target_system, AP, name.encode(), float(val),
                           getattr(mavutil.mavlink, f"MAV_PARAM_TYPE_{ptype}"))


def vfr(mav, timeout=2.0):
    return mav.recv_match(type="VFR_HUD", blocking=True, timeout=timeout)


def main():
    from pymavlink import mavutil
    M = mavutil.mavlink
    ap = argparse.ArgumentParser()
    ap.add_argument("--connect", default="tcp:127.0.0.1:5760")
    ap.add_argument("--seed-hover", type=int, default=1300, help="deliberately-wrong hover seed PWM")
    ap.add_argument("--alt", type=float, default=80.0)
    ap.add_argument("--watch", type=float, default=24.0)
    args = ap.parse_args()

    backend = ArduPilotBackend(
        device=args.connect, baud=0, switch_channel=7,
        track_threshold_us=1300, dive_threshold_us=1700,
        mapping=ArduCopterRcMapping(control_mode="stabilize", hover_throttle_us=args.seed_hover,
                                    hover_learn=True, hover_learn_gain=60.0))
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
    set_param(mav, "ARMING_CHECK", 0)
    v0 = vfr(mav, 5)
    home = v0.alt if v0 else 0.0

    # GUIDED takeoff to altitude
    mav.set_mode(GUIDED)
    armed = False
    end = time.time() + 90
    last = 0.0
    while time.time() < end and not armed:
        if time.time() - last > 4:
            mav.mav.command_long_send(mav.target_system, AP, M.MAV_CMD_COMPONENT_ARM_DISARM,
                                      0, 1, 0, 0, 0, 0, 0, 0)
            last = time.time()
        x = mav.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
        if x:
            armed = bool(x.base_mode & M.MAV_MODE_FLAG_SAFETY_ARMED)
    if not armed:
        print("FAIL: never armed"); return 1
    print(f"  armed; GUIDED takeoff to {args.alt:.0f} m ...")
    mav.mav.command_long_send(mav.target_system, AP, M.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, args.alt)
    end = time.time() + 120
    while time.time() < end:
        v = vfr(mav, 2)
        if v and (v.alt - home) >= args.alt - 6:
            break

    # Switch to STABILIZE and HOLD (thrust 0.5) with a deliberately-low hover seed.
    for i in range(1, 7):
        set_param(mav, f"FLTMODE{i}", STABILIZE)
    mav.set_mode(STABILIZE)
    print(f"\n  STABILIZE, hover seed={args.seed_hover} (deliberately low). Commanding HOLD;")
    print(f"  watching the learner for {args.watch:.0f}s:\n")
    print(f"  {'t':>4}  {'AGL m':>7}  {'climb m/s':>9}  {'hover_pwm':>9}")
    hold = GuidanceIntent(0.0, 0.0, 0.0, 0.5, time.monotonic())
    start = time.time()
    nextp = 0.0
    climbs = []
    while time.time() - start < args.watch:
        backend.send_intent(hold)
        backend._drain()          # updates backend._climb_mps from VFR_HUD
        el = time.time() - start
        if el >= nextp:
            v = vfr(mav, 0.5)
            agl = (v.alt - home) if v else float("nan")
            print(f"  {el:4.0f}  {agl:7.1f}  {backend._climb_mps:9.2f}  {backend._hover_pwm:9.0f}")
            if el > 6:
                climbs.append(backend._climb_mps)
            nextp += 2.0
        time.sleep(0.05)

    settled = (sum(abs(c) for c in climbs) / len(climbs)) if climbs else 99.0
    print(f"\n  learned hover_pwm = {backend._hover_pwm:.0f} (seed was {args.seed_hover})")
    print(f"  mean |climb| after settle = {settled:.2f} m/s")
    ok = settled < 0.6
    print(f"\n  {'PASS' if ok else 'NEEDS REVIEW'}: adaptive hover "
          f"{'converged to level' if ok else 'did not settle'}")

    backend.release()
    mav.set_mode(GUIDED)
    mav.mav.command_long_send(mav.target_system, AP, M.MAV_CMD_NAV_LAND, 0, 0, 0, 0, 0, 0, 0, 0)
    backend.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
