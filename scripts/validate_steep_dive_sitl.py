"""SITL check: ArduCopter STABILIZE achieves + holds the STEEP dive lean.

The adaptive dive lean commands up to ~25-30° nose-down on a ground attack (vs the
±12° the basic link check used). This confirms the production ArduPilotBackend's
RC-override pitch maps a steep lean correctly and the FC tracks it via
RC_CHANNELS_OVERRIDE (deflection = pitch_deg / angle_max_deg into STABILIZE).

    docker run -d --rm --name pifpv-sitl -p 127.0.0.1:5760:5760 pifpv-sitl:4.6
    .venv/bin/python scripts/validate_steep_dive_sitl.py --connect tcp:127.0.0.1:5760
"""
from __future__ import annotations
import argparse
import math
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


def hold_pitch(backend, mav, M, cmd_deg, secs):
    """Command a fixed nose-down pitch; return the settled measured pitch (deg)."""
    intent = GuidanceIntent(0.0, cmd_deg, 0.0, 0.5, time.monotonic())
    end = time.time() + secs
    pitches = []
    while time.time() < end:
        backend._drain()                 # parses ATTITUDE into the backend
        backend.send_intent(intent)
        p = backend.pitch_deg()          # +nose-up; backend's parsed ATTITUDE.pitch
        if p != 0.0:
            pitches.append(p)
        time.sleep(0.05)
    settled = pitches[-15:] if len(pitches) >= 15 else pitches
    return sum(settled) / len(settled) if settled else float("nan")


def main():
    from pymavlink import mavutil
    M = mavutil.mavlink
    ap = argparse.ArgumentParser()
    ap.add_argument("--connect", default="tcp:127.0.0.1:5760")
    ap.add_argument("--alt", type=float, default=120.0)
    args = ap.parse_args()
    backend = ArduPilotBackend(device=args.connect, baud=0, switch_channel=7,
                               track_threshold_us=1300, dive_threshold_us=1700,
                               mapping=ArduCopterRcMapping(control_mode="stabilize",
                                                           angle_max_deg=45.0, hover_learn=True))
    backend.open()
    mav = backend._mav
    print(f"connecting {args.connect} ...")
    mav.wait_heartbeat(timeout=90)
    mav.target_component = AP
    mav.mav.request_data_stream_send(mav.target_system, AP, M.MAV_DATA_STREAM_ALL, 10, 1)
    backend._request_streams()
    # Match the FC lean limit to the mapping so full stick = 45°.
    mav.mav.param_set_send(mav.target_system, AP, b"ANGLE_MAX", 4500.0, M.MAV_PARAM_TYPE_INT32)
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

    print(f"\nSteep-dive lean tracking (RC override into STABILIZE, ANGLE_MAX 45°):")
    results = []
    for cmd in (-10.0, -25.0, -30.0):
        meas = hold_pitch(backend, mav, M, cmd, 4.0)
        # arrest between cases so it doesn't run away
        backend.send_intent(GuidanceIntent(0, 0, 0, 0.5, time.monotonic()))
        for _ in range(20):
            backend._drain(); backend.send_intent(GuidanceIntent(0, 0, 0, 0.5, time.monotonic())); time.sleep(0.05)
        err = meas - cmd
        print(f"  cmd {cmd:+5.0f}°  ->  measured {meas:+6.1f}°  (err {err:+5.1f}°)")
        results.append((cmd, meas))
    backend.send_intent(GuidanceIntent(0, 0, 0, 0.5, time.monotonic()))
    backend.release()
    # Pass if the steep commands are tracked within a reasonable band and monotonic.
    ok = all(abs(m - c) < 6.0 for c, m in results) and results[2][1] < results[0][1]
    print(f"\n{'PASS' if ok else 'FAIL'}: STABILIZE tracks the steep dive lean "
          f"(25-30° achieved, within ~6° of command, monotonic).")
    mav.set_mode(GUIDED)
    mav.mav.command_long_send(mav.target_system, AP, M.MAV_CMD_NAV_LAND, 0, 0, 0, 0, 0, 0, 0, 0)
    backend.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
