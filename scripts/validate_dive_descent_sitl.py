"""SITL check: a GENTLE geometry-matched dive actually descends (hold-band fix).

The agnostic DIVE only offsets throttle by `dive_descent` (~0.12), so the
adaptive-hover hold band MUST be below that or the hover PI loop cancels the
descent and the aircraft never dives (see docs/guidance.md). Unit tests
cover the backend math; this confirms it end-to-end on ArduCopter: drive the
PRODUCTION ArduPilotBackend (STABILIZE, adaptive hover ON) with a gentle dive
intent and a gentle climb intent and measure real altitude change.

    docker run -d --rm --name pifpv-sitl -p 127.0.0.1:5760:5760 pifpv-sitl:4.6
    .venv/bin/python scripts/validate_dive_descent_sitl.py --connect tcp:127.0.0.1:5760
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


def vfr(mav, timeout=3.0):
    return mav.recv_match(type="VFR_HUD", blocking=True, timeout=timeout)


def _alt(mav, home_alt):
    v = vfr(mav, 3)
    return (v.alt if v else home_alt) - home_alt


def fly_segment(backend, mav, M, label, thrust, secs, home_alt):
    """Hold a fixed intent (gentle pitch + given thrust) and measure descent.

    Drains telemetry every tick (as the pipeline does via read_switch) so VFR_HUD
    feeds the adaptive-hover loop — the gentle dive's small throttle offset only
    descends correctly when the learned hover is accurate."""
    a0 = _alt(mav, home_alt)
    intent = GuidanceIntent(0.0, -6.0, 0.0, thrust, time.monotonic())
    end = time.time() + secs
    while time.time() < end:
        backend._drain()              # feed VFR_HUD -> adaptive hover (pipeline does this)
        backend.send_intent(intent)
        time.sleep(0.05)
    a1 = _alt(mav, home_alt)
    rate = (a1 - a0) / secs           # +climb, -descend
    print(f"  [{label}] thrust={thrust:.2f}  alt {a0:5.1f}->{a1:5.1f} AGL  "
          f"rate {rate:+5.2f} m/s  hover_pwm={backend._hover_pwm:.0f}")
    return rate


def main():
    from pymavlink import mavutil
    M = mavutil.mavlink
    ap = argparse.ArgumentParser()
    ap.add_argument("--connect", default="tcp:127.0.0.1:5760")
    ap.add_argument("--alt", type=float, default=120.0)
    args = ap.parse_args()

    # Production backend: STABILIZE + adaptive hover ON, shipped hold band 0.05.
    backend = ArduPilotBackend(
        device=args.connect, baud=0, switch_channel=7,
        track_threshold_us=1300, dive_threshold_us=1700,
        mapping=ArduCopterRcMapping(control_mode="stabilize", hover_learn=True,
                                    hover_learn_band=0.05),
    )
    backend.open()
    mav = backend._mav
    print(f"connecting to {args.connect} ...")
    mav.wait_heartbeat(timeout=90)
    mav.target_component = AP
    mav.mav.request_data_stream_send(mav.target_system, AP, M.MAV_DATA_STREAM_ALL, 10, 1)
    backend._request_streams()
    print("  settling 60s for EKF ...")
    time.sleep(60)
    v0 = vfr(mav, 5)
    home_alt = v0.alt if v0 else 0.0

    # GUIDED takeoff (reliable), then hand to STABILIZE for the RC-override dive.
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
    mav.mav.command_long_send(mav.target_system, AP, M.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, args.alt)
    end = time.time() + 150
    while time.time() < end:
        v = vfr(mav, 2)
        if v and v.alt - home_alt >= args.alt - 6:
            break
    print(f"  reached {(vfr(mav,3).alt - home_alt):.0f} m AGL\n")

    for i in range(1, 7):
        backend._mav.mav.param_set_send(mav.target_system, AP, f"FLTMODE{i}".encode(),
                                        float(STABILIZE), M.MAV_PARAM_TYPE_INT32)
    mav.set_mode(STABILIZE)
    time.sleep(1.0)

    # Let the adaptive-hover loop learn the real hover (draining VFR_HUD + holding
    # neutral) before the gentle dive — otherwise a wrong hover guess swamps the
    # small dive throttle offset.
    print("  converging adaptive hover (STABILIZE, neutral) 20s ...")
    hold_i = GuidanceIntent(0.0, 0.0, 0.0, 0.5, time.monotonic())
    end = time.time() + 20
    while time.time() < end:
        backend._drain()
        backend.send_intent(hold_i)
        time.sleep(0.05)
    print(f"  learned hover_pwm={backend._hover_pwm:.0f}, "
          f"climb_fresh={(time.monotonic()-backend._climb_t)<0.5 if backend._climb_t else False}\n")

    print(f"Gentle agnostic-dive commits through the production backend (alt {args.alt:.0f} m):")
    hold = fly_segment(backend, mav, M, "HOLD  (neutral)    ", 0.50, 4.0, home_alt)
    dive = fly_segment(backend, mav, M, "DIVE  (gentle 0.38)", 0.38, 5.0, home_alt)
    climb = fly_segment(backend, mav, M, "CLIMB (gentle 0.62)", 0.62, 4.0, home_alt)

    backend.send_intent(GuidanceIntent(0, 0, 0, 0.5, time.monotonic()))
    backend.release()
    ok = dive < -0.5 and climb > 0.3
    print(f"\n{'PASS' if ok else 'FAIL'}: gentle dive descends ({dive:+.2f}) and gentle climb "
          f"climbs ({climb:+.2f}) — hold band does NOT swallow them (hold {hold:+.2f}).")
    mav.set_mode(GUIDED)
    mav.mav.command_long_send(mav.target_system, AP, M.MAV_CMD_NAV_LAND, 0, 0, 0, 0, 0, 0, 0, 0)
    backend.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
