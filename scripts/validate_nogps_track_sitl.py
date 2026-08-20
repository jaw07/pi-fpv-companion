#!/usr/bin/env python3
"""Prove TRACK engages and flies with GPS GENUINELY ABSENT — the real flight sequence.

Engaging TRACK on the airframe failed with "needs an altitude/position estimate".
validate_rate_sitl.py could not have caught that: it climbs using GUIDED, which needs
GPS, so it silently validated the rate law on a GPS-equipped vehicle.

This turns GPS OFF in the simulator and REBOOTS the autopilot, so it comes up with no
GPS at all, then walks the ACTUAL flight sequence:

    arm in ALT_HOLD (pilot mode, baro only)  ->  climb on the throttle stick
      ->  switch to GUIDED_NOGPS (what ch7/auto_guided does)
      ->  drive the production rate law in TRACK  ->  confirm it responds

Each step is reported pass/fail with the FC's own reason, so a failure says WHICH
link broke rather than just "did not work".

    docker run -d --rm --name pifpv-sitl -p 5760:5760 pifpv-sitl:4.6
    .venv/bin/python scripts/validate_nogps_track_sitl.py

SAFETY: arms, so it refuses any non-SITL connection.
"""
from __future__ import annotations
import argparse, math, sys, time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from pymavlink import mavutil
from pi_fpv_companion.guidance.rate_control import RateConfig, RateState, compute_rate_intent
from pi_fpv_companion.types import Detection, FilteredTarget, GuidanceMode

W, H = 720, 576
STABILIZE, ALT_HOLD, GUIDED_NOGPS = 0, 2, 20

# The EKF source set under test — the same values config/imx500.yaml enforces.
NOGPS_PARAMS = {
    "EK3_SRC1_POSXY": 0, "EK3_SRC1_VELXY": 0,
    "EK3_SRC1_POSZ": 1,  "EK3_SRC1_VELZ": 0,
    "GPS_TYPE": 0, "SIM_GPS_DISABLE": 1, "SIM_GPS1_ENABLE": 0,
    "ARMING_CHECK": 0, "GUID_OPTIONS": 8,
}


def drain_text(m, seen, quiet=False):
    while True:
        s = m.recv_match(type="STATUSTEXT", blocking=False)
        if s is None:
            return
        if s.text not in seen:
            seen.add(s.text)
            if not quiet:
                print(f"      FC: {s.text}")


def set_param(m, name, val, tries=6):
    for _ in range(tries):
        m.mav.param_set_send(m.target_system, m.target_component, name.encode(),
                             float(val), mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
        t0 = time.time()
        while time.time() - t0 < 1.5:
            p = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=1)
            if p and p.param_id.strip("\x00") == name:
                return abs(p.param_value - float(val)) < 1e-6
    return False


def set_mode(m, mode, seen, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        m.mav.set_mode_send(m.target_system,
                            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode)
        t0 = time.time()
        while time.time() - t0 < 2:
            h = m.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
            drain_text(m, seen)
            if h and h.custom_mode == mode:
                return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--connect", default="tcp:127.0.0.1:5760")
    ap.add_argument("--alt", type=float, default=25.0)
    ap.add_argument("--track-s", type=float, default=12.0)
    args = ap.parse_args()

    if not args.connect.startswith(("tcp:", "udp:", "udpin:", "udpout:")):
        print(f"REFUSING: {args.connect!r} is not a SITL socket; this script ARMS.")
        return 2

    results = []
    def step(label, ok, detail=""):
        results.append((label, ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
        return ok

    m = mavutil.mavlink_connection(args.connect)
    m.wait_heartbeat(timeout=60)
    seen = set()
    print("connected\n--- applying GPS-denied param set ---")
    for k, v in NOGPS_PARAMS.items():
        ok = set_param(m, k, v)
        print(f"  {k:<16} = {v}   {'ok' if ok else 'NOT SET (may not exist on this build)'}")

    print("\n--- rebooting autopilot so it comes up with no GPS ---")
    m.mav.command_long_send(m.target_system, m.target_component,
                            mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN, 0,
                            1, 0, 0, 0, 0, 0, 0)
    time.sleep(6)
    m = mavutil.mavlink_connection(args.connect)
    m.wait_heartbeat(timeout=90)
    # GLOBAL_POSITION_INT is essential here: VFR_HUD.alt reads 0.0 for the whole flight
    # when GPS is disabled (measured), so a harness that trusts it concludes the vehicle
    # never took off. relative_alt comes from the EKF/baro and survives GPS-denied.
    for msg in ("ATTITUDE", "VFR_HUD", "GPS_RAW_INT", "STATUSTEXT", "GLOBAL_POSITION_INT"):
        m.mav.command_long_send(m.target_system, m.target_component,
                                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                                getattr(mavutil.mavlink, f"MAVLINK_MSG_ID_{msg}"),
                                100000, 0, 0, 0, 0, 0)
    print("  rebooted")
    g = m.recv_match(type="GPS_RAW_INT", blocking=True, timeout=8)
    fix = getattr(g, "fix_type", None)
    step("GPS is genuinely absent", fix in (None, 0, 1), f"fix_type={fix}")

    seen = set()
    print("\n--- flight sequence ---")
    step("enter ALT_HOLD (baro-only pilot mode)", set_mode(m, ALT_HOLD, seen))

    print("    arming (waiting for IMU/EKF settle) ...")
    armed = False
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
                if a.text not in seen:
                    seen.add(a.text); print(f"      FC: {a.text}")
            elif a.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
                armed = a.result == mavutil.mavlink.MAV_RESULT_ACCEPTED
                break
    if not step("ARM without GPS", armed):
        print("\n  -> could not arm GPS-free; nothing downstream is testable.")
        return 1

    # Climb on the throttle stick. Use STABILIZE for the climb: its throttle is DIRECT,
    # so there is no altitude controller to satisfy and no takeoff detection to trip.
    # (ALT_HOLD's mid-stick-holds behaviour left the vehicle sitting on the ground.)
    print("    climbing on the throttle stick (STABILIZE, direct throttle) ...")
    set_mode(m, STABILIZE, seen, timeout=10)
    alt0 = None
    end = time.time() + 90
    alt = 0.0
    thr_out = 0
    while time.time() < end:
        m.mav.rc_channels_override_send(m.target_system, m.target_component,
                                        1500, 1500, 1900, 1500, 0, 0, 0, 0)
        m.recv_match(type=["VFR_HUD", "GLOBAL_POSITION_INT"], blocking=True, timeout=1)
        drain_text(m, seen)
        g = m.messages.get("GLOBAL_POSITION_INT")
        v = m.messages.get("VFR_HUD")
        if g is not None:
            if alt0 is None:
                alt0 = g.relative_alt / 1000.0
            alt = g.relative_alt / 1000.0
            thr_out = getattr(v, "throttle", 0)
            if alt - alt0 >= args.alt:
                break
    climbed = alt0 is not None and (alt - alt0) >= args.alt * 0.6
    step("climb on baro alone (relative_alt)", climbed,
         f"gained {alt - (alt0 or 0):.1f}m (FC throttle out {thr_out}%)")
    if not climbed:
        print("      note: motors did not lift it; the TRACK checks below cannot "
              "validate flight behaviour, only the commands the companion issues.")
    # LEVEL OFF before engaging, as a pilot would. Entering TRACK at 10m/s of residual
    # climb is not a fair test of "does TRACK hold altitude" — momentum carries the
    # aircraft for a long way whatever thrust is commanded, and the hover trim
    # deliberately ignores large climb rates (they say nothing about hover).
    print("    levelling off before engage ...")
    set_mode(m, ALT_HOLD, seen, timeout=10)
    end = time.time() + 30
    while time.time() < end:
        m.mav.rc_channels_override_send(m.target_system, m.target_component,
                                        1500, 1500, 1500, 1500, 0, 0, 0, 0)
        v = m.recv_match(type="VFR_HUD", blocking=True, timeout=1)
        drain_text(m, seen)
        if v and abs(v.climb) < 1.0:
            break
    m.mav.rc_channels_override_send(m.target_system, m.target_component,
                                    0, 0, 0, 0, 0, 0, 0, 0)
    print(f"      climb at engage: {getattr(v, 'climb', float('nan')):+.2f} m/s")

    # Hand the sticks back before the mode change (this is what release() does).
    m.mav.rc_channels_override_send(m.target_system, m.target_component,
                                    0, 0, 0, 0, 0, 0, 0, 0)

    entered = set_mode(m, GUIDED_NOGPS, seen)
    step("enter GUIDED_NOGPS without GPS  <-- the ch7/TRACK engage", entered)
    if not entered:
        print("\n  -> TRACK cannot engage GPS-free. The EKF source set is not sufficient.")
        return 1

    # Drive the production TRACK law and confirm the airframe responds.
    cfg, st = RateConfig(W, H), RateState()
    st.hover = 0.45; st.sm_thr = st.hover   # starting guess; trimmed online below
    dt = 1 / 22.0
    yaws, thrusts, alts = [], [], []
    pitch = roll = climb = 0.0
    t0 = time.monotonic()
    while time.monotonic() - t0 < args.track_s:
        now = time.monotonic()
        while True:
            msg = m.recv_match(type=["ATTITUDE", "VFR_HUD", "GLOBAL_POSITION_INT"],
                               blocking=False)
            if msg is None:
                break
            if msg.get_type() == "ATTITUDE":
                pitch, roll = msg.pitch, msg.roll
            elif msg.get_type() == "GLOBAL_POSITION_INT":
                alt = msg.relative_alt / 1000.0      # NOT VFR_HUD.alt — see above
            else:
                climb = msg.climb
        # Target off to the RIGHT so TRACK must yaw toward it — a measurable response.
        tgt = FilteredTarget(
            detection=Detection(x=0.85 * W, y=0.5 * H, w=40, h=40, confidence=0.9, class_id=0),
            track_id=1, vx_px_s=0.0, vy_px_s=0.0, quality=0.9, timestamp=now)
        # Mirror the pipeline's ONLINE HOVER TRIM. TRACK holds altitude by commanding
        # thrust = state.hover, so without the learning loop a hardcoded hover guess makes
        # it climb or sink indefinitely — which is a harness artifact, not a controller
        # fault. The real trim runs in Pipeline._tick_rate and keys off VFR_HUD.climb
        # (which, unlike VFR_HUD.alt, DOES survive GPS-denied).
        if abs(pitch) < 0.26:
            from pi_fpv_companion.guidance.rate_control import trim_hover
            st.hover = trim_hover(st.hover, climb, dt, cfg)
        ri = compute_rate_intent(tgt, cfg, st, now=now, mode=GuidanceMode.TRACK,
                                 pitch_rad=pitch, roll_rad=roll,
                                 gamma_rad=math.atan2(climb, 5.0), agl_m=max(alt, 0.0))
        m.mav.set_attitude_target_send(
            int((now - t0) * 1000) & 0xFFFFFFFF, m.target_system, m.target_component,
            0b10000000, [1.0, 0.0, 0.0, 0.0],
            ri.roll_rate, ri.pitch_rate, ri.yaw_rate, ri.thrust)
        yaws.append(ri.yaw_rate); thrusts.append(ri.thrust); alts.append(alt)
        time.sleep(dt)

    drain_text(m, seen)
    step("TRACK commanded a yaw toward the target",
         max(abs(y) for y in yaws) > 0.05, f"peak |yaw| {max(abs(y) for y in yaws):.3f} rad/s")
    floor = cfg.min_thrust_frac * st.hover
    step("throttle stayed above the floor", min(thrusts) >= floor - 1e-6,
         f"min {min(thrusts):.3f} floor {floor:.3f}")
    # Judge the SETTLED half: the first moments after engage are an entry transient
    # (whatever residual climb existed is still being arrested), not a hold failure.
    settled = alts[len(alts) // 2:]
    alt_span = max(settled) - min(settled)
    if climbed:
        step("held altitude in TRACK (no runaway)", alt_span < 15.0,
             f"settled span {alt_span:.1f}m (full {max(alts)-min(alts):.1f}m), "
             f"hover trimmed to {st.hover:.3f}")
    else:
        print(f"  [SKIP] held altitude in TRACK — vehicle never flew "
              f"(alt span {alt_span:.1f}m would pass vacuously)")
    step("still in GUIDED_NOGPS at the end",
         (m.recv_match(type="HEARTBEAT", blocking=True, timeout=5) or
          type("x", (), {"custom_mode": -1})).custom_mode == GUIDED_NOGPS)

    set_mode(m, ALT_HOLD, seen, timeout=8)
    m.mav.command_long_send(m.target_system, m.target_component,
                            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 0, 21196, 0,0,0,0,0)
    bad = [l for l, ok in results if not ok]
    print(f"\n=== {len(results) - len(bad)}/{len(results)} passed ===")
    if bad:
        print("  failed: " + ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
