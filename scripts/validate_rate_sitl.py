#!/usr/bin/env python3
"""SITL validation of the PRODUCTION guided_nogps body-rate law — no Gazebo needed.

sitl_gz_rate.py exercises this same law but needs ROS 2 + a Gazebo camera topic. That
left the rate path — the one the aircraft actually flies with control_mode:
guided_nogps — with NO runnable SITL check. This drives
`rate_control.compute_rate_intent` (the shipped controller, not a copy) against real
ArduCopter physics using a PROJECTED ground target, and asserts the output-conditioning
properties: bounded throttle, bounded attitude, incremental commands, no oscillation,
plus closed-loop FOV retention (does the pitch-fold keep the target in frame?).

The target is a fixed point on the ground whose pixel position is computed each tick
by projecting through the vehicle's current attitude — so as the vehicle pitches 45°
nose-down, the target moves realistically from the bottom of the frame toward the top,
exactly as a body-fixed camera would see it. Pass --static-target to revert to the
original constant (0.52, 0.80) for pure output-conditioning checks.

    docker run -d --rm --name pifpv-sitl -p 5760:5760 pifpv-sitl:4.6
    .venv/bin/python scripts/validate_rate_sitl.py --connect tcp:127.0.0.1:5760

SAFETY: this ARMS the vehicle, so it refuses anything that is not a tcp/udp SITL
connection. It must never be pointed at a serial device / real flight controller.
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
ALT_HOLD, GUIDED, GUIDED_NOGPS = 2, 4, 20


# --- Geometric target projection ------------------------------------------------

def _ned_to_body(n, e, d, pitch, roll, yaw):
    """Rotate a NED vector into body frame (X-fwd Y-right Z-down) via 321 Euler."""
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(yaw), math.sin(yaw)
    bx = cp * cy * n + cp * sy * e - sp * d
    by = (sr * sp * cy - cr * sy) * n + (sr * sp * sy + cr * cy) * e + sr * cp * d
    bz = (cr * sp * cy + sr * sy) * n + (cr * sp * sy - sr * cy) * e + cr * cp * d
    return bx, by, bz


def _body_to_ned(bx, by, bz, pitch, roll, yaw):
    """Rotate body frame to NED (transpose of NED-to-body)."""
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(yaw), math.sin(yaw)
    n = cp * cy * bx + (sr * sp * cy - cr * sy) * by + (cr * sp * cy + sr * sy) * bz
    e = cp * sy * bx + (sr * sp * sy + cr * cy) * by + (cr * sp * sy - sr * cy) * bz
    d = -sp * bx + sr * cp * by + cr * cp * bz
    return n, e, d


def _place_ground_target(cx_n, cy_n, agl, yaw, hfov_rad, vfov_rad):
    """NED offset of a ground target that appears at (cx_n, cy_n) in a level camera
    at height `agl` with heading `yaw`.  Returns (north, east) relative to the vehicle."""
    ry = (cx_n - 0.5) * 2.0 * math.tan(hfov_rad / 2.0)
    rz = (cy_n - 0.5) * 2.0 * math.tan(vfov_rad / 2.0)
    rx = 1.0
    rn, re, rd = _body_to_ned(rx, ry, rz, 0.0, 0.0, yaw)
    if rd <= 1e-6:
        raise ValueError(f"pixel ({cx_n},{cy_n}) does not point at ground")
    t = agl / rd
    return t * rn, t * re


def _project_to_frame(rel_n, rel_e, agl, pitch, roll, yaw, hfov_rad, vfov_rad):
    """Project a ground target into normalised camera coords.
    `rel_n/e` = horizontal NED from vehicle to target, `agl` = height above target.
    Returns (cx_n, cy_n, slant_range, in_fov)."""
    bx, by, bz = _ned_to_body(rel_n, rel_e, agl, pitch, roll, yaw)
    if bx <= 0.01:
        return 0.5, 0.5, 0.0, False
    half_h = math.tan(hfov_rad / 2.0)
    half_v = math.tan(vfov_rad / 2.0)
    cx_n = 0.5 + (by / bx) / (2.0 * half_h)
    cy_n = 0.5 + (bz / bx) / (2.0 * half_v)
    dist = math.sqrt(rel_n ** 2 + rel_e ** 2 + agl ** 2)
    in_fov = -0.05 <= cx_n <= 1.05 and -0.05 <= cy_n <= 1.05
    return cx_n, cy_n, dist, in_fov


# --- Helpers ---------------------------------------------------------------------

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
    ap.add_argument("--climb-timeout", type=float, default=90.0,
                    help="seconds to wait for climb (increase for emulated SITL)")
    ap.add_argument("--static-target", action="store_true",
                    help="fixed (0.52, 0.80) target — no geometric projection")
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
    for name, val in (("GUID_OPTIONS", 72), ("ARMING_CHECK", 0)):
        m.mav.param_set_send(m.target_system, m.target_component, name.encode(),
                             float(val), mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
        time.sleep(0.3)
    print("  GUID_OPTIONS=72 (ThrustAsThrust+BodyFrameRates), ARMING_CHECK=0")

    # ---- Climb to altitude in ALT_HOLD -----------------------------------------
    if not set_mode(m, ALT_HOLD):
        print("could not enter ALT_HOLD"); return 1
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
    print(f"  climbing to {args.alt:.0f}m AGL via RC throttle ...")
    end = time.time() + args.climb_timeout
    agl = 0.0
    while time.time() < end:
        m.mav.rc_channels_override_send(m.target_system, m.target_component,
                                        1500, 1500, 1800, 1500, 0, 0, 0, 0)
        gpi = m.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=2)
        if gpi:
            agl = gpi.relative_alt / 1000.0
            if agl >= args.alt * 0.92:
                break
        time.sleep(0.05)
    m.mav.rc_channels_override_send(m.target_system, m.target_component,
                                    1500, 1500, 1500, 1500, 0, 0, 0, 0)
    print(f"  at {agl:.1f}m AGL")
    if agl < args.alt * 0.7:
        print(f"ABORT: only reached {agl:.1f}m AGL of {args.alt:.0f}m. A grounded vehicle "
              "passes every bounds check vacuously — refusing to report a result.")
        return 1
    if not set_mode(m, GUIDED_NOGPS):
        print("could not enter GUIDED_NOGPS"); return 1
    print("  in GUIDED_NOGPS — driving the production rate law")

    # ---- Rate-law setup --------------------------------------------------------
    cfg = RateConfig(W, H)
    hfov_rad = math.radians(cfg.hfov_deg)
    vfov_rad = math.radians(cfg.vfov_deg)
    focal_v = (H / 2.0) / math.tan(vfov_rad / 2.0)

    # ---- Projected-target initialisation ---------------------------------------
    target_n = target_e = target_size_m = 0.0
    lat0 = lon0 = lat_cur = lon_cur = None
    lat_scale = lon_scale = 0.0
    yaw0 = 0.0

    if not args.static_target:
        print("  capturing initial state for projected target ...")
        for _ in range(60):
            msg = m.recv_match(type=["ATTITUDE", "GLOBAL_POSITION_INT"],
                               blocking=True, timeout=0.5)
            if msg is None:
                continue
            if msg.get_type() == "ATTITUDE":
                yaw0 = msg.yaw
            elif msg.get_type() == "GLOBAL_POSITION_INT":
                lat0 = msg.lat
                lon0 = msg.lon
                agl = msg.relative_alt / 1000.0
        if lat0 is None:
            print("ABORT: no GLOBAL_POSITION_INT — cannot place target"); return 1
        lat_cur, lon_cur = lat0, lon0
        target_n, target_e = _place_ground_target(0.52, 0.80, agl, yaw0,
                                                   hfov_rad, vfov_rad)
        init_range = math.sqrt(target_n ** 2 + target_e ** 2 + agl ** 2)
        target_size_m = 40.0 * init_range / focal_v
        lat_scale = 1.113195e5 * 1e-7
        lon_scale = 1.113195e5 * math.cos(math.radians(lat0 * 1e-7)) * 1e-7
        ground_dist = math.sqrt(target_n ** 2 + target_e ** 2)
        print(f"  target: {ground_dist:.0f}m ground, {init_range:.0f}m slant, "
              f"{target_size_m:.1f}m physical, heading {math.degrees(yaw0):.0f} deg\n")
    else:
        print("  static target at (0.52, 0.80)\n")

    st = RateState()
    st.hover = 0.35
    st.sm_thr = st.hover
    dt = 1.0 / args.hz

    rec = {"thrust": [], "climb": [], "pitch": [], "roll": [], "alt": [],
           "yaw_rate": [], "d_thrust": [], "cmd_mode": [], "phase": [],
           "in_fov": [], "target_cy": []}
    prev_thr = st.hover
    prev_t = None
    t0 = time.monotonic()
    phase_end = t0 + args.track_s
    mode = GuidanceMode.TRACK
    pitch = roll = yaw_cur = climb = 0.0
    alt_now = agl
    att_count = vfr_count = gpi_count = 0

    while True:
        now = time.monotonic()
        if mode is GuidanceMode.TRACK and now >= phase_end:
            mode = GuidanceMode.DIVE
            st.reset(); st.sm_thr = st.hover
            phase_end = now + args.dive_s
            print(f"  -> DIVE at t={now - t0:.1f}s agl={alt_now:.1f}m")
        elif mode is GuidanceMode.DIVE and now >= phase_end:
            break

        for _ in range(10):
            msg = m.recv_match(type=["ATTITUDE", "VFR_HUD", "GLOBAL_POSITION_INT"],
                               blocking=True, timeout=0.005)
            if msg is None:
                break
            mt = msg.get_type()
            if mt == "ATTITUDE":
                pitch, roll, yaw_cur = msg.pitch, msg.roll, msg.yaw
                att_count += 1
            elif mt == "VFR_HUD":
                climb = msg.climb
                vfr_count += 1
            elif mt == "GLOBAL_POSITION_INT":
                alt_now = msg.relative_alt / 1000.0
                lat_cur = msg.lat
                lon_cur = msg.lon
                gpi_count += 1

        # --- Synthetic target: projected or static ------------------------------
        tgt_cy = 0.80
        if args.static_target:
            tgt = _ft(0.52, 0.80, 40, now)
            in_fov = True
        else:
            veh_n = (lat_cur - lat0) * lat_scale
            veh_e = (lon_cur - lon0) * lon_scale
            rel_n = target_n - veh_n
            rel_e = target_e - veh_e
            cx_proj, cy_proj, slant, in_fov = _project_to_frame(
                rel_n, rel_e, max(alt_now, 0.1),
                pitch, roll, yaw_cur, hfov_rad, vfov_rad)
            if in_fov:
                h_px = max(4.0, target_size_m * focal_v / max(slant, 1.0))
                tgt = _ft(min(max(cx_proj, 0.0), 1.0),
                          min(max(cy_proj, 0.0), 1.0), h_px, now)
                tgt_cy = cy_proj
            else:
                tgt = None

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
        rec["cmd_mode"].append("TRACK" if mode is GuidanceMode.TRACK else "DIVE")
        rec["phase"].append(ri.phase)
        rec["in_fov"].append(in_fov)
        rec["target_cy"].append(tgt_cy)
        if prev_t is not None:
            real_dt = now - prev_t
            if real_dt > 1e-4:
                rec["d_thrust"].append(abs(ri.thrust - prev_thr) / real_dt)
        prev_t = now
        prev_thr = ri.thrust
        time.sleep(dt)

    # ---- Land and report -------------------------------------------------------
    set_mode(m, 9)   # LAND
    elapsed = time.monotonic() - t0
    print(f"\n  telemetry: {att_count} ATT, {vfr_count} VFR, {gpi_count} GPI in {elapsed:.1f}s")
    print(f"  samples: {len(rec['thrust'])} control ticks")
    print("\n=== results ===")
    ok = True

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
    sc = sum(1 for a, b in zip(rec["yaw_rate"], rec["yaw_rate"][1:]) if a * b < 0)
    per_s = sc / max(1e-6, len(rec["yaw_rate"]) * dt)
    chk("no yaw oscillation", per_s < 3.0, f"{per_s:.2f} sign changes/s")

    # ---- FOV retention (projected target only) ---------------------------------
    if not args.static_target:
        n_track = sum(1 for m in rec["cmd_mode"] if m == "TRACK")
        n_dive = sum(1 for m in rec["cmd_mode"] if m == "DIVE")
        fov_track = sum(1 for m, f in zip(rec["cmd_mode"], rec["in_fov"])
                        if m == "TRACK" and f)
        fov_dive = sum(1 for m, f in zip(rec["cmd_mode"], rec["in_fov"])
                       if m == "DIVE" and f)
        pct_track = 100.0 * fov_track / max(n_track, 1)
        pct_dive = 100.0 * fov_dive / max(n_dive, 1)
        chk("FOV retention (TRACK)", pct_track > 90.0,
            f"{pct_track:.0f}% of {n_track} ticks")
        chk("FOV retention (DIVE)", pct_dive > 50.0,
            f"{pct_dive:.0f}% of {n_dive} ticks")

        # Show where the target sat in the frame during each phase.
        for label, cmd in (("TRACK", "TRACK"), ("DIVE", "DIVE")):
            cy_vals = [c for m, c, f in zip(rec["cmd_mode"], rec["target_cy"], rec["in_fov"])
                       if m == cmd and f]
            if cy_vals:
                print(f"  target cy during {label}: "
                      f"{min(cy_vals):.2f} – {max(cy_vals):.2f} (goal {cfg.vert_goal:.2f})")

        # How many ticks entered SEARCH (target lost, controller nose-down to reacquire).
        n_search = sum(1 for p in rec["phase"] if p == "SEARCH")
        if n_search:
            print(f"  SEARCH ticks (target lost): {n_search}")

    print(f"\n  AGL {rec['alt'][0]:.1f} -> {rec['alt'][-1]:.1f} m over "
          f"{len(rec['alt']) * dt:.0f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
