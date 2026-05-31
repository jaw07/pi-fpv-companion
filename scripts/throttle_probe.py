#!/usr/bin/env python3
"""Isolated throttle->descent probe. Takes off, then at LEVEL attitude sweeps the
SET_ATTITUDE_TARGET thrust field (0.5 hover, 0.0 cut, 0.5, 1.0 climb) and logs altitude +
climb rate. Answers: does backing off the throttle actually drop the craft? Run with
--guided-opts 8 (ThrustAsThrust) or 0 (default climb-rate interpretation) to compare."""
import sys, time, math, argparse
from pymavlink import mavutil
M = mavutil.mavlink
GUIDED, GUIDED_NOGPS, AP = 4, 20, 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt", type=float, default=60.0)
    ap.add_argument("--guided-opts", type=int, default=8)
    a = ap.parse_args()
    m = mavutil.mavlink_connection("tcp:127.0.0.1:5760"); m.wait_heartbeat(timeout=60)
    m.target_component = AP
    m.mav.request_data_stream_send(m.target_system, AP, M.MAV_DATA_STREAM_ALL, 10, 1)
    for n, v in [("FRAME_CLASS", 1), ("FRAME_TYPE", 1), ("ARMING_CHECK", 0), ("GUID_OPTIONS", a.guided_opts)]:
        m.mav.param_set_send(m.target_system, AP, n.encode(), float(v), M.MAV_PARAM_TYPE_INT32); time.sleep(0.4)
    print("GUID_OPTIONS=%d (bit3 ThrustAsThrust=%s)" % (a.guided_opts, bool(a.guided_opts & 8)), flush=True)
    time.sleep(14)
    m.set_mode(GUIDED); time.sleep(1)
    t0 = time.time(); armed = False; last = 0
    while time.time() - t0 < 75 and not armed:
        if time.time() - last > 5:
            force = 21196 if (time.time() - t0 > 30) else 0
            m.mav.command_long_send(m.target_system, AP, M.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, force, 0, 0, 0, 0, 0); last = time.time()
        hb = m.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
        if hb: armed = bool(hb.base_mode & M.MAV_MODE_FLAG_SAFETY_ARMED)
    print("armed=%s; takeoff to %.0f" % (armed, a.alt), flush=True)
    m.mav.command_long_send(m.target_system, AP, M.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, a.alt)
    end = time.time() + 45; home = None
    while time.time() < end:
        v = m.recv_match(type="VFR_HUD", blocking=True, timeout=2)
        if v:
            if home is None: home = v.alt
            if v.alt - home >= a.alt - 1.0: break
    m.set_mode(GUIDED_NOGPS); time.sleep(1.0)
    print("at altitude; throttle sweep (level attitude, rates=0):", flush=True)

    def hold(thr, secs, label):
        t = time.time(); a0 = None; a1 = None
        while time.time() - t < secs:
            # mask 0b10000000: ignore attitude, use body rates (=0 -> hold level) + thrust
            m.mav.set_attitude_target_send(0, m.target_system, AP, 0b10000000, [1, 0, 0, 0], 0, 0, 0, thr)
            v = m.recv_match(type="VFR_HUD", blocking=False)
            if v:
                if a0 is None: a0 = v.alt
                a1 = v.alt; clb = v.climb
            time.sleep(0.05)
        if a0 is not None:
            print("  thr=%.2f %-6s alt %.1f->%.1f (%+.1f m in %.0fs = %+.1f m/s) climb=%+.1f" % (
                thr, label, a0, a1, a1 - a0, secs, (a1 - a0) / secs, clb), flush=True)

    hold(0.50, 3, "hover");  hold(0.00, 5, "CUT");  hold(0.50, 3, "hover");  hold(1.00, 4, "CLIMB"); hold(0.00, 5, "CUT2")
    m.set_mode(GUIDED); m.mav.command_long_send(m.target_system, AP, M.MAV_CMD_NAV_LAND, 0, 0, 0, 0, 0, 0, 0, 0)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
