#!/usr/bin/env python3
"""Confirm VFR_HUD (and the other telemetry the hover learner needs) arrives."""
import time
import sys
from pymavlink import mavutil

m = mavutil.mavlink_connection("/dev/serial0", baud=115200)
print("waiting heartbeat...")
if m.wait_heartbeat(timeout=6) is None:
    print("NO HEARTBEAT")
    sys.exit(0)
# Request VFR_HUD @10Hz the way the backend does (id 74).
m.mav.command_long_send(m.target_system, m.target_component,
                        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                        74, 100000, 0, 0, 0, 0, 0)
seen = {}
shown = 0
t0 = time.time()
while time.time() - t0 < 6:
    msg = m.recv_match(blocking=False)
    if msg is None:
        time.sleep(0.005)
        continue
    t = msg.get_type()
    seen[t] = seen.get(t, 0) + 1
    if t == "VFR_HUD" and shown < 4:
        print("  VFR_HUD climb=%+.2f m/s  alt=%.2f  gs=%.2f  thr=%d%%"
              % (msg.climb, msg.alt, msg.groundspeed, msg.throttle))
        shown += 1
keys = ("VFR_HUD", "HEARTBEAT", "ATTITUDE", "RC_CHANNELS")
print("rates over 6s:", {k: seen.get(k, 0) for k in keys})
print("VFR_HUD streaming OK" if seen.get("VFR_HUD", 0) > 0 else "VFR_HUD NOT arriving")
