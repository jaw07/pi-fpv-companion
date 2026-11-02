"""Direct, verified flight-mode setter. Never arms; refuses to run on an armed FC.

The census script's restore leaned on the backend's own _command_mode/_service_mode
retry machinery and FAILED to bring an FC back out of GUIDED_NOGPS, stranding it.
This does it directly and verifies from a FRESH heartbeat, alternating the legacy
SET_MODE and COMMAND_LONG/DO_SET_MODE encodings because some builds honour only one.

    .venv/bin/python var/hit/fc_mode_restore.py --device /dev/cu.usbmodem1101 --mode 0
"""
from __future__ import annotations
import argparse, sys, time
from pymavlink import mavutil

NAMES = {0: "STABILIZE", 2: "ALT_HOLD", 5: "LOITER", 20: "GUIDED_NOGPS"}


def read_mode(m, timeout=5.0):
    t0 = time.time()
    cur = None
    while time.time() - t0 < timeout:
        h = m.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        if h is not None:
            cur = (h.custom_mode, bool(h.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED))
    return cur


def set_mode(m, mode: int, attempts: int = 20) -> bool:
    for i in range(attempts):
        if i % 2 == 0:
            m.mav.set_mode_send(m.target_system,
                                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode)
        else:
            m.mav.command_long_send(m.target_system, m.target_component,
                                    mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
                                    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                                    mode, 0, 0, 0, 0, 0)
        cur = read_mode(m, 1.2)
        if cur and cur[0] == mode:
            print(f"  -> mode {mode} ({NAMES.get(mode,'?')}) CONFIRMED after {i+1} attempt(s)")
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="/dev/cu.usbmodem1101")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--mode", type=int, required=True)
    args = ap.parse_args()

    m = mavutil.mavlink_connection(args.device, baud=args.baud)
    m.wait_heartbeat(timeout=15)
    cur = read_mode(m)
    if cur is None:
        print("no heartbeat"); return 1
    mode_now, armed = cur
    print(f"before: mode={mode_now} ({NAMES.get(mode_now,'?')}) armed={armed}")
    if armed:
        print("ABORT: FC is ARMED. Refusing to change mode."); return 2
    if mode_now == args.mode:
        print("  already in the requested mode"); return 0
    ok = set_mode(m, args.mode)
    fin = read_mode(m)
    print(f"after:  mode={fin[0] if fin else '?'} armed={fin[1] if fin else '?'}")
    if not ok:
        print("FAILED — set it manually in your GCS")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
