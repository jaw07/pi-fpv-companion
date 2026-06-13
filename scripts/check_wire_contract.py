#!/usr/bin/env python3
r"""Verify the STANDBY safety command contract on a live MAVLink stream or a tlog.

This is the Tier-1 "prove it on the wire" bench tool: it watches the ACTUAL
MAVLink traffic between the companion and the FC and asserts that STANDBY injects
nothing, disarmed injects nothing, etc. (see pi_fpv_companion.safety_contract).

Because the companion owns the FC UART, tap the wire with mavproxy so this tool
gets an independent copy of BOTH directions:

    # on the Pi, FC on /dev/serial0:
    mavproxy.py --master=/dev/serial0,115200 \
        --out=udp:127.0.0.1:14550 \      # point the companion's fc.uart_device here
        --out=udp:127.0.0.1:14551 \      # this tool reads here
        --daemon --state-basedir=/tmp --logfile=/tmp/wire.tlog

    # then run the companion (uart_device: udpout:127.0.0.1:14550) and engage/disengage,
    # arm/disarm, on the bench (PROPS OFF), and run this against the other output:
    python scripts/check_wire_contract.py --device udpin:127.0.0.1:14551 --duration 120

Or analyse the captured tlog afterwards:
    python scripts/check_wire_contract.py --tlog /tmp/wire.tlog

Exit code is non-zero if any contract violation is found (CI/bench-gate friendly).
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pi_fpv_companion.safety_contract import ContractChecker, ContractConfig  # noqa: E402


def _feed(checker: ContractChecker, msg, switch_channel: int) -> None:
    t = getattr(msg, "_timestamp", None) or time.time()
    mt = msg.get_type()
    if mt == "HEARTBEAT":
        # only the autopilot's heartbeat carries the armed bit
        from pymavlink import mavutil
        if getattr(msg, "autopilot", 0) != mavutil.mavlink.MAV_AUTOPILOT_INVALID:
            armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            checker.on_heartbeat(t, armed)
    elif mt == "RC_CHANNELS":
        checker.on_rc_channels(t, getattr(msg, f"chan{switch_channel}_raw"))
    elif mt == "RC_CHANNELS_OVERRIDE":
        chans = [getattr(msg, f"chan{i}_raw", 0) for i in range(1, 9)]
        checker.on_rc_override(t, chans)
    elif mt == "SET_ATTITUDE_TARGET":
        checker.on_attitude_target(t)
    elif mt == "COMMAND_LONG":
        from pymavlink import mavutil
        if msg.command == mavutil.mavlink.MAV_CMD_DO_SET_MODE:
            checker.on_set_mode(t, int(msg.param2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--device", help="pymavlink connection (udpin:host:port, serial:/dev/...,...)")
    g.add_argument("--tlog", help="analyse a captured .tlog file instead of a live stream")
    ap.add_argument("--switch-channel", type=int, default=7)
    ap.add_argument("--track-threshold", type=int, default=1300)
    ap.add_argument("--duration", type=float, default=120.0, help="live capture seconds")
    args = ap.parse_args(argv)

    from pymavlink import mavutil
    cfg = ContractConfig(switch_channel=args.switch_channel,
                         track_threshold_us=args.track_threshold)
    checker = ContractChecker(cfg=cfg)

    if args.tlog:
        conn = mavutil.mavlink_connection(args.tlog)
        while True:
            msg = conn.recv_match(blocking=False)
            if msg is None:
                break
            _feed(checker, msg, args.switch_channel)
    else:
        conn = mavutil.mavlink_connection(args.device)
        print(f"capturing {args.device} for {args.duration:.0f}s "
              "(engage/disengage + arm/disarm now, PROPS OFF)...")
        end = time.time() + args.duration
        while time.time() < end:
            msg = conn.recv_match(blocking=True, timeout=1.0)
            if msg is not None:
                _feed(checker, msg, args.switch_channel)

    print(checker.report())
    return 0 if checker.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
