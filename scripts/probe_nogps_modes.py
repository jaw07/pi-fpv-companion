"""Which ArduCopter flight mode can we use assuming we NEVER have GPS?

This disables GPS entirely in SITL (the right model for a bare analog FPV quad)
and, for each candidate mode, measures whether ArduCopter will:
  - ENTER the mode,
  - ARM in it,
  - STEER (yaw responds to the command channel that mode uses).

Candidates (both GPS-free, commanded via RC_CHANNELS_OVERRIDE sticks):
  - ALT_HOLD  (2) — self-levelling + baro altitude hold; the companion injects
    roll/pitch/yaw/throttle sticks. A normal *pilot* mode. THE CHOSEN PATH
    (ArduPilotBackend flies this; see docs/guidance.md).
  - STABILIZE (0) — self-levelling, manual throttle (fallback if no baro).

(GUIDED_NOGPS also works GPS-free but was retired in favour of ALT_HOLD; this
probe established that and the comparison is recorded in guidance.md.)

    docker run -d --rm --name pifpv-sitl --platform linux/amd64 \
        -p 5760:5760 radarku/ardupilot-sitl:latest
    .venv/bin/python scripts/probe_nogps_modes.py --connect tcp:127.0.0.1:5760

GPS is turned OFF here (SIM_GPS_DISABLE=1, GPS_TYPE=0, EK2_GPS_TYPE=3) and the
autopilot is rebooted so it comes up with no GPS at all — unlike the other SITL
scripts which lean on the sim GPS.
"""
from __future__ import annotations
import argparse
import math
import sys
import threading
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from pymavlink import mavutil

from pi_fpv_companion.fc.ardupilot import ArduPilotBackend

STABILIZE, ALT_HOLD = 0, 2
AP = 1  # autopilot component


class Checks:
    def __init__(self):
        self.results = []

    def record(self, name, ok, detail=""):
        self.results.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        return ok

    def summary(self):
        passed = sum(1 for _, ok, _ in self.results if ok)
        print(f"\n{'='*64}\n{passed}/{len(self.results)} checks passed")
        return 0 if passed == len(self.results) else 1


def set_param(mav, name, val, ptype="INT32"):
    from pymavlink import mavutil
    mav.mav.param_set_send(mav.target_system, AP, name.encode(), float(val),
                           getattr(mavutil.mavlink, f"MAV_PARAM_TYPE_{ptype}"))
    mav.mav.param_request_read_send(mav.target_system, AP, name.encode(), -1)
    end = time.time() + 8.0
    while time.time() < end:
        pv = mav.recv_match(type="PARAM_VALUE", blocking=True, timeout=end - time.time())
        if pv is not None and pv.param_id.strip("\x00") == name:
            return pv.param_value
    return None


def wait_heartbeat(mav, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        hb = mav.recv_match(type="HEARTBEAT", blocking=True, timeout=5.0)
        if hb is not None:
            return hb
        print("  ... waiting for SITL heartbeat")
    return None


def cur_mode(mav, timeout=5.0):
    hb = mav.recv_match(type="HEARTBEAT", blocking=True, timeout=timeout)
    return hb.custom_mode if hb else None


def enter_mode(mav, mode, timeout=15.0):
    mav.set_mode(mode)
    end = time.time() + timeout
    while time.time() < end:
        hb = mav.recv_match(type="HEARTBEAT", blocking=True, timeout=5.0)
        if hb is not None and hb.custom_mode == mode:
            return True
    return False


def arm(mav, timeout=40.0):
    from pymavlink import mavutil
    armed, last = False, 0.0
    end = time.time() + timeout
    while time.time() < end and not armed:
        if time.time() - last > 3.0:
            mav.mav.command_long_send(mav.target_system, AP,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
            last = time.time()
        m = mav.recv_match(type=["HEARTBEAT", "STATUSTEXT"], blocking=True, timeout=2.0)
        if m is None:
            continue
        if m.get_type() == "STATUSTEXT" and any(k in m.text for k in ("Arm", "PreArm", "EKF", "GPS")):
            print(f"      STATUSTEXT: {m.text}")
        elif m.get_type() == "HEARTBEAT":
            armed = bool(m.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
    return armed


def disarm(mav):
    from pymavlink import mavutil
    mav.mav.command_long_send(mav.target_system, AP,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 0, 0, 0, 0, 0, 0, 0)


def request_streams(mav):
    mav.mav.request_data_stream_send(mav.target_system, AP,
                                     mavutil.mavlink.MAV_DATA_STREAM_ALL, 20, 1)


def measure_yaw(mav, send_fn, climb_s=6.0, yaw_s=5.0, hz=20.0):
    """Run send_fn(phase) at hz; phase 'climb' then 'yaw'. Return mean yawspeed
    (rad/s) measured from ATTITUDE during the yaw phase."""
    def _run(phase, secs):
        request_streams(mav)        # keep EXTRA1 (ATTITUDE) alive across the run
        end, samples = time.time() + secs, []
        while time.time() < end:
            send_fn(phase)
            m = mav.recv_match(type="ATTITUDE", blocking=False)
            if m is not None:
                samples.append(m.yawspeed)
            time.sleep(1.0 / hz)
        return samples
    _run("climb", climb_s)
    s = _run("yaw", yaw_s)
    return (sum(s) / len(s)) if s else None


def main():
    from pymavlink import mavutil
    ap = argparse.ArgumentParser()
    ap.add_argument("--connect", default="tcp:127.0.0.1:5760")
    ap.add_argument("--boot-timeout", type=float, default=180.0)
    args = ap.parse_args()
    chk = Checks()

    # Backend only used for the GUIDED_NOGPS SET_ATTITUDE_TARGET path (production
    # send_intent); everything else drives the link directly.
    backend = ArduPilotBackend(device=args.connect, baud=0, switch_channel=7,
                               track_threshold_us=1300, dive_threshold_us=1700)
    backend.open()
    mav = backend._mav
    print(f"connecting to {args.connect} (boot timeout {args.boot_timeout:.0f}s) ...")
    if not chk.record("SITL heartbeat", wait_heartbeat(mav, args.boot_timeout) is not None):
        return chk.summary()
    mav.target_component = AP
    mav.mav.request_data_stream_send(mav.target_system, AP,
                                     mavutil.mavlink.MAV_DATA_STREAM_ALL, 15, 1)

    # ---- turn GPS OFF entirely, then reboot so it comes up GPS-less ----------
    # Covers both EKF flavours: EKF3 (4.6 default) via EK3_SRC*, EKF2 (4.0.x) via
    # EK2_GPS_TYPE; and both sim-GPS param names (4.6 SIM_GPS1_ENABLE, 4.0.x
    # SIM_GPS_DISABLE). Missing params just return None — harmless. AHRS_EKF_TYPE
    # left at the FC default so we don't point at an EKF with no cores.
    print("\n  disabling GPS (GPS_TYPE=0, sim GPS off, EKF source = no-GPS) ...")
    set_param(mav, "GPS_TYPE", 0)             # no GPS driver
    set_param(mav, "SIM_GPS1_ENABLE", 0)      # 4.6 sim GPS off
    set_param(mav, "SIM_GPS_DISABLE", 1)      # 4.0.x sim GPS off
    set_param(mav, "EK3_SRC1_POSXY", 0)       # EKF3: no GPS horizontal position
    set_param(mav, "EK3_SRC1_VELXY", 0)       # EKF3: no GPS horizontal velocity
    set_param(mav, "EK3_SRC1_POSZ", 1)        # EKF3: height from baro
    set_param(mav, "EK2_GPS_TYPE", 3)         # EKF2 (4.0.x): inhibit GPS use
    set_param(mav, "ARMING_CHECK", 0)         # sim convenience (no GPS/EKF gating)
    print("  rebooting autopilot ...")
    mav.mav.command_long_send(mav.target_system, AP,
        mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN, 0, 1, 0, 0, 0, 0, 0, 0)
    backend.close()
    time.sleep(14.0)   # let arducopter fully restart before reconnecting (avoids
                       # connecting to a half-dead socket -> autoreconnect churn)
    backend = ArduPilotBackend(device=args.connect, baud=0, switch_channel=7,
                               track_threshold_us=1300, dive_threshold_us=1700)
    backend.open()
    mav = backend._mav
    if not chk.record("reconnected after reboot", wait_heartbeat(mav, args.boot_timeout) is not None):
        return chk.summary()
    mav.target_component = AP
    # A fresh TCP session has no stream subscriptions; re-request until ATTITUDE
    # actually flows (EXTRA1) so the steer measurement has data.
    def ensure_attitude(timeout=30.0):
        end = time.time() + timeout
        while time.time() < end:
            request_streams(mav)
            if mav.recv_match(type="ATTITUDE", blocking=True, timeout=2.0) is not None:
                return True
        return False
    if not chk.record("ATTITUDE telemetry streaming after reboot", ensure_attitude()):
        return chk.summary()
    time.sleep(2.0)

    # confirm GPS really is gone
    graw = mav.recv_match(type="GPS_RAW_INT", blocking=True, timeout=5.0)
    fix = graw.fix_type if graw else None
    sats = graw.satellites_visible if graw else None
    chk.record("GPS is disabled (no fix / 0 sats)", (fix in (0, 1, None)) and (sats in (0, 255, None)),
               f"fix_type={fix} sats={sats}")

    # ---- RC override keepalive (for ALT_HOLD / STABILIZE) -------------------
    rc = {"thr": 1500, "yaw": 1500, "on": False}
    stop = threading.Event()

    def rc_loop():
        while not stop.is_set():
            if rc["on"]:
                # ch1 roll, ch2 pitch, ch3 throttle, ch4 yaw; 0 = release ch5..8
                mav.mav.rc_channels_override_send(mav.target_system, AP,
                    1500, 1500, rc["thr"], rc["yaw"], 0, 0, 0, 0)
            stop.wait(0.2)

    threading.Thread(target=rc_loop, daemon=True).start()

    results = {}

    def test_rc_mode(label, mode):
        # pin FLTMODE1..6 so the static sim ch5 can't pull us out of `mode`
        for i in range(1, 7):
            set_param(mav, f"FLTMODE{i}", mode)
        ok_enter = enter_mode(mav, mode)
        chk.record(f"{label}: enter", ok_enter, f"mode={cur_mode(mav)}")
        rc["on"] = True
        ok_arm = arm(mav) if ok_enter else False
        chk.record(f"{label}: arm", ok_arm)
        yaw = None
        if ok_arm:
            def send(phase):
                rc["thr"] = 1700 if phase == "climb" else 1550
                rc["yaw"] = 1500 if phase == "climb" else 1700  # yaw right
            yaw = measure_yaw(mav, lambda phase: send(phase))
        steer = yaw is not None and yaw > 0.05
        chk.record(f"{label}: +yaw cmd -> +yaw rate",
                   steer, f"{math.degrees(yaw):.1f} dps" if yaw is not None else "no data")
        rc["on"] = False
        rc["thr"], rc["yaw"] = 1500, 1500
        disarm(mav)
        results[label] = (ok_enter, ok_arm, steer)
        time.sleep(2.0)

    print("\n  === ALT_HOLD (RC override) ===")
    test_rc_mode("ALT_HOLD", ALT_HOLD)
    print("\n  === STABILIZE (RC override) ===")
    test_rc_mode("STABILIZE", STABILIZE)

    stop.set()
    backend.close()

    print(f"\n{'mode':<14}{'enter':<8}{'arm':<8}{'steer':<8}")
    for label, (e, a, s) in results.items():
        print(f"{label:<14}{'yes' if e else 'NO':<8}{'yes' if a else 'NO':<8}{'yes' if s else 'NO':<8}")
    return chk.summary()


if __name__ == "__main__":
    sys.exit(main())
