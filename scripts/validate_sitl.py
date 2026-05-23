"""ArduPilot SITL validation of the GPS-denied ALT_HOLD RC-override control path.

Proves the production `ArduPilotBackend.send_intent()` (RC_CHANNELS_OVERRIDE AETR
sticks) steers a real ArduCopter in ALT_HOLD in the correct sense, and pins down
the stick signs (deployment-safety.md §4). It drives the *real* send path — the
same call the pipeline makes — and uses the link directly only for setup
(params/mode/arm) and for reading ATTITUDE back.

Run against the radarku/ardupilot-sitl container (TCP 5760, no MAVProxy):
    docker run -d --rm --name pifpv-sitl -p 5760:5760 radarku/ardupilot-sitl:latest
    .venv/bin/python scripts/validate_sitl.py --connect tcp:127.0.0.1:5760

ARMING_CHECK is zeroed (SITL convenience). ALT_HOLD needs only baro + IMU — no
GPS — so this is the right model for a GPS-denied airframe (docs/gps-denied-modes.md).
"""
from __future__ import annotations
import argparse
import math
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from pi_fpv_companion.fc.ardupilot import ArduPilotBackend
from pi_fpv_companion.types import GuidanceIntent

ALT_HOLD = 2  # ArduCopter flight-mode number


class Checks:
    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def record(self, name: str, ok: bool, detail: str = "") -> bool:
        self.results.append((name, ok, detail))
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
        return ok

    def summary(self) -> int:
        passed = sum(1 for _, ok, _ in self.results if ok)
        total = len(self.results)
        print(f"\n{'='*60}\n{passed}/{total} checks passed")
        for name, ok, detail in self.results:
            if not ok:
                print(f"  FAIL: {name} — {detail}")
        return 0 if passed == total else 1


def _recv(mav, types, timeout):
    end = time.time() + timeout
    while time.time() < end:
        m = mav.recv_match(type=types, blocking=True, timeout=end - time.time())
        if m is not None:
            return m
    return None


def _att_sample(mav, secs):
    end = time.time() + secs
    samples = []
    while time.time() < end:
        m = mav.recv_match(type="ATTITUDE", blocking=True, timeout=end - time.time())
        if m is not None:
            samples.append((m.roll, m.pitch, m.yaw, m.rollspeed, m.pitchspeed, m.yawspeed))
    if not samples:
        return None
    n = len(samples)
    avg = [sum(s[i] for s in samples) / n for i in range(6)]
    return {"roll": avg[0], "pitch": avg[1], "yaw": avg[2],
            "rollspeed": avg[3], "pitchspeed": avg[4], "yawspeed": avg[5],
            "last": samples[-1], "n": n}


def _hold_intent(backend, mav, intent, secs, hz=20.0):
    """Stream one intent at `hz` (the pipeline cadence) via the production
    send_intent (RC override), collecting ATTITUDE. Returns attitude stats."""
    end = time.time() + secs
    period = 1.0 / hz
    samples = []
    while time.time() < end:
        backend.send_intent(intent)
        m = mav.recv_match(type="ATTITUDE", blocking=False)
        if m is not None:
            samples.append((m.roll, m.pitch, m.yaw, m.rollspeed, m.pitchspeed, m.yawspeed))
        time.sleep(period)
    if not samples:
        return None
    n = len(samples)
    avg = [sum(s[i] for s in samples) / n for i in range(6)]
    return {"roll": avg[0], "pitch": avg[1], "yaw": avg[2],
            "rollspeed": avg[3], "pitchspeed": avg[4], "yawspeed": avg[5],
            "last": samples[-1], "n": n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--connect", default="tcp:127.0.0.1:5760")
    ap.add_argument("--boot-timeout", type=float, default=180.0)
    args = ap.parse_args()

    chk = Checks()
    print(f"connecting to {args.connect} (boot timeout {args.boot_timeout:.0f}s) ...")

    backend = ArduPilotBackend(device=args.connect, baud=0, switch_channel=7,
                               track_threshold_us=1300, dive_threshold_us=1700)
    backend.open()
    mav = backend._mav
    AP = 1

    hb = None
    deadline = time.time() + args.boot_timeout
    while time.time() < deadline:
        hb = mav.recv_match(type="HEARTBEAT", blocking=True, timeout=5.0)
        if hb is not None:
            break
        print("  ... still waiting for SITL heartbeat")
    if not chk.record("SITL heartbeat received", hb is not None,
                      f"sys={mav.target_system}" if hb else "no heartbeat"):
        return chk.summary()
    mav.target_component = AP
    mav.mav.request_data_stream_send(
        mav.target_system, AP, backend._mavutil.mavlink.MAV_DATA_STREAM_ALL, 15, 1)

    def set_param(name, val, ptype="INT32"):
        mav.mav.param_set_send(mav.target_system, AP, name.encode(), float(val),
                               getattr(backend._mavutil.mavlink, f"MAV_PARAM_TYPE_{ptype}"))
        mav.mav.param_request_read_send(mav.target_system, AP, name.encode(), -1)
        end = time.time() + 8.0
        while time.time() < end:
            pv = mav.recv_match(type="PARAM_VALUE", blocking=True, timeout=end - time.time())
            if pv is not None and pv.param_id.strip("\x00") == name:
                return pv.param_value
        return None

    # Leave AHRS_EKF_TYPE at the FC default (EKF3 on 4.6, EKF2 on 4.0.x) — forcing
    # a specific EKF can brick arming if that EKF has no cores on this build.
    set_param("ARMING_CHECK", 0)
    for i in range(1, 7):
        set_param(f"FLTMODE{i}", ALT_HOLD)   # pin so the static sim ch5 can't move us
    time.sleep(3.0)

    # ---- enter ALT_HOLD --------------------------------------------------
    mav.set_mode(ALT_HOLD)
    in_mode = False
    deadline = time.time() + 15.0
    while time.time() < deadline:
        m = mav.recv_match(type="HEARTBEAT", blocking=True, timeout=5.0)
        if m is not None and m.custom_mode == ALT_HOLD:
            in_mode = True
            break
    chk.record("ArduCopter in ALT_HOLD (baro alt-hold, no GPS needed)", in_mode,
               "entered ALT_HOLD" if in_mode else "mode never became 2")

    # ---- arm (throttle is at radio-low; no override yet) -----------------
    armed = False
    deadline = time.time() + 60.0
    last_arm = 0.0
    while time.time() < deadline and not armed:
        if time.time() - last_arm > 4.0:
            mav.mav.command_long_send(
                mav.target_system, AP,
                backend._mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
            last_arm = time.time()
        m = mav.recv_match(type=["HEARTBEAT", "STATUSTEXT"], blocking=True, timeout=2.0)
        if m is None:
            continue
        if m.get_type() == "STATUSTEXT" and ("Arm" in m.text or "PreArm" in m.text):
            print(f"     STATUSTEXT: {m.text}")
        elif m.get_type() == "HEARTBEAT":
            armed = bool(m.base_mode & backend._mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
    if not chk.record("armed in ALT_HOLD", armed, "motors armed" if armed else "never armed"):
        return chk.summary()

    # ---- climb (throttle stick up) so attitude control is live ----------
    print("\n  climbing (RC override throttle up) ...")
    _hold_intent(backend, mav, GuidanceIntent(0.0, 0.0, 0.0, 0.85, time.monotonic()), 8.0)
    base = _att_sample(mav, 1.0)
    chk.record("RC override stream accepted (airborne, still ALT_HOLD)",
               base is not None, f"{base['n']} ATTITUDE msgs" if base else "no ATTITUDE")

    # thrust 0.5 -> hold altitude while we exercise each axis.
    yr = _hold_intent(backend, mav, GuidanceIntent(0.0, 0.0, 30.0, 0.5, time.monotonic()), 5.0)
    chk.record("+yaw-rate command -> + measured yaw rate",
               yr is not None and yr["yawspeed"] > 0.05,
               f"cmd +30 dps -> {math.degrees(yr['yawspeed']):.1f} dps" if yr else "no data")

    yl = _hold_intent(backend, mav, GuidanceIntent(0.0, 0.0, -30.0, 0.5, time.monotonic()), 5.0)
    chk.record("-yaw-rate command -> - measured yaw rate",
               yl is not None and yl["yawspeed"] < -0.05,
               f"cmd -30 dps -> {math.degrees(yl['yawspeed']):.1f} dps" if yl else "no data")

    # nose-down pitch (pitch_deg<0 = approach); ArduPilot ATTITUDE.pitch +nose-up.
    pd = _hold_intent(backend, mav, GuidanceIntent(0.0, -10.0, 0.0, 0.5, time.monotonic()), 5.0)
    chk.record("nose-down pitch command -> nose-down attitude",
               pd is not None and pd["pitch"] < -0.03,
               f"cmd pitch -10 deg -> {math.degrees(pd['pitch']):.1f} deg" if pd else "no data")

    rl = _hold_intent(backend, mav, GuidanceIntent(12.0, 0.0, 0.0, 0.5, time.monotonic()), 5.0)
    chk.record("+roll command -> +roll attitude",
               rl is not None and rl["roll"] > 0.03,
               f"cmd roll +12 deg -> {math.degrees(rl['roll']):.1f} deg" if rl else "no data")

    hb2 = _recv(mav, "HEARTBEAT", 5.0)
    held = (hb2 is not None and hb2.custom_mode == ALT_HOLD
            and bool(hb2.base_mode & backend._mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED))
    chk.record("held ALT_HOLD + armed throughout", held,
               "ArduPilot accepted the whole override stream" if held else "mode/arm changed")

    # ---- release -> pilot regains control, then land/disarm -------------
    print("\n  releasing to pilot, descending + disarming ...")
    backend.release()
    _hold_intent(backend, mav, GuidanceIntent(0.0, 0.0, 0.0, 0.2, time.monotonic()), 4.0)
    mav.arducopter_disarm()
    backend.close()
    return chk.summary()


if __name__ == "__main__":
    sys.exit(main())
