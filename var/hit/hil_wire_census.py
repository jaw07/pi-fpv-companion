"""Census EVERY MAVLink send the companion makes to a REAL FC. Read-only w.r.t. arming.

hil_standby_check.py reported PASS with all-zero counts, which proves nothing. This
counts sends at the source, unconditionally, so "nothing was transmitted" and "the
instrument was broken" cannot be confused.

SAFETY: never arms. Aborts immediately if the FC reports ARMED, and (unless
--allow-mode-cmds) monkey-patches the mode-command path to a no-op so no DO_SET_MODE
can reach a bench FC with props on.

    .venv/bin/python var/hit/hil_wire_census.py --device /dev/cu.usbmodem1101 --seconds 20
"""
from __future__ import annotations
import argparse, sys, threading, time
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pymavlink import mavutil


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="/dev/cu.usbmodem1101")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--force-mode", choices=["standby", "track", "dive"], default=None)
    ap.add_argument("--auto-guided", action="store_true")
    ap.add_argument("--allow-mode-cmds", action="store_true",
                    help="permit DO_SET_MODE to reach the FC (default: blocked)")
    args = ap.parse_args()

    from pi_fpv_companion.camera.synthetic import SyntheticCamera
    from pi_fpv_companion.fc.ardupilot import ArduPilotBackend, ArduCopterRcMapping
    from pi_fpv_companion.guidance.rate_control import RateConfig
    from pi_fpv_companion.guidance.safety import SafetyConfig
    from pi_fpv_companion.guidance.visual_servo import ServoConfig
    from pi_fpv_companion.pipeline import Pipeline
    from pi_fpv_companion.track.multi_target import MultiObjectTracker
    from pi_fpv_companion.types import GuidanceMode

    fc = ArduPilotBackend(device=args.device, baud=args.baud, switch_channel=7,
                          track_threshold_us=1300, dive_threshold_us=1700,
                          auto_guided=args.auto_guided,
                          mapping=ArduCopterRcMapping(control_mode="guided_nogps"))
    fc.open()
    fc.wait_ready(timeout=15)

    if fc.is_armed():
        print("ABORT: FC reports ARMED. Refusing to run against an armed FC.")
        fc.close(); return 2
    start_mode = fc._current_mode
    print(f"FC disarmed OK. mode={start_mode}  auto_guided={args.auto_guided}")

    blocked = Counter()
    if not args.allow_mode_cmds:
        def _blocked_send_mode(mode, _f=fc):
            blocked["do_set_mode_BLOCKED"] += 1
        fc._send_mode = _blocked_send_mode
        print("SAFETY: DO_SET_MODE is BLOCKED (props on the bench FC).")

    census, log = Counter(), []
    real = fc._mav.mav
    lk = threading.Lock()

    def wrap(orig, name_fn):
        def inner(*a, **k):
            n = name_fn(a)
            sw = fc._last_switch
            with lk:
                census[n] += 1
                if n in ("rc_channels_override_nonzero", "set_attitude_target", "do_set_mode"):
                    log.append((n, sw.mode.name if sw else "UNKNOWN", bool(fc._armed), a[2:10]))
            return orig(*a, **k)
        return inner

    def cmd_name(a):
        if len(a) > 2 and a[2] == mavutil.mavlink.MAV_CMD_DO_SET_MODE: return "do_set_mode"
        if len(a) > 2 and a[2] == mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL: return "set_message_interval"
        return "command_long_other"

    real.rc_channels_override_send = wrap(real.rc_channels_override_send,
        lambda a: "rc_channels_override_nonzero" if any(a[2:10]) else "rc_channels_override_zero")
    real.set_attitude_target_send = wrap(real.set_attitude_target_send, lambda a: "set_attitude_target")
    real.command_long_send = wrap(real.command_long_send, cmd_name)
    real.heartbeat_send = wrap(real.heartbeat_send, lambda a: "gcs_heartbeat")
    real.request_data_stream_send = wrap(real.request_data_stream_send, lambda a: "request_data_stream")

    cam = SyntheticCamera(width=720, height=576, fps=22)
    servo = ServoConfig(frame_width=720, frame_height=576, max_yaw_rate_dps=60.0,
                        max_pitch_deg=15.0, pixel_deadzone_px=10.0, yaw_p_gain=0.3,
                        yaw_ff_gain=0.0, desired_bbox_frac=0.30, closure_p_gain=50.0)
    force = GuidanceMode[args.force_mode.upper()] if args.force_mode else None
    pipe = Pipeline(cam, MultiObjectTracker(iou_threshold=0.3, max_lost_frames=8),
                    servo, SafetyConfig(watchdog_timeout_s=1.0, require_armed=True), fc,
                    force_mode=force, rate_cfg=RateConfig(720, 576))

    print(f"running {args.seconds:.0f}s (force_mode={args.force_mode or 'real switch'}) ...")
    end = time.time() + args.seconds
    frames = cam.frames()
    ticks = 0
    while time.time() < end:
        pipe.tick(next(frames)); ticks += 1
        time.sleep(0.02)
    armed_end = fc.is_armed()
    sw = fc.read_switch()

    # RESTORE the FC's starting flight mode. With --force-mode there is never a
    # disengage edge, so set_engaged(False) never fires and the FC would be LEFT in
    # GUIDED_NOGPS — an orphaned engage on a bench FC with props on.
    #
    # Do NOT use the backend's _command_mode/_service_mode retry for this: it FAILED to
    # bring an FC back out of GUIDED_NOGPS on 2026-08-18 and stranded the board. Drive
    # it directly and verify from a FRESH heartbeat, alternating the legacy SET_MODE and
    # COMMAND_LONG encodings (some builds honour only one).
    if args.allow_mode_cmds and start_mode is not None:
        fc._target_mode = None                    # cancel any in-flight backend retry
        mav = fc._mav
        for i in range(20):
            cur = None
            for _ in range(6):
                h = mav.recv_match(type="HEARTBEAT", blocking=True, timeout=0.5)
                if h is not None:
                    cur = h.custom_mode
            if cur == start_mode:
                print(f"\nFC mode restored to {start_mode} (confirmed)")
                break
            if i == 0:
                print(f"\nrestoring FC mode {cur} -> {start_mode} ...")
            if i % 2 == 0:
                mav.mav.set_mode_send(mav.target_system,
                                      mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                                      start_mode)
            else:
                mav.mav.command_long_send(mav.target_system, mav.target_component,
                                          mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
                                          mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                                          start_mode, 0, 0, 0, 0, 0)
        else:
            print(f"\n*** FAILED to restore mode {start_mode} — SET IT MANUALLY IN YOUR GCS ***")
        h = mav.recv_match(type="HEARTBEAT", blocking=True, timeout=3)
        fc._current_mode = h.custom_mode if h else fc._current_mode
    final_mode = fc._current_mode
    fc.close()

    print(f"\n=== wire census ({ticks} ticks) ===")
    if not census:
        print("  (NOTHING transmitted)")
    for k in sorted(census):
        mark = "  <-- TOUCHES FLIGHT CONTROL" if k in ("rc_channels_override_nonzero","set_attitude_target","do_set_mode") else ""
        print(f"  {k:<32} {census[k]:5d}{mark}")
    for k in sorted(blocked):
        print(f"  {k:<32} {blocked[k]:5d}  (suppressed by --safety, would have been sent)")
    for n, mode, armed, payload in log[:12]:
        print(f"    ! {n} switch={mode} armed={armed} chans={payload}")
    print(f"\nswitch read: mode={sw.mode.name} pwm={sw.pwm_us}   armed at end: {armed_end}"
          f"   FC mode at end: {final_mode} (started {start_mode})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
