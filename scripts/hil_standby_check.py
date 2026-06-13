#!/usr/bin/env python3
"""Hardware-in-the-loop STANDBY safety-contract check (real FC, synthetic camera).

Runs the REAL pipeline + REAL ArduPilotBackend against the REAL flight controller
(props OFF, no camera needed) and verifies — by wrapping the backend's outbound
MAVLink sends at the source — that the STANDBY/disarmed command contract holds on
actual hardware:

  - STANDBY (or FC not in the control_mode's flight mode): NOTHING but the
    zero-override "hand back" burst, then silence.
  - DISARMED: no SET_ATTITUDE_TARGET, no non-zero override, in any state.

A SyntheticCamera drives the pipeline so it ticks exactly as in flight; the real
FC supplies the live switch/armed/mode telemetry that drives the decisions. The
engage switch comes from the real FC's RC_CHANNELS (so with no TX bound it reads
STANDBY — the case we most need to prove). Use --force-mode to also exercise the
disarmed-while-engaged guard.

    python scripts/hil_standby_check.py --seconds 12               # natural STANDBY
    python scripts/hil_standby_check.py --force-mode track --seconds 12   # disarmed+engaged guard

Exit non-zero on any contract violation.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pi_fpv_companion.safety_contract import ContractChecker, ContractConfig
from pi_fpv_companion.types import GuidanceMode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="/dev/serial0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--switch-channel", type=int, default=7)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--force-mode", choices=["standby", "track", "dive"], default=None)
    args = ap.parse_args(argv)

    from pi_fpv_companion.camera.synthetic import SyntheticCamera
    from pi_fpv_companion.fc.ardupilot import ArduPilotBackend, ArduCopterRcMapping
    from pi_fpv_companion.guidance.rate_control import RateConfig
    from pi_fpv_companion.guidance.safety import SafetyConfig
    from pi_fpv_companion.guidance.visual_servo import ServoConfig
    from pi_fpv_companion.pipeline import Pipeline

    fc = ArduPilotBackend(device=args.device, baud=args.baud,
                          switch_channel=args.switch_channel,
                          track_threshold_us=1300, dive_threshold_us=1700,
                          mapping=ArduCopterRcMapping(control_mode="guided_nogps"))
    fc.open()
    fc.wait_ready(timeout=10)

    checker = ContractChecker(cfg=ContractConfig(switch_channel=args.switch_channel))

    # Wrap the backend's outbound MAVLink sends so every transmission is observed
    # with the live switch/armed state — this is the wire-level proof, at the source.
    real = fc._mav.mav

    def _state():
        sw = fc.read_switch()
        checker.on_rc_channels(time.time(), sw.pwm_us)
        checker.on_heartbeat(time.time(), fc.is_armed())

    orig_override = real.rc_channels_override_send
    orig_attitude = real.set_attitude_target_send
    orig_cmd = real.command_long_send

    def wrap_override(*a, **k):
        _state(); checker.on_rc_override(time.time(), list(a[2:10]))
        return orig_override(*a, **k)

    def wrap_attitude(*a, **k):
        _state(); checker.on_attitude_target(time.time())
        return orig_attitude(*a, **k)

    def wrap_cmd(*a, **k):
        from pymavlink import mavutil
        if len(a) > 2 and a[2] == mavutil.mavlink.MAV_CMD_DO_SET_MODE:
            _state(); checker.on_set_mode(time.time(), int(a[5]))
        return orig_cmd(*a, **k)

    real.rc_channels_override_send = wrap_override
    real.set_attitude_target_send = wrap_attitude
    real.command_long_send = wrap_cmd

    cam = SyntheticCamera(width=720, height=576)
    force = GuidanceMode[args.force_mode.upper()] if args.force_mode else None
    pipe = Pipeline(cam, _build_tracker(), _servo(), _safety(), fc,
                    force_mode=force, rate_cfg=RateConfig(720, 576))

    print(f"running {args.seconds:.0f}s against the real FC "
          f"(force_mode={args.force_mode or 'none/real-switch'}, PROPS OFF) ...")
    end = time.time() + args.seconds
    frames = cam.frames()
    while time.time() < end:
        pipe.tick(next(frames))
        time.sleep(0.02)
    fc.close()

    print(checker.report())
    return 0 if checker.passed else 1


def _build_tracker():
    from pi_fpv_companion.track.multi_target import MultiObjectTracker
    return MultiObjectTracker(iou_threshold=0.3, max_lost_frames=8)


def _servo():
    from pi_fpv_companion.guidance.visual_servo import ServoConfig
    return ServoConfig(frame_width=720, frame_height=576, max_yaw_rate_dps=60.0,
                       max_pitch_deg=15.0, pixel_deadzone_px=10.0, yaw_p_gain=0.3,
                       yaw_ff_gain=0.0, desired_bbox_frac=0.30, closure_p_gain=50.0)


def _safety():
    from pi_fpv_companion.guidance.safety import SafetyConfig
    return SafetyConfig(watchdog_timeout_s=1.0, require_armed=True)


if __name__ == "__main__":
    raise SystemExit(main())
