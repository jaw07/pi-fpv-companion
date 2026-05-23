"""SITL-backed interactive demo. Same logic as demo_synthetic.py, but the FC
endpoint is real ArduPilot SITL instead of a loopback fake. Runs with
`force_mode=TRACK` so it engages without an RC switch.

The backend injects AETR sticks via RC_CHANNELS_OVERRIDE into ALT_HOLD (the
GPS-denied path — docs/gps-denied-modes.md). Put SITL in ALT_HOLD and arm it so
there's something to steer.

For an automated PASS/FAIL validation use `scripts/validate_sitl.py` (control
sense, 9/9) or `scripts/fly_sitl.py` (full closed loop). This script is the
interactive "watch it fly" companion.

Prereq: SITL running and reachable; see docs/sitl.md.
"""
from __future__ import annotations
import argparse
import sys
import threading
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root))

from pi_fpv_companion.camera.synthetic import SyntheticCamera
from pi_fpv_companion.fc.ardupilot import ArduPilotBackend
from pi_fpv_companion.guidance.safety import SafetyConfig
from pi_fpv_companion.guidance.visual_servo import ServoConfig
from pi_fpv_companion.perf import PerfMonitor, PiBudget
from pi_fpv_companion.pipeline import Pipeline
from pi_fpv_companion.track.iou_associator import IouAssociator
from pi_fpv_companion.types import GuidanceMode
from pi_fpv_companion.video.viewer import LiveViewer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--connect", default="udpin:127.0.0.1:14550",
                    help="MAVLink endpoint (default udpin:127.0.0.1:14550)")
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--duration", type=float, default=30.0)
    args = ap.parse_args()

    backend = ArduPilotBackend(
        device=args.connect, baud=0,
        switch_channel=7, track_threshold_us=1300, dive_threshold_us=1700,
    )
    print(f"connecting to {args.connect} ...")
    backend.open()
    backend.wait_ready(timeout=15.0)
    print("heartbeat received")

    camera = SyntheticCamera(width=720, height=576, fps=20)
    tracker = IouAssociator(iou_threshold=0.2, max_lost_frames=20)
    servo = ServoConfig(
        frame_width=720, frame_height=576,
        max_yaw_rate_dps=45.0, max_pitch_deg=12.0,
        pixel_deadzone_px=30.0, yaw_p_gain=0.15, yaw_ff_gain=0.04, desired_bbox_frac=0.30, closure_p_gain=50.0,
    )
    safety = SafetyConfig(watchdog_timeout_s=0.5, require_armed=True)
    perf = PerfMonitor(PiBudget(max_tick_ms=50.0, max_rss_mb=200.0, pi_scale_factor=6.0))
    viewer = None if args.no_gui else LiveViewer(window_name="pi-fpv-companion (SITL)")

    def on_status(target, intent, gated, switch, armed, frame):
        perf.tick_end(on_status._t0)
        if viewer is not None:
            viewer.show(target, intent, gated, switch, armed, frame)

    on_status._t0 = 0.0

    pipeline = Pipeline(camera, tracker, servo, safety, backend, on_status=on_status,
                        force_mode=GuidanceMode.TRACK)
    orig_tick = pipeline.tick
    def timed_tick(bundle):
        on_status._t0 = perf.tick_start()
        return orig_tick(bundle)
    pipeline.tick = timed_tick

    threading.Timer(args.duration, pipeline.stop).start()

    try:
        pipeline.run()
    finally:
        backend.close()
        if viewer is not None:
            viewer.close()

    print()
    print(perf.report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
