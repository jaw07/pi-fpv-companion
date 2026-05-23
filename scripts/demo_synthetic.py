"""End-to-end software-only demo with live viewer + Pi-budget profiler.

  SyntheticCamera (IMX500-style) -> IouAssociator -> visual servo
                                 -> safety gate -> ArduPilotBackend
                                                       |
                                                       v UDP loopback
                                                  FakeArduCopter

Switches:
  --no-gui     run headless, console only (CI / SSH)
  --duration N seconds to run (default 8)
  --target-fps N synthetic camera fps (default 30)
"""
from __future__ import annotations
import argparse
import socket
import sys
import threading
import time
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
from pi_fpv_companion.video.viewer import LiveViewer
from tests.fakes.fake_ardupilot import FakeArduCopter


def _free_udp_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--target-fps", type=int, default=30)
    args = ap.parse_args()

    port = _free_udp_port()
    backend = ArduPilotBackend(f"udpin:127.0.0.1:{port}", 0, 7, 1300, 1700)
    backend.open()
    fake = FakeArduCopter(target_port=port)
    fake.start()
    backend.wait_ready(timeout=5.0)
    fake.armed = True
    fake.rc_channels[6] = 1800

    camera = SyntheticCamera(width=720, height=576, fps=args.target_fps)
    tracker = IouAssociator(iou_threshold=0.2, max_lost_frames=15)
    servo = ServoConfig(
        frame_width=720, frame_height=576,
        max_yaw_rate_dps=60.0, max_pitch_deg=15.0,
        pixel_deadzone_px=20.0, yaw_p_gain=0.2, yaw_ff_gain=0.05, desired_bbox_frac=0.30, closure_p_gain=50.0,
    )
    safety = SafetyConfig(watchdog_timeout_s=0.5, require_armed=True)
    perf = PerfMonitor(PiBudget(max_tick_ms=33.0, max_rss_mb=200.0, pi_scale_factor=6.0))
    viewer = None if args.no_gui else LiveViewer(window_name="pi-fpv-companion (synthetic)")

    last_print = 0.0

    def on_status(target, intent, gated, switch, armed, frame):
        nonlocal last_print
        elapsed = perf.tick_end(on_status._t0)
        if viewer is not None:
            viewer.show(target, intent, gated, switch, armed, frame)
        if frame.timestamp - last_print >= 0.5:
            last_print = frame.timestamp
            sent = gated.intent
            tpos = (f"({int(target.detection.x):>3},{int(target.detection.y):>3})"
                    if target is not None else "  none ")
            q = f"{target.quality:.2f}" if target is not None else "----"
            print(
                f"t={frame.timestamp:8.2f}  target={tpos}  "
                f"sent=(yaw={sent.yaw_rate_dps:+6.1f}dps  pitch={sent.pitch_deg:+5.1f}deg)  "
                f"q={q}  tick={elapsed:5.2f}ms  "
                f"muted={str(gated.muted):5}"
            )

    on_status._t0 = 0.0

    pipeline = Pipeline(camera, tracker, servo, safety, backend, on_status=on_status)

    # Wrap tick() so we time the whole iteration, not just the callback
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
        fake.stop()
        if viewer is not None:
            viewer.close()

    print()
    print(perf.report())
    print()
    print(f"FakeArduCopter received {len(fake.captured_overrides)} RC overrides")
    return 0


if __name__ == "__main__":
    sys.exit(main())
