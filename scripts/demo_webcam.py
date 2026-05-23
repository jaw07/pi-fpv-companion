"""Live-webcam demo: real camera + real (Haar) detector + the full pipeline.

Point your face at the laptop camera. The pipeline should lock onto it and
the guidance intent will drive the FakeArduCopter. Yaw left/right and the
forward/climb numbers should react to your head position.

Switches:
  --no-gui         run headless (perf only, no preview window)
  --duration N     seconds to run (default 30)
  --downscale F    detect at fraction of frame size (default 0.5 — faster, Pi-realistic)
  --device N       webcam index (default 0)
"""
from __future__ import annotations
import argparse
import socket
import sys
import threading
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root))

from pi_fpv_companion.camera.webcam import WebcamCamera
from pi_fpv_companion.detect.haar import HaarFaceDetector
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
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--downscale", type=float, default=0.5)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    port = _free_udp_port()
    backend = ArduPilotBackend(f"udpin:127.0.0.1:{port}", 0, 7, 1300, 1700)
    backend.open()
    fake = FakeArduCopter(target_port=port)
    fake.start()
    backend.wait_ready(timeout=5.0)
    fake.armed = True
    fake.rc_channels[6] = 1800

    detector = HaarFaceDetector(downscale=args.downscale, min_size_px=60)
    camera = WebcamCamera(
        device=args.device, width=args.width, height=args.height,
        fps=30,
    )
    tracker = IouAssociator(iou_threshold=0.2, max_lost_frames=30)
    servo = ServoConfig(
        frame_width=args.width, frame_height=args.height,
        max_yaw_rate_dps=60.0, max_pitch_deg=15.0,
        pixel_deadzone_px=30.0, yaw_p_gain=0.2, yaw_ff_gain=0.05, desired_bbox_frac=0.30, closure_p_gain=50.0,
    )
    safety = SafetyConfig(watchdog_timeout_s=0.5, require_armed=True)
    perf = PerfMonitor(PiBudget(max_tick_ms=33.0, max_rss_mb=200.0, pi_scale_factor=6.0))
    viewer = None if args.no_gui else LiveViewer(window_name="pi-fpv-companion (webcam)")

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
                f"q={q}  tick={elapsed:5.2f}ms  muted={str(gated.muted):5}"
            )

    on_status._t0 = 0.0

    pipeline = Pipeline(camera, tracker, servo, safety, backend,
                        detector=detector, detect_period_frames=1, on_status=on_status)
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
