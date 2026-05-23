"""FileCamera + ColorBlobDetector demo (PiCam path).

Generates a test video from SyntheticCamera frames (so no external assets), then
plays it back through FileCamera with the ColorBlobDetector finding the red
target. Mirrors the Pi runtime path: camera produces frames, detector adds boxes,
tracker locks, servo + safety drive a FakeArduCopter.

Real CV operations happen here (HSV mask + contour finding on actual pixels),
so this exercises the Pi budget in a way the SyntheticCamera path can't.
"""
from __future__ import annotations
import argparse
import socket
import sys
import threading
import time
from pathlib import Path

import cv2

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root))

from pi_fpv_companion.camera.file_camera import FileCamera
from pi_fpv_companion.camera.synthetic import SyntheticCamera
from pi_fpv_companion.detect.color import ColorBlobDetector
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


def _generate_test_video(path: Path, seconds: float = 8.0, fps: int = 30,
                         width: int = 720, height: int = 576) -> Path:
    if path.exists():
        return path
    print(f"generating {seconds:.1f}s test video at {path} ...")
    cam = SyntheticCamera(width=width, height=height, fps=fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    try:
        n_frames = int(seconds * fps)
        for i in range(n_frames):
            t = i / fps
            writer.write(cam.render_at(t).image)
    finally:
        writer.release()
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--video", type=Path, default=Path("/tmp/pi-fpv-test.mp4"))
    ap.add_argument("--detector", choices=("color", "haar"), default="color")
    ap.add_argument("--downscale", type=float, default=1.0,
                    help="haar detector only — detect at fraction of frame size")
    args = ap.parse_args()

    _generate_test_video(args.video, seconds=10.0)

    port = _free_udp_port()
    backend = ArduPilotBackend(f"udpin:127.0.0.1:{port}", 0, 7, 1300, 1700)
    backend.open()
    fake = FakeArduCopter(target_port=port)
    fake.start()
    backend.wait_ready(timeout=5.0)
    fake.armed = True
    fake.rc_channels[6] = 1800

    if args.detector == "color":
        detector = ColorBlobDetector(min_area_px=400)
    else:
        detector = HaarFaceDetector(downscale=args.downscale, min_size_px=60)
    print(f"detector: {args.detector}")
    camera = FileCamera(path=str(args.video), fps_override=30, loop=True)
    tracker = IouAssociator(iou_threshold=0.2, max_lost_frames=15)
    servo = ServoConfig(
        frame_width=720, frame_height=576,
        max_yaw_rate_dps=60.0, max_pitch_deg=15.0,
        pixel_deadzone_px=20.0, yaw_p_gain=0.2, yaw_ff_gain=0.05, desired_bbox_frac=0.30, closure_p_gain=50.0,
    )
    safety = SafetyConfig(watchdog_timeout_s=0.5, require_armed=True)
    perf = PerfMonitor(PiBudget(max_tick_ms=33.0, max_rss_mb=200.0, pi_scale_factor=6.0))
    viewer = None if args.no_gui else LiveViewer(window_name="pi-fpv-companion (file)")

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
