"""MjpegStreamSink (bench web stream) + NullFC + bench config wiring tests."""
from __future__ import annotations
import urllib.request

import numpy as np

from pi_fpv_companion.camera.base import FrameBundle
from pi_fpv_companion.guidance.safety import GateResult
from pi_fpv_companion.types import GuidanceMode, SwitchState, ZERO_INTENT
from pi_fpv_companion.video.mjpeg_stream import MjpegStreamSink


def _frame():
    return FrameBundle(image=np.full((576, 720, 3), 90, dtype=np.uint8),
                       width=720, height=576, timestamp=0.0, detections=[])


def _show(sink):
    switch = SwitchState(active=False, pwm_us=0, timestamp=0.0, mode=GuidanceMode.STANDBY)
    sink.show(None, ZERO_INTENT, GateResult(ZERO_INTENT, True, "standby"),
              switch, False, _frame())


def test_snapshot_serves_latest_composited_jpeg():
    sink = MjpegStreamSink(port=0, quality=80, max_fps=1000.0)   # port 0 = OS-assigned
    try:
        _show(sink)
        with urllib.request.urlopen(f"http://127.0.0.1:{sink.port}/snapshot.jpg",
                                    timeout=5) as r:
            body = r.read()
        assert r.status == 200
        assert body[:2] == b"\xff\xd8", "JPEG magic"           # SOI marker
        with urllib.request.urlopen(f"http://127.0.0.1:{sink.port}/", timeout=5) as r:
            assert b"/stream" in r.read()
    finally:
        sink.close()


def test_show_is_rate_limited():
    sink = MjpegStreamSink(port=0, max_fps=1.0)
    try:
        _show(sink); _show(sink); _show(sink)                  # back-to-back
        assert sink._seq == 1, "only the first frame within the period publishes"
    finally:
        sink.close()


def test_null_fc_pipeline_runs_camera_only():
    # Full pipeline with NO FC: permanent STANDBY, disarmed, nothing transmitted —
    # but perception + HUD preview still run (the bench-rig configuration).
    from pi_fpv_companion.camera.synthetic import SyntheticCamera
    from pi_fpv_companion.fc.null import NullFC
    from pi_fpv_companion.guidance.safety import SafetyConfig
    from pi_fpv_companion.guidance.visual_servo import ServoConfig
    from pi_fpv_companion.pipeline import Pipeline
    from pi_fpv_companion.track.iou_associator import IouAssociator

    cam = SyntheticCamera(width=720, height=576)
    servo = ServoConfig(frame_width=720, frame_height=576, max_yaw_rate_dps=60.0,
                        max_pitch_deg=15.0, pixel_deadzone_px=10.0, yaw_p_gain=0.3,
                        yaw_ff_gain=0.0, desired_bbox_frac=0.30, closure_p_gain=50.0)
    seen = []
    pipe = Pipeline(cam, IouAssociator(iou_threshold=0.2, max_lost_frames=10),
                    servo, SafetyConfig(watchdog_timeout_s=1.0, require_armed=True),
                    NullFC(), on_status=lambda *a: seen.append(a))
    for i in range(5):
        gated = pipe.tick(cam.render_at(i * 0.05))
    assert gated.muted and gated.reason == "standby"
    assert len(seen) == 5                                      # HUD callback ran every tick
    target, intent = seen[-1][0], seen[-1][1]
    assert target is not None, "perception still locks a target with no FC"


def test_bench_stream_config_loads():
    from pi_fpv_companion.config import load
    cfg = load("config/bench-stream.yaml")
    assert cfg.fc.backend == "none"
    assert cfg.video.web_stream_port == 8080
    assert cfg.recorder.enabled is False
    from pi_fpv_companion.main import _build_fc
    from pi_fpv_companion.fc.null import NullFC
    assert isinstance(_build_fc(cfg), NullFC)
