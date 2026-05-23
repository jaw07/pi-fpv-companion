from pathlib import Path

from pi_fpv_companion.config import load


def test_loads_default_yaml():
    cfg = load(Path(__file__).resolve().parent.parent / "config" / "default.yaml")
    assert cfg.video.width == 720
    assert cfg.video.height == 576
    assert cfg.video.tv_mode == "PAL"
    assert cfg.camera.type == "picam"
    assert cfg.detector.type == "nanodet"
    assert cfg.detector.input_size == 256
    assert cfg.detector.detect_period_frames == 7
    assert cfg.tracker.type == "classical"
    assert cfg.tracker.cv2_backend == "mosse"
    assert cfg.fc.backend == "ardupilot"
    assert cfg.fc.baud == 115200
    assert cfg.fc.switch_channel == 7
    # Betaflight mapping should load even when backend is ardupilot — it's just metadata
    assert cfg.fc.betaflight is not None
    assert cfg.fc.betaflight.pitch_us_per_deg == -12.0
    # Servo derives frame size from video section
    assert cfg.servo.frame_width == 720
    assert cfg.servo.frame_height == 576
    # Safety watchdog converts ms -> s
    assert cfg.safety.watchdog_timeout_s == 0.25


def test_loads_mac_dev_yaml():
    cfg = load(Path(__file__).resolve().parent.parent / "config" / "mac-dev.yaml")
    assert cfg.camera.type == "synthetic"
    assert cfg.detector.type == "none"
    assert cfg.tracker.type == "iou"
    assert cfg.fc.uart_device == "udpin:127.0.0.1:14550"


def test_unknown_keys_dont_crash(tmp_path):
    """Future-proofing — adding a new key to default.yaml shouldn't break old loaders."""
    p = tmp_path / "extra.yaml"
    p.write_text("""
video: {width: 1024, height: 768, future_key: ignored}
camera: {type: synthetic}
detector: {type: none}
tracker: {type: iou}
fc: {backend: ardupilot, uart_device: udpin:127.0.0.1:14550, baud: 115200}
guidance: {max_yaw_rate_dps: 90.0}
safety: {watchdog_timeout_ms: 100}
""")
    cfg = load(p)
    assert cfg.video.width == 1024
    assert cfg.servo.max_yaw_rate_dps == 90.0
