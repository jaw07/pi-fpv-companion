from pathlib import Path

import pytest

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


def _write(tmp_path, fc_line: str) -> Path:
    p = tmp_path / "c.yaml"
    p.write_text(f"""
camera: {{type: synthetic}}
detector: {{type: none}}
tracker: {{type: iou}}
fc: {{{fc_line}}}
""")
    return p


def test_rejects_unknown_control_mode(tmp_path):
    p = _write(tmp_path, "backend: ardupilot, control_mode: stabalize")
    with pytest.raises(ValueError, match="control_mode"):
        load(p)


def test_rejects_dive_threshold_below_track(tmp_path):
    p = _write(tmp_path, "backend: ardupilot, track_threshold_us: 1700, dive_threshold_us: 1300")
    with pytest.raises(ValueError, match="dive_threshold_us"):
        load(p)


def test_equal_thresholds_allowed(tmp_path):
    # dive == track is fine (TRACK band is just empty); only dive < track is wrong.
    p = _write(tmp_path, "backend: ardupilot, track_threshold_us: 1500, dive_threshold_us: 1500")
    assert load(p).fc.dive_threshold_us == 1500


def _write_guidance(tmp_path, guidance_line: str) -> Path:
    p = tmp_path / "g.yaml"
    p.write_text(f"""
camera: {{type: synthetic}}
detector: {{type: none}}
tracker: {{type: iou}}
fc: {{backend: ardupilot}}
guidance: {{{guidance_line}}}
""")
    return p


def test_imx500_enables_closed_loop_dive():
    cfg = load(Path(__file__).resolve().parent.parent / "config" / "imx500.yaml")
    s = cfg.servo
    assert s.dive_forward_deg == 25.0     # steep lean at full descent (fast ground attack)
    assert s.dive_climb_forward_deg == 6.0   # gentle when level/climbing
    assert s.dive_max_pitch_deg == 30.0   # DIVE's own steeper clamp
    assert s.dive_vrate_gain == 17.0      # closed-loop vertical homing enabled
    assert s.dive_max_descent_mps == 8.0
    assert s.dive_max_climb_mps == 4.0


def test_rejects_negative_vrate_gain(tmp_path):
    with pytest.raises(ValueError, match="dive_vrate_gain"):
        load(_write_guidance(tmp_path, "dive_vrate_gain: -1"))


def test_rejects_negative_vertical_clamps(tmp_path):
    with pytest.raises(ValueError, match="dive_max_descent_mps"):
        load(_write_guidance(tmp_path, "dive_max_descent_mps: -2"))


def test_dive_defaults_to_vertical_homing_off(tmp_path):
    # A guidance section that doesn't enable the dive leaves vertical homing off.
    cfg = load(_write_guidance(tmp_path, "max_yaw_rate_dps: 60"))
    assert cfg.servo.dive_vrate_gain == 0.0
    assert cfg.servo.dive_forward_deg == 10.0    # dataclass default
