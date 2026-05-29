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


def test_imx500_enables_tuned_agnostic_dive():
    cfg = load(Path(__file__).resolve().parent.parent / "config" / "imx500.yaml")
    s = cfg.servo
    assert s.dive_vertical_bias_frac == 0.50
    assert s.dive_los_band_deg == 30.0    # geometry-match descent to depression
    assert s.dive_pitch_up_max_deg == 0.0   # commit: never pitch nose-up
    assert s.camera_vfov_deg == 52.3      # real IMX500 vertical FoV (product brief)


def test_rejects_out_of_range_vertical_bias(tmp_path):
    with pytest.raises(ValueError, match="dive_vertical_bias_frac"):
        load(_write_guidance(tmp_path, "dive_vertical_bias_frac: 1.5"))


def test_rejects_nonpositive_los_band(tmp_path):
    with pytest.raises(ValueError, match="dive_los_band_deg"):
        load(_write_guidance(tmp_path, "dive_los_band_deg: 0"))


def test_rejects_implausible_vfov(tmp_path):
    with pytest.raises(ValueError, match="camera_vfov_deg"):
        load(_write_guidance(tmp_path, "camera_vfov_deg: 0"))


def test_rejects_dive_descent_swallowed_by_hover_band(tmp_path):
    # dive_descent must exceed the adaptive-hover hold band, or the hold loop
    # cancels the descent and the aircraft never dives.
    p = tmp_path / "c.yaml"
    p.write_text("""
camera: {type: synthetic}
detector: {type: none}
tracker: {type: iou}
fc: {backend: ardupilot, control_mode: stabilize, stab_hover_learn_band: 0.20}
guidance: {dive_descent: 0.12}
""")
    with pytest.raises(ValueError, match="dive_descent"):
        load(p)


def test_vfov_defaults_to_imx500_when_absent(tmp_path):
    # A guidance section that doesn't mention the camera still gets the IMX500 VFoV.
    cfg = load(_write_guidance(tmp_path, "max_yaw_rate_dps: 60"))
    assert cfg.servo.camera_vfov_deg == 52.3
    assert cfg.servo.dive_pitch_up_max_deg is None   # legacy (no cap) by default
