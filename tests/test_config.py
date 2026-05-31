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
    assert s.dive_forward_deg == 20.0     # committed lean — steeper for a faster dive (target held high)
    assert s.dive_climb_forward_deg == 6.0   # gentle when level/climbing
    assert s.dive_max_pitch_deg == 30.0   # DIVE's own steeper clamp
    assert s.dive_vrate_gain == 18.0      # closed-loop vertical homing (P); higher -> centred aim
    assert s.dive_vrate_damp == 6.0       # derivative damping (anti-wiggle), scaled with the gain
    assert s.dive_lean_tau_s == 1.5       # lean low-pass (anti-nod, steady travel to target)
    assert s.dive_max_descent_mps == 8.0
    assert s.dive_max_climb_mps == 4.0
    assert s.track_vcenter_gain == 0.0   # TRACK is pure range-hold (vertical re-centre off)
    assert s.dive_pitch_fold == 0.6      # partial fold: descends onto a high target without over-descending
    assert s.vfov_deg == 52.3            # IMX500 vertical FoV (pitch -> frame units)
    assert s.dive_terminal_lock_frac == 0.5   # commit ballistic at frame-fill
    assert s.dive_roll_gain == 0.05 and s.dive_roll_damp == 0.45   # gentle heavily-damped bank (stable, off-axis help)
    assert s.roll_compensate is True     # de-roll stays on (no-op at 0 bank), for when roll is enabled


def test_rejects_out_of_range_angle_max(tmp_path):
    # angle_max_deg is auto-written to the FC's ANGLE_MAX — a typo must be caught.
    with pytest.raises(ValueError, match="angle_max_deg"):
        load(_write(tmp_path, "backend: ardupilot, angle_max_deg: 120"))


def test_rejects_select_channel_without_multi_tracker(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("""
camera: {type: synthetic}
detector: {type: none}
tracker: {type: iou}
fc: {backend: ardupilot, select_channel: 9}
""")
    with pytest.raises(ValueError, match="select_channel"):
        load(p)


def test_rejects_zero_frame_size(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("""
video: {width: 0, height: 576}
camera: {type: synthetic}
detector: {type: none}
tracker: {type: iou}
fc: {backend: ardupilot}
""")
    with pytest.raises(ValueError, match="width/height"):
        load(p)


def test_rejects_negative_vrate_gain(tmp_path):
    with pytest.raises(ValueError, match="dive_vrate_gain"):
        load(_write_guidance(tmp_path, "dive_vrate_gain: -1"))


def test_rejects_negative_vertical_clamps(tmp_path):
    with pytest.raises(ValueError, match="dive_max_descent_mps"):
        load(_write_guidance(tmp_path, "dive_max_descent_mps: -2"))


def test_rejects_negative_dive_vrate_damp(tmp_path):
    # Negative derivative damping would AMPLIFY the dive oscillation it exists to remove.
    with pytest.raises(ValueError, match="dive_vrate_damp"):
        load(_write_guidance(tmp_path, "dive_vrate_damp: -2"))


def test_rejects_negative_dive_lean_tau(tmp_path):
    with pytest.raises(ValueError, match="dive_lean_tau_s"):
        load(_write_guidance(tmp_path, "dive_lean_tau_s: -1"))


def test_rejects_negative_closure_i_gain(tmp_path):
    # A negative closure integral inverts the range loop (drives away from hold).
    with pytest.raises(ValueError, match="closure_i_gain"):
        load(_write_guidance(tmp_path, "closure_i_gain: -1"))


def test_imx500_enables_pi_closure():
    cfg = load(Path(__file__).resolve().parent.parent / "config" / "imx500.yaml")
    s = cfg.servo
    assert s.desired_bbox_frac == 0.15    # STANDBY-preview nominal (flight holds engage dist)
    assert s.closure_p_gain == 4.0        # range-linear (inverse-size) gain scale
    assert s.closure_i_gain == 1.0        # PI integral -> holds the engage distance on a mover


def test_dive_defaults_to_vertical_homing_off(tmp_path):
    # A guidance section that doesn't enable the dive leaves vertical homing off.
    cfg = load(_write_guidance(tmp_path, "max_yaw_rate_dps: 60"))
    assert cfg.servo.dive_vrate_gain == 0.0
    assert cfg.servo.dive_forward_deg == 10.0    # dataclass default
