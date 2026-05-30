"""YAML config loader.

Parses `config/*.yaml` into typed config objects that the rest of the project
already uses (`ServoConfig`, `SafetyConfig`, `BetaflightMapping`, `NanoDetConfig`).
Also selects which Camera / Detector / Tracker / FC backend implementations to
construct — actual construction lives in `main.py` since it needs runtime
dependencies (a model file, a serial device, etc.).

Backward-compatible with the existing `config/default.yaml` shape.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from pi_fpv_companion.fc.betaflight import BetaflightMapping
from pi_fpv_companion.guidance.safety import SafetyConfig
from pi_fpv_companion.guidance.visual_servo import ServoConfig


@dataclass
class VideoSection:
    tv_mode: str = "PAL"
    width: int = 720
    height: int = 576
    framebuffer: str = "/dev/fb0"


@dataclass
class CameraSection:
    type: str = "picam"                  # picam | imx500 | webcam | file | synthetic
    framerate: int = 30
    imx500_model: str = ""
    file_path: str = ""
    webcam_device: int = 0
    # PiCam tuning (FPV defaults: short exposure = low motion blur on a moving drone)
    exposure_mode: str = "short"         # short | normal | long
    noise_reduction: str = "fast"        # off | fast | high_quality
    hflip: bool = False
    vflip: bool = False


@dataclass
class DetectorSection:
    type: str = "nanodet"                # nanodet | yolov8 | color | haar | none
    model_dir: str = ""
    input_size: int = 320
    conf_threshold: float = 0.35
    nms_threshold: float = 0.45
    classes_of_interest: List[str] = field(default_factory=list)
    detect_period_frames: int = 10


@dataclass
class TrackerSection:
    type: str = "classical"              # classical (cv2 KCF/MOSSE/CSRT) | iou (imx500 path)
    cv2_backend: str = "mosse"           # mosse | kcf | csrt | medianflow (classical only)
    reacquire_after_lost_frames: int = 30
    iou_threshold: float = 0.3


@dataclass
class FcSection:
    backend: str = "ardupilot"           # ardupilot | betaflight
    uart_device: str = "/dev/ttyAMA0"
    baud: int = 115200
    switch_channel: int = 7
    # Momentary RC channel that cycles the locked target among detections (multi_iou
    # tracker). A rising edge past 1700 µs = "next target". 0 = disabled.
    select_channel: int = 0
    switch_threshold_us: int = 1700      # betaflight 2-state engage threshold
    # ArduPilot 3-position mode switch on switch_channel:
    #   pwm >= dive_threshold_us  -> DIVE
    #   pwm >= track_threshold_us -> TRACK
    #   else                      -> STANDBY
    track_threshold_us: int = 1300
    dive_threshold_us: int = 1700
    # ArduPilot ALT_HOLD RC-override mapping (GPS-denied path): while engaged the
    # companion injects AETR sticks via RC_CHANNELS_OVERRIDE into a self-levelling
    # pilot mode (ALT_HOLD); STANDBY releases them. angle_max_deg should match the
    # FC's ANGLE_MAX; signs are TX/RCMAP dependent — bench/SITL validate (audit §4).
    # control_mode MUST match the FC's flight mode: "stabilize" (DEFAULT; direct
    # throttle, 0.5=hover; no altitude hold -> a true steep dive ~16 m/s, companion
    # owns altitude) or "althold" (throttle = climb rate, 0.5=hold via baro — gentle,
    # descent capped at PILOT_SPEED_DN ~1-5 m/s). See docs/gps-denied-modes.md.
    control_mode: str = "stabilize"
    stab_hover_throttle_us: int = 1450   # stabilize: starting hover guess (learner refines it)
    # Adaptive hover (stabilize): companion vertical-velocity hold — trims hover
    # throttle from measured climb rate so it levels out without manual tuning.
    stab_hover_learn: bool = True
    stab_hover_learn_kp: float = 50.0    # PWM per (m/s) climb (immediate damping)
    stab_hover_learn_gain: float = 20.0  # Ki: PWM per (m/s) climb per second (slow trim)
    # Adaptive hover HOLDS altitude only while |thrust-0.5| < this on the open-loop
    # THRUST-STICK vertical path (the fallback); the closed-loop DIVE commands a
    # vertical RATE instead, which the loop tracks directly (band not used).
    stab_hover_learn_band: float = 0.05
    stab_hover_min_us: int = 1200        # safety clamp on the learned hover
    stab_hover_max_us: int = 1700
    angle_max_deg: float = 45.0
    pilot_yaw_rate_dps: float = 180.0
    rc_roll_channel: int = 1
    rc_pitch_channel: int = 2
    rc_throttle_channel: int = 3
    rc_yaw_channel: int = 4
    rc_roll_sign: int = 1
    rc_pitch_sign: int = 1
    rc_yaw_sign: int = 1
    # Startup FC validation: on boot, confirm the FC params the companion needs and
    # WRITE any that differ (verified). ANGLE_MAX (from angle_max_deg) and the
    # companion's RC channels' *_OPTION=0 are always enforced; `enforce_params` adds
    # explicit name->value overrides (e.g. {SR2_EXTRA2: 5}). Serial/baud are NOT
    # touched (the link the companion is on must already be correct to connect).
    enforce_params_on_start: bool = True
    enforce_params: Dict[str, float] = field(default_factory=dict)
    betaflight: Optional[BetaflightMapping] = None


@dataclass
class CpuSection:
    # Pin the NCNN detector worker to a dedicated core set so it can't starve
    # camera capture + the main pipeline loop. Linux + >=4 cores only; otherwise
    # a silent no-op. On the 4-core Zero 2W: pipeline gets {0,1}, detector {2,3}.
    pin: bool = True


@dataclass
class AppConfig:
    video: VideoSection
    camera: CameraSection
    detector: DetectorSection
    tracker: TrackerSection
    fc: FcSection
    servo: ServoConfig
    safety: SafetyConfig
    cpu: CpuSection


def _video(d: Dict[str, Any]) -> VideoSection:
    return VideoSection(
        tv_mode=d.get("tv_mode", "PAL"),
        width=d.get("width", 720),
        height=d.get("height", 576),
        framebuffer=d.get("framebuffer", "/dev/fb0"),
    )


def _camera(d: Dict[str, Any]) -> CameraSection:
    return CameraSection(
        type=d.get("type", "picam"),
        framerate=d.get("framerate", 30),
        imx500_model=d.get("imx500_model", ""),
        file_path=d.get("file_path", ""),
        webcam_device=d.get("webcam_device", 0),
        exposure_mode=d.get("exposure_mode", "short"),
        noise_reduction=d.get("noise_reduction", "fast"),
        hflip=d.get("hflip", False),
        vflip=d.get("vflip", False),
    )


def _detector(d: Dict[str, Any]) -> DetectorSection:
    return DetectorSection(
        type=d.get("type", "nanodet"),
        model_dir=d.get("model_dir", ""),
        input_size=d.get("input_size", 320),
        conf_threshold=d.get("conf_threshold", 0.35),
        nms_threshold=d.get("nms_threshold", 0.45),
        classes_of_interest=list(d.get("classes_of_interest", [])),
        detect_period_frames=d.get("detect_period_frames", 10),
    )


def _tracker(d: Dict[str, Any]) -> TrackerSection:
    # Accept the old "kcf" value as shorthand for classical+kcf
    type_val = d.get("type", "classical")
    cv2_backend = d.get("cv2_backend", "mosse")
    if type_val in ("kcf", "mosse", "csrt", "medianflow"):
        cv2_backend = type_val
        type_val = "classical"
    return TrackerSection(
        type=type_val,
        cv2_backend=cv2_backend,
        reacquire_after_lost_frames=d.get("reacquire_after_lost_frames", 30),
        iou_threshold=d.get("iou_threshold", 0.3),
    )


def _fc(d: Dict[str, Any]) -> FcSection:
    bf_d = d.get("betaflight")
    bf = None
    if bf_d:
        bf = BetaflightMapping(
            roll_us_per_deg=bf_d.get("roll_us_per_deg", 12.0),
            pitch_us_per_deg=bf_d.get("pitch_us_per_deg", -12.0),
            yaw_us_per_dps=bf_d.get("yaw_us_per_dps", 5.0),
            throttle_us_per_thrust=bf_d.get("throttle_us_per_thrust", 0.0),
            throttle_neutral_us=bf_d.get("throttle_neutral_us", 1500),
            stick_min_us=bf_d.get("stick_min_us", 1000),
            stick_max_us=bf_d.get("stick_max_us", 2000),
        )
    return FcSection(
        backend=d.get("backend", "ardupilot"),
        uart_device=d.get("uart_device", "/dev/ttyAMA0"),
        baud=d.get("baud", 115200),
        switch_channel=d.get("switch_channel", 7),
        select_channel=d.get("select_channel", 0),
        switch_threshold_us=d.get("switch_threshold_us", 1700),
        track_threshold_us=d.get("track_threshold_us", 1300),
        dive_threshold_us=d.get("dive_threshold_us", 1700),
        control_mode=d.get("control_mode", "stabilize"),
        stab_hover_throttle_us=d.get("stab_hover_throttle_us", 1450),
        stab_hover_learn=d.get("stab_hover_learn", True),
        stab_hover_learn_kp=d.get("stab_hover_learn_kp", 50.0),
        stab_hover_learn_gain=d.get("stab_hover_learn_gain", 20.0),
        stab_hover_learn_band=d.get("stab_hover_learn_band", 0.05),
        stab_hover_min_us=d.get("stab_hover_min_us", 1200),
        stab_hover_max_us=d.get("stab_hover_max_us", 1700),
        angle_max_deg=d.get("angle_max_deg", 45.0),
        pilot_yaw_rate_dps=d.get("pilot_yaw_rate_dps", 180.0),
        rc_roll_channel=d.get("rc_roll_channel", 1),
        rc_pitch_channel=d.get("rc_pitch_channel", 2),
        rc_throttle_channel=d.get("rc_throttle_channel", 3),
        rc_yaw_channel=d.get("rc_yaw_channel", 4),
        rc_roll_sign=d.get("rc_roll_sign", 1),
        rc_pitch_sign=d.get("rc_pitch_sign", 1),
        rc_yaw_sign=d.get("rc_yaw_sign", 1),
        enforce_params_on_start=d.get("enforce_params_on_start", True),
        enforce_params=dict(d.get("enforce_params", {})),
        betaflight=bf,
    )


def _servo(d: Dict[str, Any], width: int, height: int) -> ServoConfig:
    return ServoConfig(
        frame_width=width,
        frame_height=height,
        max_yaw_rate_dps=d.get("max_yaw_rate_dps", 60.0),
        max_pitch_deg=d.get("max_pitch_deg", 15.0),
        pixel_deadzone_px=d.get("pixel_deadzone_px", 20.0),
        yaw_p_gain=d.get("yaw_p_gain", 0.15),
        yaw_ff_gain=d.get("yaw_ff_gain", 0.05),
        lead_time_s=d.get("lead_time_s", 0.0),
        desired_bbox_frac=d.get("desired_bbox_frac", 0.30),
        closure_p_gain=d.get("closure_p_gain", 4.0),
        closure_i_gain=d.get("closure_i_gain", 0.0),
        pitch_p_gain=d.get("pitch_p_gain", 0.15),
        track_vcenter_gain=d.get("track_vcenter_gain", 0.10),
        dive_forward_deg=d.get("dive_forward_deg", 10.0),
        dive_climb_forward_deg=d.get("dive_climb_forward_deg", 6.0),
        dive_max_pitch_deg=d.get("dive_max_pitch_deg", 30.0),
        dive_lean_ramp_s=d.get("dive_lean_ramp_s", 0.5),
        dive_center_frac=d.get("dive_center_frac", 0.30),
        dive_vrate_gain=d.get("dive_vrate_gain", 0.0),
        dive_vrate_damp=d.get("dive_vrate_damp", 0.0),
        dive_max_descent_mps=d.get("dive_max_descent_mps", 8.0),
        dive_max_climb_mps=d.get("dive_max_climb_mps", 4.0),
        yaw_sign=d.get("yaw_sign", 1.0),
        pitch_sign=d.get("pitch_sign", 1.0),
    )


def _safety(d: Dict[str, Any]) -> SafetyConfig:
    timeout_ms = d.get("watchdog_timeout_ms", 250)
    return SafetyConfig(
        watchdog_timeout_s=timeout_ms / 1000.0,
        require_armed=d.get("require_armed", True),
        min_track_quality=d.get("min_track_quality", 0.35),
    )


_VALID_CONTROL_MODES = ("stabilize", "althold")


def _validate(cfg: AppConfig) -> None:
    """Catch dangerous/no-op misconfigurations at load time rather than in flight.

    - control_mode typo would silently fall back to the althold throttle formula
      AND disable the control_ready interlock (no expected mode) — so the
      companion would push sticks regardless of the FC's actual mode.
    - dive_threshold below track_threshold makes the 3-position switch reach DIVE
      (full commit) before TRACK, with TRACK unreachable — it would commit the
      aircraft where the pilot expected follow-only.
    """
    fc = cfg.fc
    if fc.backend == "ardupilot":
        if fc.control_mode not in _VALID_CONTROL_MODES:
            raise ValueError(
                f"fc.control_mode must be one of {_VALID_CONTROL_MODES}, "
                f"got {fc.control_mode!r}"
            )
        if fc.dive_threshold_us < fc.track_threshold_us:
            raise ValueError(
                f"fc.dive_threshold_us ({fc.dive_threshold_us}) must be >= "
                f"fc.track_threshold_us ({fc.track_threshold_us}); otherwise the "
                "switch reaches DIVE before TRACK and TRACK is unreachable"
            )
        # angle_max_deg is auto-written to the FC's ANGLE_MAX on boot, so a typo
        # here becomes a dangerous lean limit on the aircraft. Bound it.
        if not 0.0 < fc.angle_max_deg <= 80.0:
            raise ValueError(
                f"fc.angle_max_deg ({fc.angle_max_deg}) must be in (0, 80] — it is "
                "written to the FC's ANGLE_MAX, so an out-of-range value is unsafe"
            )

    # select_channel only does anything with the multi-target tracker; a non-zero
    # value on a single-target tracker silently no-ops the operator's select switch.
    if fc.select_channel and cfg.tracker.type != "multi_iou":
        raise ValueError(
            f"fc.select_channel ({fc.select_channel}) needs tracker.type 'multi_iou' "
            f"(got {cfg.tracker.type!r}); single-target trackers can't cycle targets"
        )

    if cfg.video.width <= 0 or cfg.video.height <= 0:
        raise ValueError(
            f"video.width/height must be > 0 (got {cfg.video.width}x{cfg.video.height}); "
            "the guidance servo divides by the frame size"
        )

    s = cfg.servo
    if s.dive_vrate_gain < 0.0:
        raise ValueError(
            f"guidance.dive_vrate_gain ({s.dive_vrate_gain}) must be >= 0 "
            "(m/s of climb command per unit normalised vertical frame error)"
        )
    if s.dive_max_descent_mps < 0.0 or s.dive_max_climb_mps < 0.0:
        raise ValueError(
            "guidance.dive_max_descent_mps / dive_max_climb_mps must be >= 0"
        )
    if s.dive_vrate_damp < 0.0:
        raise ValueError(
            f"guidance.dive_vrate_damp ({s.dive_vrate_damp}) must be >= 0 "
            "(derivative damping on the DIVE vertical homing; negative would amplify oscillation)"
        )
    if s.closure_i_gain < 0.0:
        raise ValueError(
            f"guidance.closure_i_gain ({s.closure_i_gain}) must be >= 0 "
            "(deg of forward lean per size-frac·s of accumulated range error; a "
            "negative gain inverts the closure integral and drives away from hold)"
        )


def load(path: str | Path) -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text())
    video = _video(raw.get("video", {}))
    cfg = AppConfig(
        video=video,
        camera=_camera(raw.get("camera", {})),
        detector=_detector(raw.get("detector", {})),
        tracker=_tracker(raw.get("tracker", {})),
        fc=_fc(raw.get("fc", {})),
        servo=_servo(raw.get("guidance", {}), video.width, video.height),
        safety=_safety(raw.get("safety", {})),
        cpu=CpuSection(pin=raw.get("cpu", {}).get("pin", True)),
    )
    _validate(cfg)
    return cfg
