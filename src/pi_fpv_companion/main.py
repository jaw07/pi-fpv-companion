"""Production entry point.

Reads `config/<name>.yaml`, constructs the right Camera / Detector / Tracker /
FC backend / sink, and runs the Pipeline. The flight target is the Pi (IMX500 +
framebuffer/DRM TV out + real UART); video output goes to the analog composite,
so the only sinks are the framebuffer/DRM ones (run headless with --no-gui where
no framebuffer device exists, e.g. a dev laptop).

Usage:
    python -m pi_fpv_companion --config config/imx500.yaml
    python -m pi_fpv_companion --config config/mac-dev.yaml --no-gui

The factory functions below are the only place that knows about the concrete
implementations. Everything downstream speaks Protocols.
"""
from __future__ import annotations
import argparse
import signal
import sys
from pathlib import Path
from typing import Tuple

from pi_fpv_companion.config import AppConfig, load
from pi_fpv_companion.detect.nanodet import COCO_CLASSES
from pi_fpv_companion.perf import PerfMonitor, PiBudget
from pi_fpv_companion.pipeline import Pipeline
from pi_fpv_companion.types import GuidanceMode


def _resolve_class_ids(names) -> Tuple[int, ...]:
    """COCO class name list -> tuple of integer ids. Unknown names are dropped with a warning."""
    if not names:
        return ()
    name_to_id = {n: i for i, n in enumerate(COCO_CLASSES)}
    ids = []
    for n in names:
        if n in name_to_id:
            ids.append(name_to_id[n])
        else:
            print(f"WARN: unknown class name in classes_of_interest: {n!r}")
    return tuple(ids)


def _build_detector(cfg: AppConfig):
    t = cfg.detector.type
    if t == "none":
        return None
    if t == "color":
        from pi_fpv_companion.detect.color import ColorBlobDetector
        return ColorBlobDetector(min_area_px=400)
    if t == "haar":
        from pi_fpv_companion.detect.haar import HaarFaceDetector
        return HaarFaceDetector(min_size_px=60, downscale=0.5)
    target_ids = _resolve_class_ids(cfg.detector.classes_of_interest)
    if t == "nanodet":
        from pi_fpv_companion.detect.nanodet import NanoDetConfig, NanoDetDetector
        if not cfg.detector.model_dir:
            raise SystemExit("config.detector.model_dir is empty — set it to a NanoDet NCNN model dir")
        return NanoDetDetector(NanoDetConfig(
            model_dir=Path(cfg.detector.model_dir),
            input_size=cfg.detector.input_size,
            conf_threshold=cfg.detector.conf_threshold,
            nms_threshold=cfg.detector.nms_threshold,
            target_class_ids=target_ids,
        ))
    if t == "yolov8":
        from pi_fpv_companion.detect.yolov8 import Yolov8Config, Yolov8Detector
        if not cfg.detector.model_dir:
            raise SystemExit("config.detector.model_dir is empty — set it to a YOLOv8 NCNN model dir")
        return Yolov8Detector(Yolov8Config(
            model_dir=Path(cfg.detector.model_dir),
            input_size=cfg.detector.input_size,
            conf_threshold=cfg.detector.conf_threshold,
            nms_threshold=cfg.detector.nms_threshold,
            target_class_ids=target_ids,
        ))
    raise SystemExit(f"unknown detector type: {t}")


def _build_camera(cfg: AppConfig):
    t = cfg.camera.type
    if t == "synthetic":
        from pi_fpv_companion.camera.synthetic import SyntheticCamera
        return SyntheticCamera(width=cfg.video.width, height=cfg.video.height, fps=cfg.camera.framerate)
    if t == "file":
        from pi_fpv_companion.camera.file_camera import FileCamera
        if not cfg.camera.file_path:
            raise SystemExit("config.camera.file_path is empty for camera.type=file")
        return FileCamera(path=cfg.camera.file_path, fps_override=cfg.camera.framerate)
    if t == "webcam":
        from pi_fpv_companion.camera.webcam import WebcamCamera
        return WebcamCamera(
            device=cfg.camera.webcam_device,
            width=cfg.video.width, height=cfg.video.height,
            fps=cfg.camera.framerate,
        )
    if t == "picam":
        from pi_fpv_companion.camera.picam import PiCamCamera
        return PiCamCamera(
            width=cfg.video.width, height=cfg.video.height,
            framerate=cfg.camera.framerate,
            exposure_mode=cfg.camera.exposure_mode,
            noise_reduction=cfg.camera.noise_reduction,
            hflip=cfg.camera.hflip,
            vflip=cfg.camera.vflip,
        )
    if t == "imx500":
        from pi_fpv_companion.camera.imx500 import IMX500Camera
        model = cfg.camera.imx500_model or "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
        return IMX500Camera(
            model_path=model,
            width=cfg.video.width, height=cfg.video.height,
            framerate=cfg.camera.framerate,
            conf_threshold=cfg.detector.conf_threshold,
            target_class_ids=_resolve_class_ids(cfg.detector.classes_of_interest),
        )
    raise SystemExit(f"unknown camera type: {t}")


def _build_tracker(cfg: AppConfig):
    t = cfg.tracker.type
    if t == "iou":
        from pi_fpv_companion.track.iou_associator import IouAssociator
        return IouAssociator(
            iou_threshold=cfg.tracker.iou_threshold,
            max_lost_frames=cfg.tracker.reacquire_after_lost_frames,
        )
    if t == "multi_iou":   # multi-target IoU + operator selection (fc.select_channel)
        from pi_fpv_companion.track.multi_target import MultiObjectTracker
        return MultiObjectTracker(
            iou_threshold=cfg.tracker.iou_threshold,
            max_lost_frames=cfg.tracker.reacquire_after_lost_frames,
        )
    if t == "classical":
        from pi_fpv_companion.track.cv2_tracker import ClassicalCv2Tracker
        return ClassicalCv2Tracker(
            cv2_backend=cfg.tracker.cv2_backend,
            max_lost_frames=cfg.tracker.reacquire_after_lost_frames,
        )
    if t == "kcf":   # backward-compat alias
        from pi_fpv_companion.track.kcf_tracker import KcfTracker
        return KcfTracker(max_lost_frames=cfg.tracker.reacquire_after_lost_frames)
    raise SystemExit(f"unknown tracker type: {t}")


def _build_fc(cfg: AppConfig):
    if cfg.fc.backend == "ardupilot":
        from pi_fpv_companion.fc.ardupilot import ArduPilotBackend, ArduCopterRcMapping
        return ArduPilotBackend(
            device=cfg.fc.uart_device, baud=cfg.fc.baud,
            switch_channel=cfg.fc.switch_channel,
            select_channel=cfg.fc.select_channel,
            track_threshold_us=cfg.fc.track_threshold_us,
            dive_threshold_us=cfg.fc.dive_threshold_us,
            mapping=ArduCopterRcMapping(
                control_mode=cfg.fc.control_mode,
                hover_throttle_us=cfg.fc.stab_hover_throttle_us,
                hover_learn=cfg.fc.stab_hover_learn,
                hover_learn_kp=cfg.fc.stab_hover_learn_kp,
                hover_learn_gain=cfg.fc.stab_hover_learn_gain,
                hover_learn_band=cfg.fc.stab_hover_learn_band,
                hover_min_us=cfg.fc.stab_hover_min_us,
                hover_max_us=cfg.fc.stab_hover_max_us,
                angle_max_deg=cfg.fc.angle_max_deg,
                pilot_yaw_rate_dps=cfg.fc.pilot_yaw_rate_dps,
                roll_channel=cfg.fc.rc_roll_channel,
                pitch_channel=cfg.fc.rc_pitch_channel,
                throttle_channel=cfg.fc.rc_throttle_channel,
                yaw_channel=cfg.fc.rc_yaw_channel,
                roll_sign=cfg.fc.rc_roll_sign,
                pitch_sign=cfg.fc.rc_pitch_sign,
                yaw_sign=cfg.fc.rc_yaw_sign,
            ),
        )
    if cfg.fc.backend == "betaflight":
        from pi_fpv_companion.fc.betaflight import BetaflightBackend
        if cfg.fc.betaflight is None:
            raise SystemExit("config.fc.betaflight (mapping) is required when backend=betaflight")
        return BetaflightBackend(
            device=cfg.fc.uart_device, baud=cfg.fc.baud,
            switch_channel=cfg.fc.switch_channel,
            switch_threshold_us=cfg.fc.switch_threshold_us,
            mapping=cfg.fc.betaflight,
        )
    raise SystemExit(f"unknown fc backend: {cfg.fc.backend}")


def _enforce_fc_params(cfg: AppConfig, fc) -> None:
    """Startup FC validation: confirm the params the companion needs and write any
    that differ. Always enforces ANGLE_MAX (= angle_max_deg, so commanded lean =
    actual lean) and the companion's RC channels' *_OPTION = 0 (so the FC leaves
    them for us); plus any fc.enforce_params overrides. Does NOT touch serial/baud
    (the link we're on must already be right). Skipped if disabled or unsupported."""
    if not getattr(cfg.fc, "enforce_params_on_start", False):
        return
    ensure = getattr(fc, "ensure_params", None)
    if not callable(ensure):
        return
    desired: dict = {"ANGLE_MAX": round(cfg.fc.angle_max_deg * 100.0)}
    desired[f"RC{cfg.fc.switch_channel}_OPTION"] = 0
    if cfg.fc.select_channel:
        desired[f"RC{cfg.fc.select_channel}_OPTION"] = 0
    desired.update(cfg.fc.enforce_params)
    print("  validating FC params ...")
    try:
        status = ensure(desired)
    except Exception as e:
        print(f"  WARN: FC param validation failed: {e}")
        return
    for name, st in status.items():
        print(f"    {name}: {st}")
    # guided_nogps RATE path: the thrust field MUST be real throttle, not a climb-rate,
    # or the dive planes. Enforce GUID_OPTIONS bit 3 (ThrustAsThrust), OR-ing it in so
    # other guided bits are preserved. (Bench finding: a wrong/missing bit silently turns
    # "throttle 0" into "hold altitude".)
    if cfg.fc.control_mode == "guided_nogps":
        ensure_bits = getattr(fc, "ensure_param_bits", None)
        if callable(ensure_bits):
            from pi_fpv_companion.fc.ardupilot import GUID_OPTIONS_THRUST_AS_THRUST
            try:
                st = ensure_bits("GUID_OPTIONS", GUID_OPTIONS_THRUST_AS_THRUST)
                print(f"    GUID_OPTIONS(ThrustAsThrust): {st}")
            except Exception as e:
                print(f"  WARN: GUID_OPTIONS check failed: {e}")


def _build_sink(cfg: AppConfig, no_gui: bool):
    if no_gui:
        return None
    from pi_fpv_companion.video.framebuffer import FramebufferSink

    # Prefer the legacy /dev/fb0 path if available (older Pi OS, or fkms).
    # Fall back to DRM dumb-buffer on /dev/dri/card0 (Trixie + default KMS).
    # Flight output is the analog composite / TV out via one of these; there is
    # no on-screen-window path. With no framebuffer device, run with --no-gui.
    fb = cfg.video.framebuffer
    if fb == "/dev/fb0" and Path(fb).exists():
        from pi_fpv_companion.video.framebuffer import LinuxFramebuffer
        return FramebufferSink(LinuxFramebuffer(device=fb))
    if Path("/dev/dri/card0").exists():
        from pi_fpv_companion.video.drm_framebuffer import DrmFramebuffer
        return FramebufferSink(DrmFramebuffer())
    raise SystemExit(
        "no framebuffer device (/dev/fb0 or /dev/dri/card0) for video output; "
        "pass --no-gui to run headless"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pi-fpv-companion")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--force-mode", choices=["standby", "track", "dive"], default=None,
                    help="bench/test: force the guidance mode, ignoring the RC switch")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="stop after N seconds (0 = run until SIGINT/SIGTERM)")
    ap.add_argument("--pi-scale", type=float, default=6.0,
                    help="Mac->Pi scaling factor for perf estimates (see perf.py docstring)")
    args = ap.parse_args(argv)

    cfg = load(args.config)
    print(f"loaded config: {args.config}")
    print(f"  camera   {cfg.camera.type}")
    print(f"  detector {cfg.detector.type}  (period={cfg.detector.detect_period_frames} frames)")
    if cfg.detector.type in ("nanodet", "yolov8"):
        print("  WARN: CPU detector is a DEV/SIM path (~4 Hz on Zero 2W — a "
              "slideshow, not a tracker). Flight detector is IMX500 "
              "(camera.type: imx500). See docs/architecture-audit.md §3.")
    print(f"  tracker  {cfg.tracker.type}")
    print(f"  fc       {cfg.fc.backend} @ {cfg.fc.uart_device}")
    if cfg.detector.classes_of_interest:
        print(f"  classes  {cfg.detector.classes_of_interest}")

    from pi_fpv_companion.cpu_affinity import compute_split, pin_current_thread
    pipeline_cores, detector_cores = compute_split(cfg.cpu.pin)
    if pipeline_cores is not None:
        # Pin the main process (main loop + camera capture thread + output) to
        # the pipeline cores; the detector worker re-pins itself to its own set.
        pin_current_thread(pipeline_cores)
        print(f"  cpu      pipeline={sorted(pipeline_cores)} detector={sorted(detector_cores)}")

    detector = _build_detector(cfg)
    camera = _build_camera(cfg)
    tracker = _build_tracker(cfg)
    fc = _build_fc(cfg)
    sink = _build_sink(cfg, no_gui=args.no_gui)
    perf = PerfMonitor(PiBudget(max_tick_ms=33.0, max_rss_mb=200.0, pi_scale_factor=args.pi_scale))

    fc.open()
    if hasattr(fc, "wait_ready"):
        try:
            fc.wait_ready(timeout=10.0)
        except Exception as e:
            print(f"WARN: FC didn't return heartbeat within timeout: {e}")
        _enforce_fc_params(cfg, fc)

    def on_status(target, intent, gated, switch, armed, frame, tracks=None):
        perf.tick_end(on_status._t0)
        if sink is not None:
            sink.show(target, intent, gated, switch, armed, frame, tracks)

    on_status._t0 = 0.0

    force_mode = GuidanceMode[args.force_mode.upper()] if args.force_mode else None
    if force_mode is not None:
        print(f"  FORCE    mode={force_mode.name} (ignoring RC switch — bench/test)")

    pipeline = Pipeline(
        camera, tracker, cfg.servo, cfg.safety, fc,
        detector=detector,
        detect_period_frames=cfg.detector.detect_period_frames,
        detector_cpu_affinity=detector_cores,
        on_status=on_status,
        force_mode=force_mode,
        camera_watchdog_s=5.0,   # restart the process if the camera stalls
    )

    orig_tick = pipeline.tick

    def timed_tick(bundle):
        on_status._t0 = perf.tick_start()
        return orig_tick(bundle)
    pipeline.tick = timed_tick

    signal.signal(signal.SIGINT, lambda *_: pipeline.stop())
    signal.signal(signal.SIGTERM, lambda *_: pipeline.stop())

    if args.duration > 0:
        import threading
        threading.Timer(args.duration, pipeline.stop).start()

    try:
        pipeline.run()
    finally:
        fc.close()
        if sink is not None:
            sink.close()

    print()
    print(perf.report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
