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
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Tuple

from pi_fpv_companion.config import AppConfig, load
from pi_fpv_companion.detect.coco import COCO_CLASSES
from pi_fpv_companion.perf import PerfMonitor, PiBudget
from pi_fpv_companion.pipeline import Pipeline
from pi_fpv_companion.types import GuidanceMode


def _resolve_class_ids(names, labels=COCO_CLASSES) -> Tuple[int, ...]:
    """Class name list -> integer ids against `labels` (the model's OWN label set —
    COCO-80 or, for a VisDrone-fine-tuned model, the 10 VisDrone classes). Resolving
    against the wrong set silently mis-indexes (e.g. 'car' is COCO id 2 but VisDrone
    id 3), so the caller passes the model's labels. Unknown names dropped with a warning."""
    if not names:
        return ()
    name_to_id = {n: i for i, n in enumerate(labels)}
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
    if t == "imx500":
        from pi_fpv_companion.camera.imx500 import IMX500Camera, DecoderProfile
        from pi_fpv_companion.detect.coco import COCO_CLASSES as _COCO
        model = cfg.camera.imx500_model or "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
        # Resolve classes_of_interest against the model's OWN label set (VisDrone vs
        # COCO), the same set the decoder uses — see DecoderProfile.for_model.
        labels = DecoderProfile.for_model(model).labels or _COCO
        return IMX500Camera(
            model_path=model,
            width=cfg.video.width, height=cfg.video.height,
            framerate=cfg.camera.framerate,
            conf_threshold=cfg.detector.conf_threshold,
            target_class_ids=_resolve_class_ids(cfg.detector.classes_of_interest, labels),
            zoom=cfg.camera.zoom,
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
            confirm_hits=cfg.tracker.confirm_hits,
            confirm_window=cfg.tracker.confirm_window,
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
            auto_guided=cfg.fc.auto_guided,
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
    if cfg.fc.backend == "none":
        from pi_fpv_companion.fc.null import NullFC
        return NullFC()
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
    # NEVER touch FC params while ARMED. The pass re-runs on every service start,
    # including a camera-watchdog restart MID-FLIGHT — param reads/writes then are
    # pointless traffic at best (values already match from the preflight boot) and a
    # mid-air EEPROM write at worst. Wait briefly for a HEARTBEAT to learn the armed
    # state; unknown (FC silent) falls through to the normal enforce, which aborts
    # fast on its own when the link is down.
    known = getattr(fc, "armed_known", None)
    if callable(known):
        deadline = time.monotonic() + 2.0
        while not known() and time.monotonic() < deadline:
            time.sleep(0.05)
        if known() and fc.is_armed():
            print("  FC is ARMED — skipping param validation (mid-flight restart; "
                  "params were enforced at the preflight boot)")
            return
    desired: dict = {"ANGLE_MAX": round(cfg.fc.angle_max_deg * 100.0)}
    desired[f"RC{cfg.fc.switch_channel}_OPTION"] = 0
    if cfg.fc.select_channel:
        desired[f"RC{cfg.fc.select_channel}_OPTION"] = 0
    desired.update(cfg.fc.enforce_params)
    print("  validating FC params ...")
    try:
        status = ensure(desired)
        # One retry for failures: ensure_params aborts the pass on the first slow
        # read (a busy FC at boot), which would otherwise silently skip params
        # later in the list — including safety-critical ones like FS_GCS_TIMEOUT.
        failed = {n: desired[n] for n, st in status.items()
                  if st in ("read-fail", "write-fail")}
        if failed:
            print(f"  retrying {len(failed)} failed param(s) ...")
            status.update(ensure(failed))
    except Exception as e:
        print(f"  WARN: FC param validation failed: {e}")
        return
    for name, st in status.items():
        print(f"    {name}: {st}")
    bad = [n for n, st in status.items() if st not in ("ok", "set")]
    if bad:
        # Loud, because flying on FC-side defaults can be unsafe: e.g. an
        # unenforced FS_GCS_TIMEOUT (5 s default) means the next camera-watchdog
        # restart trips the GCS failsafe and the FC LANDs itself mid-flight.
        print(f"  ERROR: FC params NOT enforced: {', '.join(bad)} — the FC is flying "
              "on its stored values for these. Verify them in Mission Planner "
              "before flight.")
    # guided_nogps RATE path: the thrust field MUST be real throttle, not a climb-rate,
    # or the dive planes. Enforce GUID_OPTIONS bit 3 (ThrustAsThrust), OR-ing it in so
    # other guided bits are preserved. (Bench finding: a wrong/missing bit silently turns
    # "throttle 0" into "hold altitude".)
    ensure_bits = getattr(fc, "ensure_param_bits", None)
    if cfg.fc.control_mode == "guided_nogps" and callable(ensure_bits):
        from pi_fpv_companion.fc.ardupilot import GUID_OPTIONS_THRUST_AS_THRUST
        try:
            st = ensure_bits("GUID_OPTIONS", GUID_OPTIONS_THRUST_AS_THRUST)
            print(f"    GUID_OPTIONS(ThrustAsThrust): {st}")
        except Exception as e:
            print(f"  WARN: GUID_OPTIONS check failed: {e}")
    if callable(ensure_bits):
        # Scope the GCS failsafe to modes where the companion actually has authority:
        # without FS_OPTIONS bit 4, a Pi death/restart LANDs a craft being flown
        # manually on the sticks (see FS_OPTIONS_CONTINUE_PILOT_GCS).
        from pi_fpv_companion.fc.ardupilot import FS_OPTIONS_CONTINUE_PILOT_GCS
        try:
            st = ensure_bits("FS_OPTIONS", FS_OPTIONS_CONTINUE_PILOT_GCS)
            print(f"    FS_OPTIONS(ContinuePilotModesOnGCSLoss): {st}")
        except Exception as e:
            print(f"  WARN: FS_OPTIONS check failed: {e}")


def _build_sink(cfg: AppConfig, no_gui: bool):
    if no_gui:
        return None
    # BENCH: browser MJPEG stream instead of the TV-out (no VTX/composite needed).
    if cfg.video.web_stream_port:
        from pi_fpv_companion.video.mjpeg_stream import MjpegStreamSink
        return MjpegStreamSink(port=cfg.video.web_stream_port,
                               quality=cfg.video.web_stream_quality,
                               max_fps=cfg.video.web_stream_fps)
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

    # Without this, Python logging has NO handler: INFO records (param-ok lines, mode
    # confirmations — the breadcrumbs that reconstruct a flight) are dropped entirely,
    # and only WARNING+ reach stderr via the last-resort handler. journald adds its
    # own timestamps, so the format stays terse.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load(args.config)
    print(f"loaded config: {args.config}")
    print(f"  camera   {cfg.camera.type}")
    print(f"  detector {cfg.detector.type}  (period={cfg.detector.detect_period_frames} frames)")
    print(f"  tracker  {cfg.tracker.type}")
    print(f"  fc       {cfg.fc.backend} @ {cfg.fc.uart_device}")
    if cfg.detector.classes_of_interest:
        print(f"  classes  {cfg.detector.classes_of_interest}")

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

    recorder = None
    if cfg.recorder.enabled:
        from pi_fpv_companion.flight_log import FlightRecorder
        recorder = FlightRecorder(cfg.recorder.directory, rate_hz=cfg.recorder.rate_hz,
                                  max_bytes=cfg.recorder.max_bytes,
                                  keep_files=cfg.recorder.keep_files)
        print(f"  recorder {cfg.recorder.directory} @ {cfg.recorder.rate_hz:g} Hz")

    def on_status(target, intent, gated, switch, armed, frame, tracks=None):
        perf.tick_end(on_status._t0)
        if recorder is not None:
            recorder.record(target, intent, gated, switch, armed)
        if sink is not None:
            sink.show(target, intent, gated, switch, armed, frame, tracks)

    on_status._t0 = 0.0

    force_mode = GuidanceMode[args.force_mode.upper()] if args.force_mode else None
    if force_mode is not None:
        print(f"  FORCE    mode={force_mode.name} (ignoring RC switch — bench/test)")

    # guided_nogps body-RATE path: build a RateConfig (frame dims from the camera) so the
    # pipeline dispatches to the rate controller. None for STABILIZE/ALT_HOLD (RC-override path).
    rate_cfg = None
    if cfg.fc.control_mode == "guided_nogps":
        from pi_fpv_companion.guidance.rate_control import RateConfig
        rate_cfg = RateConfig(frame_width=cfg.video.width, frame_height=cfg.video.height)
        print(f"  guided_nogps RATE path active (frame {cfg.video.width}x{cfg.video.height})")

    pipeline = Pipeline(
        camera, tracker, cfg.servo, cfg.safety, fc,
        detector=detector,
        detect_period_frames=cfg.detector.detect_period_frames,
        on_status=on_status,
        force_mode=force_mode,
        camera_watchdog_s=2.0,   # restart the process if the camera stalls (≈50 frames @25fps)
        # bail+restart if a (re)opened camera gives no frame within the grace; must
        # cover the IMX500 firmware upload, which varies per Pi (see CameraSection)
        first_frame_grace_s=cfg.camera.first_frame_grace_s,
        rate_cfg=rate_cfg,
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
        if recorder is not None:
            recorder.close()
        if sink is not None:
            sink.close()

    print()
    print(perf.report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
