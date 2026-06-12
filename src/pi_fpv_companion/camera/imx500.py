"""Sony IMX500 AI Camera via picamera2.

The IMX500 runs the neural network on-sensor (Sony AITRIOS NPU). The Pi just
reads frames + per-frame detection metadata — no CPU inference. This is the
flight camera and the only real-camera path.

Setup on Pi:
    sudo apt install -y imx500-all
    # ships /usr/share/imx500-models/ with several pre-converted .rpk files
    # the bundled SSD-MobileNetV2-FPN-Lite COCO model is the canonical starting point

`picamera2.devices.imx500.IMX500` is Pi-only. Imports are lazy.
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np

from pi_fpv_companion.camera.base import FrameBundle
from pi_fpv_companion.detect.coco import COCO_CLASSES
from pi_fpv_companion.types import Detection


_DEFAULT_MODEL = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"


@dataclass(frozen=True)
class DecoderProfile:
    """How to read one model's 4 output tensors. Different on-sensor-postprocessed
    (_pp) models emit the SAME four tensors (boxes, scores, classes, count) but with
    different box ORDER, box SCALE, and a real-vs-fixed count — so the decoder needs a
    per-model profile rather than the single hardcoded SSD layout it began with.

      box_order:  'yxyx' (SSD-MobileNet / NanoDet) | 'xyxy' (YOLOv8n / YOLO11n)
      box_scale:  'normalized' (0..1 of the network input — SSD/NanoDet) |
                  'input_px'   (0..input_size px — YOLO; divided back to 0..1 here)
      count_is_real: [3] is the valid detection count (YOLO) vs a fixed candidate
                  cap of 100 (SSD), in which case we rely on conf_threshold to cut.
      labels: name table; None -> the model's intrinsics labels, else COCO_CLASSES."""
    box_order: str = "yxyx"
    box_scale: str = "normalized"
    input_size: int = 320
    count_is_real: bool = False
    labels: Optional[Tuple[str, ...]] = None

    @classmethod
    def for_model(cls, model_path: str) -> "DecoderProfile":
        """Pick a profile from the .rpk filename. YOLOv8n/YOLO11n share the Ultralytics
        export convention (xyxy, input-pixel boxes, real count, COCO-80); everything
        else keeps the original SSD/NanoDet layout (yxyx, normalized, fixed count)."""
        name = Path(model_path).name.lower()
        if "yolo" in name:
            return cls(box_order="xyxy", box_scale="input_px", input_size=640,
                       count_is_real=True, labels=tuple(COCO_CLASSES))
        return cls()   # SSD / NanoDet default


class IMX500Camera:
    """IMX500 camera. Outputs FrameBundle with detections already populated by the
    sensor — the downstream detector module should be a no-op."""

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL,
        width: int = 720,
        height: int = 576,
        framerate: int = 30,
        conf_threshold: float = 0.35,
        target_class_ids: Tuple[int, ...] = (),
        zoom: float = 1.0,
    ) -> None:
        # target_class_ids: if empty, accept any class; else filter to these COCO ids
        self._model_path = model_path
        self._width = width
        self._height = height
        self._fps = framerate
        self._conf_threshold = conf_threshold
        self._target_class_set = set(target_class_ids) if target_class_ids else None
        # Digital zoom: >1.0 centre-crops the sensor (ScalerCrop) so a distant target
        # fills more of the network input -> better far-target detection, at the cost
        # of FOV. 1.0 = full frame (off). Applied in open().
        self._zoom = max(1.0, float(zoom))
        self._imx500 = None
        self._picam = None
        self._intrinsics = None
        self._profile = DecoderProfile.for_model(model_path)   # box order/scale/labels per model
        # Network input size for box de-normalization; the profile default is a
        # guess (640 for YOLO), REPLACED in open() by the sensor's real value via
        # get_input_size() so 320/416/640 models all decode correctly.
        self._input_size = (self._profile.input_size, self._profile.input_size)
        self._labels = self._profile.labels or COCO_CLASSES    # may be replaced by intrinsics in open()
        self._running = False

    def open(self) -> None:
        from picamera2 import Picamera2
        from picamera2.devices import IMX500

        if not Path(self._model_path).exists():
            raise RuntimeError(f"IMX500 model file not found: {self._model_path}")

        self._imx500 = IMX500(self._model_path)
        self._intrinsics = self._imx500.network_intrinsics
        if self._intrinsics is None:
            # Older firmware; fall back to defaults
            from picamera2.devices.imx500 import NetworkIntrinsics
            self._intrinsics = NetworkIntrinsics()
            self._intrinsics.task = "object detection"

        # Label table: the profile's own map wins (YOLO ships none in intrinsics and
        # is canonical COCO-80); otherwise the .rpk's intrinsics labels (the bundled
        # SSD COCO model carries a 90-entry map — NOT the 80-class COCO_CLASSES);
        # COCO_CLASSES is the final fallback.
        self._labels = (self._profile.labels
                        or getattr(self._intrinsics, "labels", None)
                        or COCO_CLASSES)

        self._picam = Picamera2(self._imx500.camera_num)
        config = self._picam.create_preview_configuration(
            main={"size": (self._width, self._height), "format": "BGR888"},
            controls={"FrameRate": float(self._fps)},
            buffer_count=12,
        )
        self._picam.configure(config)
        self._imx500.show_network_fw_progress_bar()
        self._picam.start(config, show_preview=False)
        # The network's TRUE input size (e.g. 416 or 640) — YOLO emits boxes in
        # input-PIXEL space, so the decoder must divide by THIS, not a hardcoded
        # guess. A 416 model's boxes divided by 640 collapse to ~65% of frame,
        # top-left — a persistent mis-placed phantom box. Read it from the sensor.
        try:
            iw, ih = self._imx500.get_input_size()
            if iw and ih:
                self._input_size = (int(iw), int(ih))
        except Exception:
            pass   # fall back to the profile default set in __init__
        self._apply_zoom()
        self._running = True

    def _apply_zoom(self) -> None:
        """Centre-crop the sensor for digital zoom (no-op when zoom<=1). Uses
        ScalerCrop relative to the sensor's full active area so the cropped (zoomed)
        image is what's scaled into the network input as well as the displayed frame."""
        if self._zoom <= 1.0:
            return
        try:
            full = self._picam.camera_properties.get("ScalerCropMaximum")
            if not full:
                return
            fx, fy, fw, fh = full
            cw, ch = int(fw / self._zoom), int(fh / self._zoom)
            cx, cy = fx + (fw - cw) // 2, fy + (fh - ch) // 2
            self._picam.set_controls({"ScalerCrop": (cx, cy, cw, ch)})
        except Exception:
            # Non-fatal: fall back to full FOV if the platform rejects the crop.
            pass

    def close(self) -> None:
        self._running = False
        if self._picam is not None:
            self._picam.stop()
            self._picam.close()
            self._picam = None
        self._imx500 = None
        self._intrinsics = None

    def frames(self) -> Iterator[FrameBundle]:
        if not self._running:
            self.open()
        while self._running:
            request = self._picam.capture_request()
            try:
                frame = request.make_array("main")
                metadata = request.get_metadata()
            finally:
                request.release()

            dets = self._decode_detections(metadata, frame.shape[1], frame.shape[0])
            yield FrameBundle(
                image=frame, width=frame.shape[1], height=frame.shape[0],
                timestamp=time.monotonic(), detections=dets,
            )

    def _decode_detections(self, metadata: dict, frame_w: int, frame_h: int) -> List[Detection]:
        """Convert the IMX500 NPU output (4 tensors: boxes, scores, classes, count)
        to our Detection list, per the model's DecoderProfile (box order/scale/count).
        Boxes are mapped to a 0..1 fraction of the network input, then to the captured
        frame size."""
        if self._imx500 is None:
            return []
        outputs = self._imx500.get_outputs(metadata, add_batch=True)
        if outputs is None or len(outputs) < 4:
            return []
        try:
            boxes = np.array(outputs[0]).reshape(-1, 4)
            scores = np.array(outputs[1]).reshape(-1)
            classes = np.array(outputs[2]).reshape(-1).astype(int)
            raw_n = int(np.array(outputs[3]).reshape(-1)[0])
        except (IndexError, ValueError):
            return []

        p = self._profile
        # YOLO's [3] is the real count; SSD's is a fixed 100 cap (score-sorted
        # candidates), so there we scan all and lean on conf_threshold.
        n = raw_n if p.count_is_real else len(boxes)
        # YOLO boxes come in network-input PIXELS (0..input_w); SSD already 0..1.
        # Use the sensor's REAL input width (set in open()), not the profile guess.
        box_div = float(self._input_size[0]) if p.box_scale == "input_px" else 1.0

        out: List[Detection] = []
        for i in range(min(n, len(boxes))):
            if scores[i] < self._conf_threshold:
                continue
            cid = int(classes[i])
            if self._target_class_set is not None and cid not in self._target_class_set:
                continue
            if p.box_order == "xyxy":
                xmin, ymin, xmax, ymax = boxes[i] / box_div
            else:  # yxyx (SSD / NanoDet)
                ymin, xmin, ymax, xmax = boxes[i] / box_div
            x1, y1 = float(xmin * frame_w), float(ymin * frame_h)
            x2, y2 = float(xmax * frame_w), float(ymax * frame_h)
            if x2 <= x1 or y2 <= y1:
                continue
            out.append(Detection(
                x=(x1 + x2) / 2, y=(y1 + y2) / 2,
                w=x2 - x1, h=y2 - y1,
                confidence=float(scores[i]),
                class_id=cid,
                class_name=self._labels[cid] if 0 <= cid < len(self._labels) else "unknown",
            ))
        return out
