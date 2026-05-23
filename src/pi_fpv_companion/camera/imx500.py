"""Sony IMX500 AI Camera via picamera2.

The IMX500 runs the neural network on-sensor (Sony AITRIOS NPU). The Pi just
reads frames + per-frame detection metadata — no CPU inference. This is the
strategic primary path for Pi Zero 2W; the PiCam+CPU path is a fallback for
users who don't have the IMX500.

Setup on Pi:
    sudo apt install -y imx500-all
    # ships /usr/share/imx500-models/ with several pre-converted .rpk files
    # the bundled SSD-MobileNetV2-FPN-Lite COCO model is the canonical starting point

`picamera2.devices.imx500.IMX500` is Pi-only. Imports are lazy.
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np

from pi_fpv_companion.camera.base import FrameBundle
from pi_fpv_companion.detect.nanodet import COCO_CLASSES
from pi_fpv_companion.types import Detection


_DEFAULT_MODEL = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"


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
    ) -> None:
        # target_class_ids: if empty, accept any class; else filter to these COCO ids
        self._model_path = model_path
        self._width = width
        self._height = height
        self._fps = framerate
        self._conf_threshold = conf_threshold
        self._target_class_set = set(target_class_ids) if target_class_ids else None
        self._imx500 = None
        self._picam = None
        self._intrinsics = None
        self._labels = COCO_CLASSES   # replaced with the model's own labels in open()
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

        # The .rpk carries its own label map (90 entries for the bundled SSD
        # COCO model — note: NOT the 80-class COCO_CLASSES). Use it for names.
        self._labels = getattr(self._intrinsics, "labels", None) or COCO_CLASSES

        self._picam = Picamera2(self._imx500.camera_num)
        config = self._picam.create_preview_configuration(
            main={"size": (self._width, self._height), "format": "BGR888"},
            controls={"FrameRate": float(self._fps)},
            buffer_count=12,
        )
        self._picam.configure(config)
        self._imx500.show_network_fw_progress_bar()
        self._picam.start(config, show_preview=False)
        self._running = True

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
        """Convert IMX500 NPU output (in metadata) to our Detection list.

        For the bundled SSD-MobileNet-V2 FPN-Lite model the on-sensor post-processing
        emits 4 tensors: bboxes (N,4), classes (N,), scores (N,), num_dets (1,).
        Boxes are normalized [0..1] in the network input frame; we map them back
        to the captured frame size.
        """
        if self._imx500 is None:
            return []
        outputs = self._imx500.get_outputs(metadata, add_batch=True)
        if outputs is None or len(outputs) < 4:
            return []

        # Verified on-device tensor order for the bundled SSD-MobileNetV2
        # FPN-Lite .rpk:  [0]=boxes (N,4) ymin,xmin,ymax,xmax normalized,
        # [1]=scores (N,) in 0..1, [2]=class ids (N,), [3]=count — but [3] is a
        # FIXED 100 (the candidate cap), not the valid count, and candidates are
        # score-sorted, so we rely on conf_threshold to cut the tail.
        try:
            boxes = np.array(outputs[0]).reshape(-1, 4)
            scores = np.array(outputs[1]).reshape(-1)
            classes = np.array(outputs[2]).reshape(-1).astype(int)
            n = int(np.array(outputs[3]).reshape(-1)[0])
        except (IndexError, ValueError):
            return []

        out: List[Detection] = []
        for i in range(min(n, len(boxes))):
            if scores[i] < self._conf_threshold:
                continue
            cid = int(classes[i])
            if self._target_class_set is not None and cid not in self._target_class_set:
                continue
            ymin, xmin, ymax, xmax = boxes[i]
            x1 = float(xmin * frame_w)
            y1 = float(ymin * frame_h)
            x2 = float(xmax * frame_w)
            y2 = float(ymax * frame_h)
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
