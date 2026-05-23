"""YOLOv8 (ultralytics) NCNN detector.

Alternative to NanoDet for users who already have YOLOv8n NCNN model files
(e.g. exported via `yolo export model=yolov8n.pt format=ncnn`). drone-guidance
ships one at `models/yolov8n_ncnn_model/`.

Output format from ultralytics NCNN export:
    tensor shape (1, num_classes+4, num_anchors)
    e.g. (1, 84, 8400) for COCO @ 640x640

Channel layout (axis 1):
    [0:4]  cx, cy, w, h in input-pixel coordinates
    [4:]   per-class scores (already sigmoid'd)

No objectness score — confidence is just max(class_scores).

**NOT VIABLE on Pi Zero 2W** — measured on hardware: YOLOv8n inference OOM-reboots
the Pi at any input size. Activation tensors exceed the 416 MB RAM budget. Use
NanoDet-Plus on Zero 2W. This class is retained for Pi 3B+ / Pi 4 / larger SBCs
where memory isn't the binding constraint.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from pi_fpv_companion.detect.nanodet import COCO_CLASSES
from pi_fpv_companion.types import Detection


@dataclass(frozen=True)
class Yolov8Config:
    model_dir: Path                       # contains model.ncnn.param + model.ncnn.bin
    input_size: int = 256
    num_classes: int = 80
    conf_threshold: float = 0.35
    nms_threshold: float = 0.45
    target_class_ids: Tuple[int, ...] = ()
    num_threads: int = 2
    class_names: Tuple[str, ...] = tuple(COCO_CLASSES)
    # Ultralytics YOLOv8 NCNN uses these I/O blob names by default.
    # If `yolo export ... format=ncnn` produced different names, override here.
    input_blob: str = "in0"
    output_blob: str = "out0"


class Yolov8Detector:
    def __init__(self, cfg: Yolov8Config) -> None:
        self._cfg = cfg
        self._net = None
        self._letterbox = np.full((cfg.input_size, cfg.input_size, 3), 114, dtype=np.uint8)
        self._target_class_set = set(cfg.target_class_ids) if cfg.target_class_ids else None

    def open(self) -> None:
        import ncnn
        param = self._cfg.model_dir / "model.ncnn.param"
        binf = self._cfg.model_dir / "model.ncnn.bin"
        if not param.exists() or not binf.exists():
            raise RuntimeError(
                f"YOLOv8 model files missing under {self._cfg.model_dir}: "
                f"expected model.ncnn.param + model.ncnn.bin"
            )
        net = ncnn.Net()
        net.opt.num_threads = self._cfg.num_threads
        net.opt.use_vulkan_compute = False
        net.load_param(str(param))
        net.load_model(str(binf))
        self._net = net

    def close(self) -> None:
        self._net = None

    def _letterbox_preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        h, w = image.shape[:2]
        size = self._cfg.input_size
        scale = min(size / h, size / w)
        nh, nw = int(h * scale), int(w * scale)
        pad_h = (size - nh) // 2
        pad_w = (size - nw) // 2
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        self._letterbox.fill(114)
        self._letterbox[pad_h:pad_h + nh, pad_w:pad_w + nw] = resized
        return self._letterbox, scale, pad_w, pad_h

    def detect(self, image) -> List[Detection]:
        if self._net is None:
            self.open()
        import ncnn

        orig_h, orig_w = image.shape[:2]
        letterboxed, scale, pad_w, pad_h = self._letterbox_preprocess(image)

        # YOLOv8 NCNN export normalizes by 1/255; mean is 0, std is 1/255.
        mat = ncnn.Mat.from_pixels(
            letterboxed,
            ncnn.Mat.PixelType.PIXEL_BGR2RGB,            # ultralytics trains in RGB
            self._cfg.input_size,
            self._cfg.input_size,
        )
        mat.substract_mean_normalize([0.0, 0.0, 0.0], [1.0 / 255.0] * 3)

        ex = self._net.create_extractor()
        ex.input(self._cfg.input_blob, mat)
        _, output = ex.extract(self._cfg.output_blob)
        out_np = np.array(output)

        return self._decode(out_np, scale, pad_w, pad_h, orig_w, orig_h)

    def _decode(
        self,
        output: np.ndarray,
        scale: float,
        pad_w: int,
        pad_h: int,
        orig_w: int,
        orig_h: int,
    ) -> List[Detection]:
        cfg = self._cfg
        # NCNN strips the batch dim, so out is (C, N). Transpose to (N, C) for per-anchor work.
        if output.ndim == 2 and output.shape[0] == cfg.num_classes + 4:
            arr = output.T                              # (N, C)
        elif output.ndim == 2 and output.shape[1] == cfg.num_classes + 4:
            arr = output                                # already (N, C)
        else:
            raise RuntimeError(
                f"unexpected YOLOv8 output shape {output.shape}; "
                f"expected (C={cfg.num_classes+4}, N) or (N, C)"
            )

        boxes_cxywh = arr[:, :4]                        # cx, cy, w, h in input pixels
        scores = arr[:, 4:]
        max_scores = scores.max(axis=1)
        class_ids = scores.argmax(axis=1)

        keep = max_scores > cfg.conf_threshold
        if self._target_class_set is not None:
            keep &= np.isin(class_ids, list(self._target_class_set))
        if not keep.any():
            return []

        boxes_cxywh = boxes_cxywh[keep]
        scores = max_scores[keep]
        class_ids = class_ids[keep]

        # Convert cxcywh -> xyxy in input pixels
        cx, cy, bw, bh = boxes_cxywh.T
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # NMS in input space (consistent units)
        rects = np.stack([x1, y1, bw, bh], axis=1).astype(np.float32)
        idx = cv2.dnn.NMSBoxes(
            rects.tolist(), scores.astype(np.float32).tolist(),
            cfg.conf_threshold, cfg.nms_threshold,
        )
        if len(idx) == 0:
            return []
        idx = idx.flatten() if hasattr(idx, "flatten") else idx

        out: List[Detection] = []
        for i in idx:
            ix1, iy1, ix2, iy2 = x1[i], y1[i], x2[i], y2[i]
            # Unproject the letterbox
            ox1 = max(0.0, min(orig_w, (ix1 - pad_w) / scale))
            oy1 = max(0.0, min(orig_h, (iy1 - pad_h) / scale))
            ox2 = max(0.0, min(orig_w, (ix2 - pad_w) / scale))
            oy2 = max(0.0, min(orig_h, (iy2 - pad_h) / scale))
            if ox2 <= ox1 or oy2 <= oy1:
                continue
            cid = int(class_ids[i])
            out.append(Detection(
                x=float((ox1 + ox2) / 2),
                y=float((oy1 + oy2) / 2),
                w=float(ox2 - ox1),
                h=float(oy2 - oy1),
                confidence=float(scores[i]),
                class_id=cid,
                class_name=cfg.class_names[cid] if cid < len(cfg.class_names) else "unknown",
            ))
        return out
