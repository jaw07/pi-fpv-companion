"""NanoDet-Plus object detector via NCNN.

DEV / SIM PATH — NOT a flight detector (architecture-audit.md §3). On the
Zero 2W this runs ~4 Hz: a slideshow, not a tracker. The flight detector is
the IMX500 (on-sensor inference, ~30 FPS, ~0 host CPU). Keep this for
no-hardware development and for higher-RAM SBCs (Pi 4/CM4); `main.py` warns
at startup when it is selected.

Pareto-optimal pick for Pi Zero 2W: 30.4 mAP on COCO, 1.17M params, ~35 MB
activation memory at 320x320 input, ~280-320 ms inference on A53 1 GHz.
Half the activation footprint of YOLOv8n at similar accuracy — important for
fitting the 200 MB Pi budget alongside the rest of the pipeline.

Output decoding is GFL-style (Generalized Focal Loss): each anchor predicts a
discrete distribution over 8 bins per box side, plus per-class scores.

Reference pattern: drone-guidance/src/core/detector.py (same architecture,
adapted to this project's Detector Protocol).

`ncnn` is imported lazily — module is importable on machines without it.
"""
from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from pi_fpv_companion.types import Detection


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


@dataclass(frozen=True)
class NanoDetConfig:
    """All knobs for NanoDet-Plus inference.

    `model_dir` must contain `model.ncnn.param` + `model.ncnn.bin`. The model's
    own input size, num_classes, reg_max, strides, mean/std should match what
    was used at training time — these defaults match the public NanoDet-Plus-M
    COCO weights.
    """
    model_dir: Path
    input_size: int = 320
    num_classes: int = 80
    reg_max: int = 7                       # 8 distribution bins (0..7) per box side
    strides: Tuple[int, ...] = (8, 16, 32, 64)
    mean: Tuple[float, float, float] = (103.53, 116.28, 123.675)
    std: Tuple[float, float, float] = (57.375, 57.12, 58.395)
    conf_threshold: float = 0.35
    nms_threshold: float = 0.45
    target_class_ids: Tuple[int, ...] = ()
    num_threads: int = 2                   # Zero 2W has 4 A53 cores; 2 leaves room for the rest
    class_names: Tuple[str, ...] = tuple(COCO_CLASSES)


class NanoDetDetector:
    def __init__(self, cfg: NanoDetConfig) -> None:
        self._cfg = cfg
        self._net = None
        self._anchors = self._generate_anchors(cfg.input_size, cfg.strides)
        self._inv_std = tuple(1.0 / s for s in cfg.std)
        self._letterbox = np.full((cfg.input_size, cfg.input_size, 3), 114, dtype=np.uint8)
        self._target_class_set = set(cfg.target_class_ids) if cfg.target_class_ids else None

    def open(self) -> None:
        import ncnn  # lazy
        param = self._cfg.model_dir / "model.ncnn.param"
        binf = self._cfg.model_dir / "model.ncnn.bin"
        if not param.exists() or not binf.exists():
            raise RuntimeError(
                f"NanoDet model files missing under {self._cfg.model_dir}: "
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

    @staticmethod
    def _generate_anchors(input_size: int, strides: Tuple[int, ...]) -> np.ndarray:
        all_anchors = []
        for stride in strides:
            fh = math.ceil(input_size / stride)
            fw = math.ceil(input_size / stride)
            xs = np.arange(fw, dtype=np.float32) * stride
            ys = np.arange(fh, dtype=np.float32) * stride
            xx, yy = np.meshgrid(xs, ys)
            anchors = np.stack([xx.ravel(), yy.ravel()], axis=-1) + stride / 2.0
            all_anchors.append(anchors)
        return np.concatenate(all_anchors, axis=0)

    def _stride_per_anchor(self) -> np.ndarray:
        out = []
        for stride in self._cfg.strides:
            fh = math.ceil(self._cfg.input_size / stride)
            fw = math.ceil(self._cfg.input_size / stride)
            out.append(np.full(fh * fw, stride, dtype=np.float32))
        return np.concatenate(out)

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

        mat = ncnn.Mat.from_pixels(
            letterboxed,
            ncnn.Mat.PixelType.PIXEL_BGR,
            self._cfg.input_size,
            self._cfg.input_size,
        )
        mat.substract_mean_normalize(list(self._cfg.mean), list(self._inv_std))

        ex = self._net.create_extractor()
        ex.input("data", mat)
        _, output = ex.extract("output")
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
        n_per_anchor = cfg.num_classes + 4 * (cfg.reg_max + 1)
        flat = output.reshape(-1, n_per_anchor)

        class_scores = flat[:, : cfg.num_classes]
        bbox_dist = flat[:, cfg.num_classes :]

        max_scores = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1)
        keep = max_scores > cfg.conf_threshold
        if self._target_class_set is not None:
            keep &= np.isin(class_ids, list(self._target_class_set))
        if not keep.any():
            return []

        scores = max_scores[keep]
        class_ids = class_ids[keep]
        bbox_dist = bbox_dist[keep]
        anchors = self._anchors[keep]
        strides = self._stride_per_anchor()[keep]

        # GFL: softmax over the 8 distribution bins per side, take expected value
        bins = cfg.reg_max + 1
        bbox_dist = bbox_dist.reshape(-1, 4, bins)
        e = np.exp(bbox_dist - bbox_dist.max(axis=2, keepdims=True))
        probs = e / e.sum(axis=2, keepdims=True)
        proj = np.arange(bins, dtype=np.float32)
        distances = (probs * proj).sum(axis=2) * strides[:, None]

        x1 = anchors[:, 0] - distances[:, 0]
        y1 = anchors[:, 1] - distances[:, 1]
        x2 = anchors[:, 0] + distances[:, 2]
        y2 = anchors[:, 1] + distances[:, 3]
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        indices = cv2.dnn.NMSBoxes(
            boxes.astype(np.float32).tolist(),
            scores.astype(np.float32).tolist(),
            cfg.conf_threshold,
            cfg.nms_threshold,
        )
        if len(indices) == 0:
            return []
        indices = indices.flatten() if hasattr(indices, "flatten") else indices

        out: List[Detection] = []
        for idx in indices:
            bx1, by1, bx2, by2 = boxes[idx]
            ox1 = max(0.0, min(orig_w, (bx1 - pad_w) / scale))
            oy1 = max(0.0, min(orig_h, (by1 - pad_h) / scale))
            ox2 = max(0.0, min(orig_w, (bx2 - pad_w) / scale))
            oy2 = max(0.0, min(orig_h, (by2 - pad_h) / scale))
            if ox2 <= ox1 or oy2 <= oy1:
                continue
            cid = int(class_ids[idx])
            out.append(Detection(
                x=float((ox1 + ox2) / 2),
                y=float((oy1 + oy2) / 2),
                w=float(ox2 - ox1),
                h=float(oy2 - oy1),
                confidence=float(scores[idx]),
                class_id=cid,
                class_name=cfg.class_names[cid] if cid < len(cfg.class_names) else "unknown",
            ))
        return out
