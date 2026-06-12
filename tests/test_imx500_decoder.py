"""IMX500 DecoderProfile + tensor decode tests (no hardware: a fake get_outputs)."""
from __future__ import annotations
import numpy as np

from pi_fpv_companion.camera.imx500 import IMX500Camera, DecoderProfile


def test_profile_selection_by_filename():
    yolo = DecoderProfile.for_model("/x/imx500_network_yolo11n_pp.rpk")
    assert yolo.box_order == "xyxy" and yolo.box_scale == "input_px" and yolo.input_size == 640
    assert yolo.count_is_real and yolo.labels[0] == "person"
    yolo26 = DecoderProfile.for_model("/x/imx500_network_yolo26n_pp.rpk")
    assert yolo26 == yolo, "YOLO26n shares the YOLO tensor profile exactly"
    ssd = DecoderProfile.for_model("/x/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk")
    assert ssd.box_order == "yxyx" and ssd.box_scale == "normalized" and not ssd.count_is_real


class _FakeIMX500:
    def __init__(self, outputs): self._o = outputs
    def get_outputs(self, metadata, add_batch=True): return self._o


def _cam(model, conf=0.3, classes=()):
    c = IMX500Camera(model_path=model, width=720, height=576,
                     conf_threshold=conf, target_class_ids=classes)
    return c


def test_yolo_xyxy_pixel_decode_maps_to_frame():
    # YOLO: box [x1,y1,x2,y2] in 640-pixel space -> centre/size in the 720x576 frame.
    cam = _cam("imx500_network_yolo11n_pp.rpk")
    cam._imx500 = _FakeIMX500([
        np.array([[[64.0, 128.0, 192.0, 256.0]]]),   # 1 box, 640-space
        np.array([[0.9]]), np.array([[0.0]]), np.array([[1.0]]),
    ])
    dets = cam._decode_detections({}, 720, 576)
    assert len(dets) == 1
    d = dets[0]
    # x1=64/640*720=72, x2=192/640*720=216 -> cx=144, w=144 ; y1=128/640*576=115.2 ...
    assert abs(d.x - 144.0) < 0.5 and abs(d.w - 144.0) < 0.5
    assert abs(d.y - 172.8) < 0.5 and abs(d.h - 115.2) < 0.5
    assert d.class_name == "person"


def test_yolo_decode_uses_actual_input_size_not_hardcoded():
    # A 416-input YOLO emits boxes in 0..416 px. The decoder must divide by the
    # sensor's REAL input size (set in open()), not the 640 profile default —
    # otherwise boxes collapse to ~65% of frame, top-left (the @416 phantom-box bug).
    cam = _cam("imx500_network_yolo11n_416_pp.rpk")
    cam._input_size = (416, 416)                         # what open() reads from the sensor
    cam._imx500 = _FakeIMX500([
        np.array([[[0.0, 0.0, 416.0, 416.0]]]),         # full-frame box in 416-space
        np.array([[0.9]]), np.array([[0.0]]), np.array([[1.0]]),
    ])
    d = cam._decode_detections({}, 720, 576)[0]
    assert abs(d.w - 720.0) < 0.5 and abs(d.h - 576.0) < 0.5   # full frame, not 65%
    assert abs(d.x - 360.0) < 0.5 and abs(d.y - 288.0) < 0.5


def test_yolo_real_count_truncates_candidate_tail():
    # YOLO [3]=count is REAL: a high-score candidate BEYOND the count must be ignored.
    cam = _cam("imx500_network_yolo11n_pp.rpk")
    cam._imx500 = _FakeIMX500([
        np.array([[[10.0, 10.0, 50.0, 50.0], [20.0, 20.0, 60.0, 60.0]]]),
        np.array([[0.9, 0.95]]), np.array([[0.0, 0.0]]), np.array([[1.0]]),   # count=1
    ])
    assert len(cam._decode_detections({}, 720, 576)) == 1


def test_ssd_yxyx_normalized_decode_unchanged():
    # SSD: box [ymin,xmin,ymax,xmax] normalized 0..1, fixed-count cap -> scan all.
    cam = _cam("imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk")
    cam._imx500 = _FakeIMX500([
        np.array([[[0.25, 0.1, 0.75, 0.5]]]),
        np.array([[0.8]]), np.array([[0.0]]), np.array([[100.0]]),   # fixed 100 cap
    ])
    dets = cam._decode_detections({}, 720, 576)
    assert len(dets) == 1
    d = dets[0]
    # xmin=0.1*720=72, xmax=0.5*720=360 -> cx=216, w=288
    assert abs(d.x - 216.0) < 0.5 and abs(d.w - 288.0) < 0.5
