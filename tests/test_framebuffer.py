"""Framebuffer format conversion + MockFramebuffer + FramebufferSink callback."""
import numpy as np

from pi_fpv_companion.camera.base import FrameBundle
from pi_fpv_companion.guidance.safety import GateResult
from pi_fpv_companion.types import (
    Detection, FilteredTarget, GuidanceIntent, SwitchState, ZERO_INTENT,
)
from pi_fpv_companion.video.framebuffer import (
    FramebufferSink, MockFramebuffer, bgr_to_bgra, bgr_to_rgb565,
)


# ---- format conversion ----

def test_rgb565_pure_red_packs_to_0xf800():
    img = np.zeros((1, 1, 3), dtype=np.uint8)
    img[0, 0] = (0, 0, 255)        # BGR red
    out = bgr_to_rgb565(img)
    assert out[0, 0] == 0xF800     # 11111 000000 00000


def test_rgb565_pure_green_packs_to_0x07e0():
    img = np.zeros((1, 1, 3), dtype=np.uint8)
    img[0, 0] = (0, 255, 0)
    out = bgr_to_rgb565(img)
    assert out[0, 0] == 0x07E0     # 00000 111111 00000


def test_rgb565_pure_blue_packs_to_0x001f():
    img = np.zeros((1, 1, 3), dtype=np.uint8)
    img[0, 0] = (255, 0, 0)
    out = bgr_to_rgb565(img)
    assert out[0, 0] == 0x001F     # 00000 000000 11111


def test_rgb565_black_and_white():
    img = np.zeros((1, 2, 3), dtype=np.uint8)
    img[0, 0] = (0, 0, 0)
    img[0, 1] = (255, 255, 255)
    out = bgr_to_rgb565(img)
    assert out[0, 0] == 0x0000
    assert out[0, 1] == 0xFFFF


def test_rgb565_output_dtype_and_shape():
    img = np.zeros((3, 4, 3), dtype=np.uint8)
    out = bgr_to_rgb565(img)
    assert out.shape == (3, 4)
    assert out.dtype == np.uint16


def test_bgra_appends_alpha_255():
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[:, :] = (10, 20, 30)
    out = bgr_to_bgra(img)
    assert out.shape == (2, 2, 4)
    assert (out[..., :3] == img).all()
    assert (out[..., 3] == 255).all()


# ---- MockFramebuffer ----

def test_mock_records_write_and_packs_rgb565():
    fb = MockFramebuffer(width=4, height=2, bpp=16)
    fb.open()
    img = np.zeros((2, 4, 3), dtype=np.uint8)
    img[:, :] = (0, 0, 255)   # red
    fb.write(img)
    assert fb.frame_count == 1
    assert fb.last_bgr is not None
    assert fb.last_packed is not None
    assert fb.last_packed.dtype == np.uint16
    assert (fb.last_packed == 0xF800).all()
    fb.close()


def test_mock_rejects_size_mismatch():
    fb = MockFramebuffer(width=4, height=2)
    fb.open()
    img = np.zeros((3, 4, 3), dtype=np.uint8)
    try:
        fb.write(img)
    except ValueError as e:
        assert "fb" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_mock_supports_32bpp():
    fb = MockFramebuffer(width=4, height=2, bpp=32)
    fb.open()
    img = np.full((2, 4, 3), 128, dtype=np.uint8)
    fb.write(img)
    assert fb.last_packed.shape == (2, 4, 4)
    assert (fb.last_packed[..., 3] == 255).all()


# ---- FramebufferSink ----

def _inputs():
    img = np.full((576, 720, 3), 32, dtype=np.uint8)
    bundle = FrameBundle(image=img, width=720, height=576, timestamp=0.0, detections=[])
    target = FilteredTarget(
        detection=Detection(x=400, y=300, w=60, h=60, confidence=0.9, class_id=0, class_name="t"),
        track_id=1, vx_px_s=0.0, vy_px_s=0.0, quality=0.9, timestamp=0.0,
    )
    intent = GuidanceIntent(roll_deg=0.0, pitch_deg=-8.0, yaw_rate_dps=5.0, thrust=0.5, timestamp=0.0)
    switch = SwitchState(active=True, pwm_us=1800, timestamp=0.0)
    gated = GateResult(intent=intent, muted=False, reason="")
    return bundle, target, intent, switch, gated


def test_sink_writes_one_frame_per_show():
    fb = MockFramebuffer(width=720, height=576, bpp=16)
    sink = FramebufferSink(fb)
    bundle, target, intent, switch, gated = _inputs()
    sink.show(target, intent, gated, switch, armed=True, frame=bundle)
    sink.show(target, intent, gated, switch, armed=True, frame=bundle)
    assert fb.frame_count == 2
    sink.close()


def test_sink_does_not_mutate_camera_buffer():
    """The pipeline reuses camera frame buffers; sink must copy before drawing."""
    fb = MockFramebuffer(width=720, height=576, bpp=16)
    sink = FramebufferSink(fb)
    bundle, target, intent, switch, gated = _inputs()
    before = bundle.image.copy()
    sink.show(target, intent, gated, switch, armed=True, frame=bundle)
    # bundle.image still pristine — overlay was drawn on a copy
    assert np.array_equal(bundle.image, before)
    # But the framebuffer DID get a frame with overlay drawn
    assert not np.array_equal(fb.last_bgr, before)
    sink.close()
