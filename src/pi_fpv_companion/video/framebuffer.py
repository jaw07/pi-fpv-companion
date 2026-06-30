"""Framebuffer sink for the analog CVBS output path.

On Pi, `LinuxFramebuffer` mmaps `/dev/fb0` and writes pixel data into it.
The framebuffer's contents are scanned out to the composite TV pad by the
VideoCore, so writing here is what shows up on the analog VTX → goggles.

On Mac (and other dev hosts), `MockFramebuffer` keeps the bytes in a numpy
canvas — useful for tests and for round-tripping the BGR → RGB565 conversion
without a real framebuffer device.

Both expose the same `Framebuffer` interface so the pipeline doesn't care which.

`FramebufferSink` is the Pipeline-friendly status callback: it composites the
overlay (bbox + HUD) onto the captured frame and writes the result.

Format conventions:
  - input to `write()` is always BGR uint8 (HxWx3)
  - output to the device is whatever the device reports (RGB565 / BGRA8888)
  - sizes must match exactly; resize/letterbox upstream if needed
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from pi_fpv_companion.camera.base import FrameBundle
from pi_fpv_companion.guidance.safety import GateResult
from pi_fpv_companion.types import FilteredTarget, GuidanceIntent, SwitchState
from pi_fpv_companion.video.overlay import draw_overlay


class Framebuffer(Protocol):
    width: int
    height: int
    bpp: int

    def open(self) -> None: ...
    def close(self) -> None: ...
    def write(self, bgr: np.ndarray) -> None: ...


# -------------------- format conversions --------------------


def bgr_to_rgb565(bgr: np.ndarray) -> np.ndarray:
    """Pack BGR uint8 (HxWx3) into RGB565 little-endian uint16 (HxW).

    OpenCV's COLOR_BGR2BGR565 is a single C call that produces BIT-IDENTICAL output
    to the hand-rolled numpy packing below (R<<11 | G<<5 | B, little-endian) but ~10x
    faster (22ms -> 2ms on the Pi Zero 2W for 720x576) — and this conversion was the
    single hottest function in the render path (py-spy). The numpy fallback covers any
    build without the exact cvtColor code so behaviour is unchanged everywhere."""
    try:
        import cv2
        packed = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGR565)   # HxWx2 uint8
        return packed.view(np.uint16).reshape(bgr.shape[0], bgr.shape[1])
    except Exception:
        r = bgr[:, :, 2].astype(np.uint16)
        g = bgr[:, :, 1].astype(np.uint16)
        b = bgr[:, :, 0].astype(np.uint16)
        return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)


def bgr_to_bgra(bgr: np.ndarray) -> np.ndarray:
    """Pack BGR uint8 (HxWx3) into BGRA uint8 (HxWx4) with alpha=255."""
    h, w = bgr.shape[:2]
    out = np.empty((h, w, 4), dtype=np.uint8)
    out[:, :, :3] = bgr
    out[:, :, 3] = 255
    return out


# -------------------- backends --------------------


class LinuxFramebuffer:
    """`/dev/fb0` mmap writer. Reads format from `/sys/class/graphics/fb0/`."""

    def __init__(self, device: str = "/dev/fb0", sysfs_dir: str = "/sys/class/graphics/fb0") -> None:
        self._device = device
        self._sysfs = Path(sysfs_dir)
        self._fd: Optional[int] = None
        self._mmap = None
        self.width = 0
        self.height = 0
        self.bpp = 0
        self._stride = 0

    def open(self) -> None:
        import mmap as _mmap
        size_txt = (self._sysfs / "virtual_size").read_text().strip()
        w, h = size_txt.split(",")
        self.width = int(w)
        self.height = int(h)
        self.bpp = int((self._sysfs / "bits_per_pixel").read_text().strip())
        self._stride = int((self._sysfs / "stride").read_text().strip())
        total = self._stride * self.height
        self._fd = os.open(self._device, os.O_RDWR)
        self._mmap = _mmap.mmap(self._fd, total, _mmap.MAP_SHARED, _mmap.PROT_READ | _mmap.PROT_WRITE)

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def write(self, bgr: np.ndarray) -> None:
        h, w = bgr.shape[:2]
        if w != self.width or h != self.height:
            raise ValueError(
                f"frame {w}x{h} does not match framebuffer {self.width}x{self.height}; "
                f"resize before write()"
            )
        if self.bpp == 16:
            packed = bgr_to_rgb565(bgr)
            buf = packed.tobytes()
        elif self.bpp == 32:
            packed = bgr_to_bgra(bgr)
            buf = packed.tobytes()
        else:
            raise ValueError(f"unsupported framebuffer bpp: {self.bpp}")
        if len(buf) != self._stride * self.height:
            # If row stride != width * bpp/8 we'd need per-row copies; warn instead of silently truncating.
            raise ValueError(
                f"packed size {len(buf)} != stride*height {self._stride * self.height}; "
                f"row padding not handled"
            )
        self._mmap.seek(0)
        self._mmap.write(buf)


class MockFramebuffer:
    """In-memory stand-in for `LinuxFramebuffer`. The Mac doesn't have `/dev/fb0`;
    this captures bytes into a numpy canvas + optionally dumps each frame to disk."""

    def __init__(
        self,
        width: int = 720,
        height: int = 576,
        bpp: int = 16,
        dump_path: Optional[str] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.bpp = bpp
        self._dump_path = dump_path
        self.last_bgr: Optional[np.ndarray] = None
        self.last_packed: Optional[np.ndarray] = None
        self.frame_count: int = 0

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def write(self, bgr: np.ndarray) -> None:
        h, w = bgr.shape[:2]
        if w != self.width or h != self.height:
            raise ValueError(f"frame {w}x{h} != fb {self.width}x{self.height}")
        if self.bpp == 16:
            self.last_packed = bgr_to_rgb565(bgr)
        elif self.bpp == 32:
            self.last_packed = bgr_to_bgra(bgr)
        else:
            raise ValueError(f"unsupported bpp: {self.bpp}")
        self.last_bgr = bgr.copy()
        self.frame_count += 1
        if self._dump_path is not None:
            import cv2
            cv2.imwrite(self._dump_path, bgr)


# -------------------- pipeline-callback adapter --------------------


class FramebufferSink:
    """Adapts a `Framebuffer` to the Pipeline's status-callback signature.
    Renders overlay then pushes the composite to the framebuffer."""

    def __init__(self, fb: Framebuffer) -> None:
        self._fb = fb
        self._opened = False

    def open(self) -> None:
        if not self._opened:
            self._fb.open()
            self._opened = True

    def close(self) -> None:
        if self._opened:
            self._fb.close()
            self._opened = False

    def show(
        self,
        target: Optional[FilteredTarget],
        intent: GuidanceIntent,
        gated: GateResult,
        switch: SwitchState,
        armed: bool,
        frame: FrameBundle,
        tracks=None,
    ) -> None:
        if not self._opened:
            self.open()
        img = frame.image.copy()
        draw_overlay(img, target, intent, switch, armed, gated, tracks)
        self._fb.write(img)
