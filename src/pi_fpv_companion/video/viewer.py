"""cv2.imshow-based live viewer.

Acts as a Pipeline status callback. On the Pi we'd substitute a framebuffer
writer (same overlay code, different sink) — but on the Mac this gives us a
fast visual feedback loop without involving any Pi-specific output path.
"""
from __future__ import annotations
from typing import Optional

import cv2

from pi_fpv_companion.camera.base import FrameBundle
from pi_fpv_companion.guidance.safety import GateResult
from pi_fpv_companion.types import FilteredTarget, GuidanceIntent, SwitchState
from pi_fpv_companion.video.overlay import draw_overlay


class LiveViewer:
    def __init__(self, window_name: str = "pi-fpv-companion", scale: float = 1.0) -> None:
        self._window = window_name
        self._scale = scale
        self._opened = False

    def open(self) -> None:
        cv2.namedWindow(self._window, cv2.WINDOW_AUTOSIZE)
        self._opened = True

    def close(self) -> None:
        if self._opened:
            cv2.destroyWindow(self._window)
            self._opened = False

    def show(
        self,
        target: Optional[FilteredTarget],
        intent: GuidanceIntent,
        gated: GateResult,
        switch: SwitchState,
        armed: bool,
        frame: FrameBundle,
    ) -> None:
        if not self._opened:
            self.open()
        img = frame.image.copy()
        draw_overlay(img, target, intent, switch, armed, gated)
        if self._scale != 1.0:
            img = cv2.resize(img, None, fx=self._scale, fy=self._scale)
        cv2.imshow(self._window, img)
        cv2.waitKey(1)
