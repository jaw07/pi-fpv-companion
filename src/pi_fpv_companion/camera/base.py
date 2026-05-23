"""Camera abstraction.

Two backends will conform to this Protocol:

  PiCamCamera   — standard CSI Pi Camera. Yields frames only; detections come
                  from a separate on-Pi NCNN detector.
  IMX500Camera  — Sony AI Camera. Yields frames AND on-sensor detections in
                  the same FrameBundle (detector module is a no-op).

The pipeline downstream of the camera doesn't care which path is in use.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterator, List, Protocol

from pi_fpv_companion.types import Detection


@dataclass(frozen=True)
class FrameBundle:
    """One captured frame plus any detections the camera already produced.

    `image` is a numpy array (typed loosely to avoid a numpy import here);
    callers that need it as ndarray import numpy themselves.
    """
    image: object                                   # numpy ndarray, HxWx3 uint8 BGR
    width: int
    height: int
    timestamp: float                                # monotonic seconds
    detections: List[Detection] = field(default_factory=list)


class Camera(Protocol):
    """Frame-and-maybe-detections source."""

    def open(self) -> None: ...
    def close(self) -> None: ...

    def frames(self) -> Iterator[FrameBundle]:
        """Yield bundles indefinitely. Generator semantics; close ends iteration."""
        ...
