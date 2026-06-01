"""ArUco-marker detector — conforms to the Detector Protocol.

For SITL / Webots bring-up: the ArduPilot Webots worlds place ArUco markers
(DICT_4X4_50) as targets, and a fiducial gives a rock-solid, unambiguous
detection so the camera→track→servo→FC loop can be exercised against a real
rendered camera feed without depending on a trained model.

Dev/sim only (the flight camera is the IMX500, on-sensor) — same Protocol. Each
marker becomes one Detection (centre + corner bounding box; class_id = marker id).
"""
from __future__ import annotations
from typing import List

import cv2
import numpy as np

from pi_fpv_companion.types import Detection


class ArucoDetector:
    def __init__(self, dictionary: int = cv2.aruco.DICT_4X4_50,
                 only_id: int | None = None) -> None:
        """`only_id` restricts detection to a single marker id (the rest are
        ignored), which keeps the tracker from flipping between fiducials."""
        self._only_id = only_id
        # OpenCV 4.7+ uses the ArucoDetector object API; older builds use the
        # free Dictionary_get / detectMarkers functions. Support both.
        get_dict = getattr(cv2.aruco, "getPredefinedDictionary", None) \
            or cv2.aruco.Dictionary_get
        self._dict = get_dict(dictionary)
        if hasattr(cv2.aruco, "ArucoDetector"):
            params = cv2.aruco.DetectorParameters()
            self._detector = cv2.aruco.ArucoDetector(self._dict, params)
        else:                                    # pragma: no cover - legacy cv2
            self._detector = None
            self._params = cv2.aruco.DetectorParameters_create()

    def detect(self, image: object) -> List[Detection]:
        img = image if isinstance(image, np.ndarray) else np.asarray(image)
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:                                    # pragma: no cover - legacy cv2
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self._dict, parameters=self._params)
        if ids is None:
            return []
        out: List[Detection] = []
        for marker, mid in zip(corners, ids.flatten()):
            if self._only_id is not None and int(mid) != self._only_id:
                continue
            pts = marker.reshape(4, 2)
            x0, y0 = pts.min(axis=0)
            x1, y1 = pts.max(axis=0)
            out.append(Detection(
                x=float((x0 + x1) / 2.0), y=float((y0 + y1) / 2.0),
                w=float(x1 - x0), h=float(y1 - y0),
                confidence=1.0, class_id=int(mid), class_name=f"aruco{int(mid)}",
            ))
        return out
