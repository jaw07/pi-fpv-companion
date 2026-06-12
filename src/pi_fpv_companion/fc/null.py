"""Null FC backend — camera-only bench rigs with NO flight controller attached.

The pipeline runs end-to-end (camera -> detect -> track -> servo preview -> HUD)
with the switch permanently in STANDBY and nothing armed, so no guidance is ever
transmitted anywhere (there is nowhere to transmit to). Use with
`fc: {backend: none}`; pair with --force-mode track/dive to preview engaged
behaviour on the HUD.
"""
from __future__ import annotations
import time

from pi_fpv_companion.types import GuidanceIntent, GuidanceMode, SwitchState


class NullFC:
    """No-op FC: reports STANDBY/disarmed forever, swallows everything else."""

    def open(self) -> None: ...
    def close(self) -> None: ...

    def read_switch(self) -> SwitchState:
        return SwitchState(active=False, pwm_us=0, timestamp=time.monotonic(),
                           mode=GuidanceMode.STANDBY)

    def is_armed(self) -> bool:
        return False

    def release(self) -> None: ...

    def send_intent(self, intent: GuidanceIntent) -> None: ...
