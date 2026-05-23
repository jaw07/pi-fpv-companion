"""FlightController abstraction. Both ArduPilot and Betaflight backends conform.

The visual servo emits backend-agnostic intent; each backend translates to its
native wire protocol:

  ArduPilot  -> RC_CHANNELS_OVERRIDE AETR sticks in ALT_HOLD (GPS-denied)
  Betaflight -> MSP_SET_RAW_RC with intent-to-stick mapping (ANGLE mode)
"""
from __future__ import annotations
from typing import Protocol

from pi_fpv_companion.types import GuidanceIntent, SwitchState


class FlightController(Protocol):
    """Backend-agnostic FC interface used by the main loop."""

    def open(self) -> None: ...
    def close(self) -> None: ...

    def read_switch(self) -> SwitchState:
        """Return the latest cached state of the configured arming RC channel."""
        ...

    def is_armed(self) -> bool:
        """True if the FC reports itself armed (HEARTBEAT for AP, MSP_STATUS for BF)."""
        ...

    def send_intent(self, intent: GuidanceIntent) -> None:
        """Translate intent to the native protocol and send one command frame.
        Called by the pipeline every tick while engaged (TRACK/DIVE)."""
        ...

    def release(self) -> None:
        """Hand control back to the pilot. Called by the pipeline in STANDBY.
        ArduPilot clears its RC override (channels revert to the receiver);
        Betaflight sends neutral sticks (best effort — true handback needs the
        RX, demo path)."""
        ...
