#!/usr/bin/env bash
# WiFi keepalive — ensure the radio is unblocked, and NOTHING that can fight NetworkManager.
#
# NetworkManager owns reconnection. The wlan0 profile is autoconnect=yes with
# autoconnect-retries=-1 (infinite), so NM re-associates ON ITS OWN whenever the AP
# returns to range — including after a whole flight out of range. So this script does
# NOT force scans, does NOT `nmcli device connect`, and NEVER restarts NetworkManager.
#
# Why (flight-3 finding, 2026-07-02): the previous escalation did all three, and back
# home (known SSID in range = the reconnect window) it FOUGHT NM's own autoconnect —
# manual `device connect` racing autoconnect, and a NetworkManager restart every ~12 min
# that killed near-complete associations in a loop. Result: the Pi "refused to reconnect
# after a flight" and only a REFLASH fixed it (a reboot just re-armed the same churn).
# The only thing that can silently stop NM from reconnecting is a soft-blocked radio, so
# that is the ONLY thing we touch here. Both actions are idempotent and non-disruptive.
set -u

nmcli radio wifi on 2>/dev/null || true
rfkill unblock wifi 2>/dev/null || true
