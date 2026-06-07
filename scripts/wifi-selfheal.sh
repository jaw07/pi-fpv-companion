#!/usr/bin/env bash
# WiFi self-heal: re-associate wlan0 only if it has GENUINELY lost its NetworkManager
# connection. Installed because a reboot once left this Pi off WiFi for a long time.
# Run as root by wifi-selfheal.service, fired ~45s after boot + every 2 min by
# wifi-selfheal.timer. Quiet/no-op when WiFi is healthy.
#
# IMPORTANT — the health check uses ONLY local NetworkManager + kernel state. It does
# NOT ping the gateway. An earlier version pinged the gateway, but this (guest) network
# blocks ICMP, so it always read "unhealthy" and churned the link / restarted
# NetworkManager every couple of minutes — which knocked the Pi (and the ZeroTier
# bridge riding on it) offline. DO NOT reintroduce a ping-based check.
set -u

IFACE="wlan0"
STATE="/run/wifi-selfheal.fails"

healthy() {
    # Healthy = NetworkManager reports the device CONNECTED *and* it has a global IPv4.
    local st
    st="$(nmcli -t -f GENERAL.STATE device show "$IFACE" 2>/dev/null | cut -d: -f2)"
    case "$st" in 100*) : ;; *) return 1 ;; esac          # "100 (connected)"
    ip -4 addr show "$IFACE" 2>/dev/null | grep -q "scope global"
}

if healthy; then
    echo 0 > "$STATE" 2>/dev/null || true
    exit 0
fi

fails=$(( $(cat "$STATE" 2>/dev/null || echo 0) + 1 ))
echo "$fails" > "$STATE" 2>/dev/null || true
logger -t wifi-selfheal "wlan0 not connected (fail #$fails) — re-associating"

# Bring the device back up using whatever autoconnect profile is bound to it — no
# hardcoded connection name (that broke across reflashes). nmcli picks the profile.
nmcli radio wifi on 2>/dev/null || true
nmcli device connect "$IFACE" 2>/dev/null || true

# Restart NetworkManager ONLY after many consecutive genuine failures (~12+ min truly
# disconnected). It briefly drops all connectivity, so it is a true last resort, never
# a routine action — this is what the old version did far too eagerly.
if [ "$fails" -ge 6 ]; then
    logger -t wifi-selfheal "6 consecutive failures — restarting NetworkManager (last resort)"
    systemctl restart NetworkManager 2>/dev/null || true
    echo 0 > "$STATE" 2>/dev/null || true
fi
