#!/usr/bin/env bash
# pi-fpv-companion WiFi self-heal: re-associate wlan0 if it loses its link.
# Installed because a reboot once left this Pi off WiFi for a long time
# (gateway reachable from the LAN, Pi absent). Runs ~45s after boot and every
# 2 min via wifi-selfheal.timer. Idempotent and quiet when healthy.
set -u

IFACE="wlan0"
CONN="netplan-wlan0-guestnet"
STATE="/run/wifi-selfheal.fails"

healthy() {
    # Must have a global IPv4 on wlan0 AND be able to reach the default gateway.
    ip -4 addr show "$IFACE" 2>/dev/null | grep -q "inet " || return 1
    local gw
    gw="$(ip route 2>/dev/null | awk '/^default/ {print $3; exit}')"
    [ -n "$gw" ] || return 1
    ping -c1 -W2 "$gw" >/dev/null 2>&1
}

if healthy; then
    echo 0 > "$STATE"
    exit 0
fi

fails=$(( $(cat "$STATE" 2>/dev/null || echo 0) + 1 ))
echo "$fails" > "$STATE"
logger -t wifi-selfheal "wlan0 unhealthy (fail #$fails) — re-associating"

# Make sure the radio is on, then try to bring the known connection up.
nmcli radio wifi on 2>/dev/null || true
if ! nmcli connection up "$CONN" 2>/dev/null; then
    nmcli device disconnect "$IFACE" 2>/dev/null || true
    sleep 2
    nmcli device connect "$IFACE" 2>/dev/null || true
fi

# Escalate: after 3 consecutive failures, bounce NetworkManager once.
if [ "$fails" -ge 3 ]; then
    logger -t wifi-selfheal "3 consecutive failures — restarting NetworkManager"
    systemctl restart NetworkManager 2>/dev/null || true
    echo 0 > "$STATE"
fi
