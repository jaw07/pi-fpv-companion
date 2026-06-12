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

# NEVER try to heal a network that isn't there. At the flying field the home AP is
# out of range for the whole session; without this gate the script force-fails an
# activation every 2 min and restarts NetworkManager every ~12 min — guaranteed, every
# flight — which is exactly the NM churn the header warns about and the prime suspect
# for the Pi refusing WiFi after fly days. Heal ONLY when an SSID that NetworkManager
# has a saved wifi profile for is actually visible in a scan.
known_ssid_in_range() {
    local profiles visible
    profiles="$(nmcli -t -f NAME,TYPE connection show 2>/dev/null \
                | awk -F: '$2 ~ /wireless/ {print $1}')"
    [ -n "$profiles" ] || return 1
    visible="$(nmcli -t -f SSID device wifi list --rescan auto 2>/dev/null | sort -u)"
    [ -n "$visible" ] || return 1
    while IFS= read -r prof; do
        local ssid
        ssid="$(nmcli -t -f 802-11-wireless.ssid connection show "$prof" 2>/dev/null \
                | cut -d: -f2-)"
        [ -n "$ssid" ] && grep -qxF "$ssid" <<< "$visible" && return 0
    done <<< "$profiles"
    return 1
}

# Radio on FIRST (cheap, idempotent): with the radio soft-blocked the scan below is
# empty and the in-range gate would wrongly conclude "nothing to heal".
nmcli radio wifi on 2>/dev/null || true

if ! known_ssid_in_range; then
    # Out of range (field) or radio scan empty: nothing to heal — stay quiet, keep the
    # failure counter at 0 so escalation only ever counts IN-RANGE failures.
    echo 0 > "$STATE" 2>/dev/null || true
    logger -t wifi-selfheal "wlan0 down but no known SSID in range — not healing"
    exit 0
fi

fails=$(( $(cat "$STATE" 2>/dev/null || echo 0) + 1 ))
echo "$fails" > "$STATE" 2>/dev/null || true
logger -t wifi-selfheal "wlan0 not connected (fail #$fails) — re-associating"

# Bring the device back up using whatever autoconnect profile is bound to it — no
# hardcoded connection name (that broke across reflashes). nmcli picks the profile.
nmcli device connect "$IFACE" 2>/dev/null || true

# Restart NetworkManager ONLY after many consecutive genuine failures (~12+ min truly
# disconnected). It briefly drops all connectivity, so it is a true last resort, never
# a routine action — this is what the old version did far too eagerly.
if [ "$fails" -ge 6 ]; then
    logger -t wifi-selfheal "6 consecutive failures — restarting NetworkManager (last resort)"
    systemctl restart NetworkManager 2>/dev/null || true
    echo 0 > "$STATE" 2>/dev/null || true
fi
