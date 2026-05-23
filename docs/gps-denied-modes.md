# Which flight mode works with NO GPS (ever)?

> **Decision (2026-05-23): STABILIZE + RC_CHANNELS_OVERRIDE** (default
> `control_mode`). GUIDED_NOGPS / SET_ATTITUDE_TARGET and the auto_guided
> mode-switch were retired. We first moved to ALT_HOLD, then switched to STABILIZE
> because ALT_HOLD's altitude controller fundamentally limits the dive. ALT_HOLD
> remains selectable (`control_mode: althold`) for gentle/altitude-safe ops.
>
> **Dive comparison — SITL ground truth (ArduCopter 4.6.3, `measure_dive_sitl.py`,
> GUIDED takeoff to 150 m then dive at 30° lean + full throttle-down):**
>
> | control_mode | descent rate | dive path |
> |---|---|---|
> | ALT_HOLD (default `PILOT_SPEED_DN`) | 1.3 m/s | 25° |
> | ALT_HOLD (`PILOT_SPEED_DN`=10 m/s) | 4.8 m/s | 43° |
> | **STABILIZE** | **15.9 m/s** | **77° (near-vertical)** |
>
> STABILIZE dives 3–12× faster on a near-vertical path. Tradeoff: STABILIZE has
> no baro altitude hold, so the companion owns altitude (no auto floor) and
> `stab_hover_throttle_us` must be tuned so TRACK ~holds. For a diving interceptor
> that is the right trade. Also validated on 4.6: `validate_sitl.py` 9/9 (RC-override
> attitude sense, signs correct) and `probe_nogps_modes.py` 10/10 (ALT_HOLD +
> STABILIZE enter/arm/steer with GPS disabled).

Assumption: a bare analog FPV quad that **never** has GPS (no GPS, no optical
flow, no VIO). Question: which ArduCopter mode can the companion command?

**SITL ground truth (`scripts/probe_nogps_modes.py`, ArduCopter 4.0.3, 2026-05-23).**
GPS was fully disabled and the autopilot rebooted cold so it came up with no GPS
at all (`SIM_GPS_DISABLE=1`, `GPS_TYPE=0`, `EK2_GPS_TYPE=3`; confirmed
`GPS_RAW_INT fix_type=0, sats=0`). Each mode was entered, armed, and steered:

| mode          | enter | arm | steer (yaw responds) | how it's commanded |
|---------------|-------|-----|----------------------|--------------------|
| GUIDED_NOGPS  | yes   | yes | yes (+22 dps)        | `SET_ATTITUDE_TARGET` |
| ALT_HOLD      | yes   | yes | yes (+59 dps)        | `RC_CHANNELS_OVERRIDE` |
| STABILIZE     | yes   | yes | yes (+58 dps)        | `RC_CHANNELS_OVERRIDE` |

## Key correction: GUIDED_NOGPS does NOT need GPS

An earlier note worried "a real GPS-denied airframe may never get an EKF origin,
so GUIDED_NOGPS may never be enterable." **That was wrong** — it conflated two
cases:

- **GPS present but unlocked** (what the *other* SITL scripts run, with the sim
  GPS on): the EKF waits for a GPS origin before allowing GUIDED_NOGPS, so right
  after boot the mode is briefly rejected until "EKF origin set".
- **No GPS at all** (this probe, and the real airframe): there is nothing to
  wait for. The EKF aligns from the IMU in a few seconds and GUIDED_NOGPS is
  available immediately. `mode_guided_nogps` is an angle-control submode with
  `requires_position()=false` (see `ardupilot-vertical-control-research.md` §2).

So **GUIDED_NOGPS is a valid no-GPS mode** and the current design stands. On a
real no-GPS build, configure the EKF source to not expect GPS
(`EK3_SRC1_POSXY=0`, `EK3_SRC1_VELXY=0`, `EK3_SRC1_POSZ=Baro` — or the EK2
equivalents) so the EKF never waits for a GPS that will never come.

## The chosen path: STABILIZE via RC override (ALT_HOLD selectable)

The backend injects AETR sticks via `RC_CHANNELS_OVERRIDE`; `control_mode` picks
the throttle semantics and must match the FC's flight mode:

- **STABILIZE (default)** — direct throttle, no FC altitude hold. Full-down really
  cuts power → a true steep dive (15.9 m/s, 77° in SITL). TRACK altitude is held by
  the companion's **adaptive hover**: a vertical-velocity PI loop on `VFR_HUD.climb`
  (`_adaptive_throttle`) that learns the hover throttle and damps climb — so you set
  only a rough `stab_hover_throttle_us` seed and it self-corrects. SITL-proven: from
  a deliberately-low seed (1300) it learned 1474 and drove climb to 0 in ~6 s
  (`scripts/learn_hover_sitl.py`). Frozen during a commanded dive; needs `VFR_HUD`
  streamed (falls back to the fixed seed otherwise). No altitude *floor* by design.
- **ALT_HOLD** (`control_mode: althold`) — throttle is a climb *rate*, 0.5 holds
  altitude via baro, descent capped at `PILOT_SPEED_DN`. Gentler and altitude-safe
  but cannot dive aggressively (1.3–4.8 m/s).

Both are GPS-free (baro + IMU only; no EKF origin) and both are pilot modes, so
handback is automatic: when the companion stops overriding (`release()`),
ArduPilot reverts the channels to the real RC radio after its override timeout —
no GUIDED "ignore pilot / hover on timeout" lockout, and a dead Pi fail-safes to
the radio.

### Why STABILIZE over ALT_HOLD

For a **diving interceptor**, dive performance is the mission. ALT_HOLD's altitude
controller actively opposes altitude loss and rate-limits descent — fundamentally
wrong for a committed dive. STABILIZE gives the pilot/companion direct throttle, so
nosing down + cutting power is a real dive (see table at top: 3–12× the descent
rate). The cost — no baro altitude floor — is acceptable for a craft whose job is
to descend onto a target, and altitude-floor/closed-loop-vertical work is tracked
separately (`dynamic-vertical-control`). Use `control_mode: althold` if you want
the altitude-safe behaviour instead.

The mapping is `intent_to_rc_overrides()` / `_throttle_pwm()` in `fc/ardupilot.py`
(AETR PWM, like the Betaflight mapping but over MAVLink RC override). Stick signs
default to SITL-validated (+1,+1,+1); `ArduCopterRcMapping` exposes per-axis flips
and `stab_hover_throttle_us` for per-airframe hover tuning.

## Reproduce

```sh
# 4.6.3 (flight-era firmware); see docker/sitl-4.6/Dockerfile
docker build --platform linux/arm64 -t pifpv-sitl:4.6 docker/sitl-4.6
docker run -d --rm --name pifpv-sitl -p 127.0.0.1:5760:5760 pifpv-sitl:4.6
.venv/bin/python scripts/probe_nogps_modes.py  --connect tcp:127.0.0.1:5760  # GPS-off modes
.venv/bin/python scripts/measure_dive_sitl.py  --connect tcp:127.0.0.1:5760  # dive comparison
.venv/bin/python scripts/validate_sitl.py      --connect tcp:127.0.0.1:5760  # control sense
```
