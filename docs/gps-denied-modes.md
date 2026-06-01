# Which flight mode works with NO GPS (ever)?

> **Decision (2026-06-01): GUIDED_NOGPS body-RATES** are the flight path (deploy
> `control_mode: guided_nogps` in `config/imx500.yaml`). The companion commands
> body **rates** + real thrust via `SET_ATTITUDE_TARGET` (`backend.send_body_rates`,
> `guidance/rate_control.py`) while the FC sits in GUIDED_NOGPS. STABILIZE and
> ALT_HOLD + RC_CHANNELS_OVERRIDE remain as **fallbacks** (set `control_mode` to
> match the FC's mode); they are no longer the default.
>
> The earlier "retired GUIDED_NOGPS, run STABILIZE" verdict was driven by the dive
> *planing* — since traced to a missing FC parameter, **not** a mode limitation.
>
> Root cause: ArduCopter reads the `SET_ATTITUDE_TARGET` **thrust** field as a
> *climb-rate* unless `GUID_OPTIONS` **bit 3 (=8, ThrustAsThrust)** is set. Without
> it, "throttle 0" means "hold altitude", so the craft never descends and the dive
> planes. With it, the thrust field is real throttle and the dive comes down. The
> backend now SETS+VERIFIES that bit in the preflight param check
> (`ensure_param_bits`, `GUID_OPTIONS_THRUST_AS_THRUST`; SITL readback=8).
>
> Why a RATE surface (not an absolute attitude quaternion): body **rates** are
> integrated by the airframe, so a noisy detector box yields smooth motion; an
> absolute-attitude quaternion snapped to each frame and jittered. The control law
> (TRACK = range-hold, DIVE = pursuit guidance driving the velocity vector onto the
> line-of-sight) is in `guidance/rate_control.py`.
>
> **Dive comparison — SITL ground truth (ArduCopter 4.6.3, `measure_dive_sitl.py`,
> GUIDED takeoff to 150 m then dive at 30° lean + full throttle-down) for the
> RC-override fallbacks:**
>
> | control_mode | descent rate | dive path |
> |---|---|---|
> | ALT_HOLD (default `PILOT_SPEED_DN`) | 1.3 m/s | 25° |
> | ALT_HOLD (`PILOT_SPEED_DN`=10 m/s) | 4.8 m/s | 43° |
> | **STABILIZE** | **15.9 m/s** | **77° (near-vertical)** |
>
> STABILIZE dives 3–12× faster than ALT_HOLD on a near-vertical path; its tradeoff
> is no baro altitude hold (the companion owns altitude, no auto floor). These are
> the fallback paths; the guided_nogps DIVE instead flies pursuit guidance onto the
> line-of-sight.
>
> **Validation state:** guided_nogps is **SITL + Gazebo (camera-in-the-loop)
> validated** — clean TRACK→DIVE→impact at 25/40/55 m, a moving target, STANDBY
> safe-hold, Pi-death hold, and GUID_OPTIONS enforcement. The fallbacks were also
> validated on 4.6: `validate_sitl.py` 9/9 (RC-override attitude sense, signs
> correct) and `probe_nogps_modes.py` 10/10 (ALT_HOLD + STABILIZE enter/arm/steer
> with GPS disabled). **Not yet hardware-validated.**

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

## The flight path: GUIDED_NOGPS body-rates (STABILIZE / ALT_HOLD as fallbacks)

With `control_mode: guided_nogps` the FC sits in GUIDED_NOGPS and the companion
commands body **rates** + real thrust via `SET_ATTITUDE_TARGET`
(`backend.send_body_rates`); the rates are integrated by the airframe so a noisy
detector box yields smooth motion. TRACK is range-hold; DIVE flies **pursuit
guidance**, driving the velocity vector onto the line-of-sight. Requires
`GUID_OPTIONS` bit 3 (ThrustAsThrust), set+verified in the preflight param check.

### Handover / failsafe model (guided_nogps)

This differs from the RC-override fallbacks — in GUIDED_NOGPS the FC will not
auto-revert to the pilot's sticks, so the companion owns the handover explicitly:

- **STANDBY** (while the FC is in GUIDED_NOGPS) — the companion **holds a level
  hover**. It never leaves the FC coasting on the last attitude.
- **Manual recovery** — the pilot flips the FC-mode channel **out of**
  GUIDED_NOGPS; the companion then commands nothing (`control_ready()` is false)
  and the pilot has the sticks.
- **Pi death** — the FC holds via ArduCopter's GUIDED command timeout, and the
  companion also emits a ~1 Hz GCS heartbeat so `FS_GCS` is armed. For GPS-denied,
  set the GCS failsafe to **LAND** (RTL/SmartRTL need GPS).

### Fallbacks: STABILIZE / ALT_HOLD via RC override

Set `control_mode` to match the FC's flight mode; the backend then injects AETR
sticks via `RC_CHANNELS_OVERRIDE` and `control_mode` picks the throttle semantics:

- **STABILIZE** (`control_mode: stabilize`) — direct throttle, no FC altitude
  hold. Full-down really cuts power → a true steep dive (15.9 m/s, 77° in SITL).
  TRACK altitude is held by the companion's **adaptive hover**: a vertical-velocity
  PI loop on `VFR_HUD.climb` (`_adaptive_throttle`) that learns the hover throttle
  and damps climb — so you set only a rough `stab_hover_throttle_us` seed and it
  self-corrects. SITL-proven: from a deliberately-low seed (1300) it learned 1474
  and drove climb to 0 in ~6 s (`scripts/learn_hover_sitl.py`). Frozen during a
  commanded dive; needs `VFR_HUD` streamed (falls back to the fixed seed
  otherwise). No altitude *floor*.
- **ALT_HOLD** (`control_mode: althold`) — throttle is a climb *rate*, 0.5 holds
  altitude via baro, descent capped at `PILOT_SPEED_DN`. Gentler and altitude-safe
  but cannot dive aggressively (1.3–4.8 m/s).

Both fallbacks are GPS-free (baro + IMU only; no EKF origin) and both are pilot
modes, so handback is automatic: when the companion stops overriding (`release()`),
ArduPilot reverts the channels to the real RC radio after its override timeout —
no GUIDED "ignore pilot / hover on timeout" lockout, and a dead Pi fail-safes to
the radio.

How the companion actually aims and commits the dive — including the fixed-camera
FOV constraint and the closed-loop constant-bearing homing that closes onto a
target below, level, or above (descend/hold/climb) — is in `dive-guidance.md`.

For the fallbacks, the mapping is `intent_to_rc_overrides()` / `_throttle_pwm()`
in `fc/ardupilot.py` (AETR PWM, like the Betaflight mapping but over MAVLink RC
override). Stick signs default to SITL-validated (+1,+1,+1); `ArduCopterRcMapping`
exposes per-axis flips and `stab_hover_throttle_us` for per-airframe hover tuning.
The guided_nogps path instead uses `backend.send_body_rates` (`SET_ATTITUDE_TARGET`,
body rates + thrust), with the control law in `guidance/rate_control.py`.

## Reproduce

```sh
# 4.6.3 (flight-era firmware); see docker/sitl-4.6/Dockerfile
docker build --platform linux/arm64 -t pifpv-sitl:4.6 docker/sitl-4.6
docker run -d --rm --name pifpv-sitl -p 127.0.0.1:5760:5760 pifpv-sitl:4.6
.venv/bin/python scripts/probe_nogps_modes.py  --connect tcp:127.0.0.1:5760  # GPS-off modes
.venv/bin/python scripts/measure_dive_sitl.py  --connect tcp:127.0.0.1:5760  # dive comparison
.venv/bin/python scripts/validate_sitl.py      --connect tcp:127.0.0.1:5760  # control sense
```
