# ArduPilot SITL — validating the GPS-denied control surface

> **Control path is RC_CHANNELS_OVERRIDE into STABILIZE (default) or ALT_HOLD**
> (not GUIDED_NOGPS — see `docs/gps-denied-modes.md`). Validated on **ArduCopter
> 4.6.3** (build it: `docker/sitl-4.6/Dockerfile`): `validate_sitl.py` 9/9 (RC
> override steers correct sense, signs verified), `probe_nogps_modes.py` 10/10
> (STABILIZE + ALT_HOLD enter/arm/steer GPS-off), `measure_dive_sitl.py` (dive:
> STABILIZE ~16 m/s vs ALT_HOLD ~1–5 m/s → STABILIZE is the default).
> 4.6 notes: EKF3 needs ~45–60 s to settle before arming; use GUIDED `NAV_TAKEOFF`
> to get airborne (RC-override takeoff is unreliable on fresh 4.6); publish the
> port as `-p 127.0.0.1:5760:5760` (IPv6-only publish blocks IPv4). The
> GUIDED_NOGPS material below is historical context.

The MAVLink backend is wire-protocol-complete and was validated against a
loopback fake — but that fake only echoes messages. SITL is the gate that proves
a real ArduCopter flight stack accepts our commands and responds in the correct
sense. (Historical: the original path sent `SET_ATTITUDE_TARGET` (mask 3) in
GUIDED_NOGPS — validated 10/10 (2026-05-16); see "Result" below and
`docs/architecture-audit.md` §1.)

## Getting SITL running

There is **no official `ardupilot/ardupilot-sitl` Docker image** (an earlier
version of this doc was wrong). Real options, least effort first:

### A. Prebuilt community image — what we validated against

```sh
docker run -d --rm --name pifpv-sitl --platform linux/amd64 \
    -p 5760:5760 \
    radarku/ardupilot-sitl:latest
```

Key facts about this image, learned the hard way:

- It runs `sim_vehicle.py … --no-mavproxy`, so **there is no MAVProxy** — none
  of the `mode`/`arm`/`rc` console commands below apply to it. SITL's raw
  MAVLink is on **TCP 5760** (not UDP 14550). Drive it with pymavlink.
- It is **ArduCopter 4.0.3** (older line): EKF2 enabled, **EKF3 disabled by
  default**. `SET_MESSAGE_INTERVAL` is unreliable here — use
  `request_data_stream`.
- Its HEARTBEAT advertises component 0, but the autopilot that services
  params/arm/setpoints is **component 1**. Address commands to comp 1.
- amd64-only; on Apple Silicon it runs under emulation. SITL is not
  real-time-critical for this validation (we check message acceptance + sense,
  not latency), so emulation is fine — just slow to boot (~1 min).

Validate it (drives the real `ArduPilotBackend.send_intent()` path,
auto-configures, prints PASS/FAIL):

```sh
.venv/bin/python scripts/validate_sitl.py --connect tcp:127.0.0.1:5760
docker stop pifpv-sitl        # --rm cleans it up
```

To exercise the **whole production `Pipeline`** end-to-end (SyntheticCamera ->
tracker -> filter -> servo -> safety gate -> real `send_intent`) flying a
moving target in SITL, with a PASS/FAIL on the closed loop:

```sh
.venv/bin/python scripts/fly_sitl.py --connect tcp:127.0.0.1:5760
```

To probe which modes work with **no GPS at all** (disables GPS, reboots cold,
tests ALT_HOLD / STABILIZE enter+arm+steer):

```sh
.venv/bin/python scripts/probe_nogps_modes.py --connect tcp:127.0.0.1:5760
```

`scripts/fly_sitl.py --connect tcp:127.0.0.1:5760` runs the real production
Pipeline against SITL (closed loop; watch it in Mission Planner — there is no
on-screen viewer). The pilot keeps SITL in the matching mode.

### B. arm64-native, purpose-built for M1/M2 Macs

`github.com/uxduck/ardupilot-sitl-docker` builds an arm64-native image (no
emulation, full speed). Clone + follow its README. Heavier one-time build.

### C. Native build on the Mac (no Docker)

ArduPilot supports a macOS native build:
<https://ardupilot.org/dev/docs/building-setup-mac.html> — clone the repo, run
`Tools/environment_install/install-prereqs-mac.sh`, then
`./Tools/autotest/sim_vehicle.py -v ArduCopter -w`. Fast once built; installs a
toolchain + large repo on the host. This path *does* give you a MAVProxy
console (commands in the next section apply).

## Configuring SITL for GUIDED_NOGPS

`scripts/validate_sitl.py` does all of this automatically against image A. The
manual equivalent (image B/C, with a MAVProxy console) is:

```
param set ARMING_CHECK 0       # sim convenience: skip EKF/GPS-lock timing
arm throttle
mode GUIDED_NOGPS
rc 3 1500                      # mid throttle (GUIDED_NOGPS holds alt via thrust)
rc 7 1900                      # ch7 HIGH -> guidance allowed
```

**Do NOT `param set AHRS_EKF_TYPE 3`.** That selects *EKF3*, not "DCM-only"
(the original mistake here). On a build where EK3 is disabled — e.g.
ArduCopter 4.0.x — it bricks arming with `PreArm: no EKF3 cores`. AHRS must
point at an EKF that actually has cores (4.0.3 → leave it at EKF2). The
GUIDED_NOGPS / `SET_ATTITUDE_TARGET` surface is independent of EKF flavour and
of GPS presence; a default SITL (sim GPS, working EKF, in GUIDED_NOGPS) is a
valid test of it.

In a real airframe the engage switch is the **flight-MODE channel**: releasing
it leaves GUIDED_NOGPS and instantly returns full manual control, independent
of the Pi. SITL approximates this with `mode` / `rc 7`:

```
rc 7 1900           # ch7 HIGH -> guidance allowed
rc 7 1000           # ch7 LOW  -> guidance muted (safety gate stops commands)
```

## Connection string mapping

| Endpoint                | Where it goes                                    |
|-------------------------|--------------------------------------------------|
| `tcp:127.0.0.1:5760`    | SITL's native MAVLink (image A — no MAVProxy)    |
| `udpin:127.0.0.1:14550` | listen on 14550; a MAVProxy GCS forward hits us  |
| `udp:127.0.0.1:14550`   | connect outbound to a MAVProxy out port          |
| `/dev/ttyAMA0`          | real Pi UART                                     |

`validate_sitl.py` and `fly_sitl.py` both default to `tcp:127.0.0.1:5760`
(image A); pass `--connect udpin:127.0.0.1:14550` to reach a different endpoint.

## Result (2026-05-16, ArduCopter 4.0.3 SITL)

`scripts/validate_sitl.py`, 10/10. The build accepts GUIDED_NOGPS; arms in it;
the real `send_intent()` `SET_ATTITUDE_TARGET` (mask 3) stream is accepted with
no rejection, no mode kick, no disarm across the whole run; and the copter
tracks every axis in the correct sense:

| Commanded            | Measured        |
|----------------------|-----------------|
| yaw rate +30 dps     | +29.1 dps       |
| yaw rate −30 dps     | −28.4 dps       |
| pitch −10° (nose-down/approach) | −9.5°  |
| roll +12°            | +11.5°          |

## What this validates — and what it does not

Validates: a real ArduCopter accepts the exact attitude mask the flight code
sends in GUIDED_NOGPS, and responds in the correct direction. This closes the
one gap the loopback fake could not exercise (audit §1).

Does **not** validate (hardware-gated, `docs/deployment-safety.md`):
GUIDED_NOGPS arming on a real GPS-denied airframe, the switch-as-flight-mode-
channel wiring, thrust/`GUID_OPTIONS` altitude semantics, servo-gain tuning,
camera-mount sign, and re-confirmation on the exact flight firmware (4.0.3 is
older than a likely flight build; EKF3 defaults and minor `SET_ATTITUDE_TARGET`
handling differ on Copter 4.5/4.6).
