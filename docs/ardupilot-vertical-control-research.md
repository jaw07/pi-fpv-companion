# ArduPilot Copter vertical control via SET_ATTITUDE_TARGET (GUIDED_NOGPS)

Research for the DIVE gravity-descent feature (task: dynamic vertical control). Sourced
from ArduPilot master source + MAVLink message definitions (GitHub), 2026-05-23.

## TL;DR for our code
- **thrust = climb-rate by DEFAULT** (do NOT set `GUID_OPTIONS` bit 3). Our `0.5=hold,
  <0.5=descend` is correct unmodified.
- Descent rate at `thrust=0` is **`WPNAV_SPEED_DN`** (default 1.5 m/s). Raise it (≈6–10 m/s)
  for an aggressive dive. Climb at `thrust=1` is `WPNAV_SPEED_UP` (default 2.5 m/s).
- **Use Copter 4.6+** — earlier versions have a wrong SET_ATTITUDE_TARGET angular-rate frame.
- Altitude floor: stream `GLOBAL_POSITION_INT` (id 33) and clamp thrust toward 0.5 near floor.
- ⚠️ VERIFY before trusting: agent claims body-rate typemask is all-or-nothing (provide all 3
  rates or ignore all 3). Our mask=3 sends attitude-quaternion + yaw-RATE only (ignore roll/
  pitch rate). The attitude+yaw-rate pattern is common, so this may be an over-read — confirm
  our SET_ATTITUDE_TARGET is accepted on the actual 4.6 FC (armed, GUIDED_NOGPS).

## 1. thrust semantics
- `GUID_OPTIONS` bit 3 = `SetAttitudeTarget_ThrustAsThrust` = value **8**. Default 0 → thrust
  is climb rate. Set 8 → raw normalized throttle (open-loop, unsafe for a dive — don't).
- Mapping (GCS_MAVLink_Copter.cpp): clamp thrust [0,1]; `=0.5` hold; `>0.5` climb at
  `(thrust-0.5)*2*WPNAV_SPEED_UP`; `<0.5` descend at `(0.5-thrust)*2*WPNAV_SPEED_DN`.
  (We don't use GUIDED_NOGPS; the STABILIZE dive uses RC-override throttle, and the
  closed-loop DIVE commands a vertical RATE the companion tracks on `VFR_HUD.climb`
  — see `dive-guidance.md`.)
- Introduced Copter 4.1.0; present in all 4.x.

## 2. GUIDED_NOGPS
- `requires_position()=false` → no GPS needed to arm/run (angle-control submode of GUIDED).
- Altitude/climb-rate control **needs a working barometer** (EKF `EK3_SRC1_POSZ` default Baro=1;
  rangefinder=2 optional, low-AGL only).
- Always provide thrust (THROTTLE_IGNORE bit 64 → command rejected). Quaternion must be unit-norm
  unless ATTITUDE_IGNORE (128).
- **Version gotchas:** Copter <4.6 had wrong angular-rate input frame for SET_ATTITUDE_TARGET
  (fixed 4.6.0-beta1); also a guided thrust/climb-rate switching internal-error fixed in 4.6.0.
  → run **4.6+**.

## 3. Telemetry for vertical feedback
- `VFR_HUD` (id **74**): `alt` m (ArduPilot: EKF absolute/AMSL), `climb` m/s (**+up**).
- `GLOBAL_POSITION_INT` (id **33**): `relative_alt` mm **above home**, `vz` cm/s (**+down**),
  `alt` mm MSL. Best single source for an altitude floor + vspeed monitor (explicit int units).
- Request via `MAV_CMD_SET_MESSAGE_INTERVAL` (cmd **511**): param1=msg id, param2=interval µs
  (100000=10 Hz, 50000=20 Hz; -1 disable, 0 default). One-shot: `MAV_CMD_REQUEST_MESSAGE` (512).
- Recommended: GLOBAL_POSITION_INT 10–20 Hz. Send SET_ATTITUDE_TARGET ≥20–50 Hz, never lapse.
- Mind sign conventions: `vz` +down vs `VFR_HUD.climb` +up.

## 4. Descent params
- **`WPNAV_SPEED_DN`** (master `WPNAV_SPD_DN`, m/s; 4.x stable cm/s, default 150) — bounds the
  descent in climb-rate mode. THE lever. (master renamed to metric: `WPNAV_SPD_*` m/s.)
- `WPNAV_SPEED_UP` bounds climb (default 2.5 m/s / 250 cm/s).
- `PILOT_SPEED_DN` is for *pilot stick* (AltHold/Loiter) — does NOT bound GUIDED_NOGPS attitude-
  target climb rate. No dedicated GUID_* vertical-rate param.
- Aggressive dive: raise WPNAV_SPEED_DN to ≈600–1000 cm/s (6–10 m/s); validate in SITL (EKF/baro
  lag worsens at high descent rates / in propwash).

## 5. Altitude floor / terrain
- Software floor (recommended, in our loop): monitor `GLOBAL_POSITION_INT.relative_alt` at
  10–20 Hz; near the floor, clamp commanded thrust toward/above 0.5 and level attitude.
- Rangefinder AGL: `RNGFND1_TYPE`, `RNGFND1_MIN`/`MAX` m (def 0.2/7.0), `RNGFND1_ORIENT=25` (Down);
  `EK3_SRC1_POSZ=2` to feed it. Low-AGL backstop only.
- ArduPilot backstop: `FENCE_ALT_MIN` (m), frame via `FENCE_ALT_MIN_TP` (default Above Home),
  enable bit 3 in `FENCE_TYPE` + `FENCE_ENABLE=1`, `FENCE_ACTION`. Note: min-alt fence only arms
  after first climbing above it; a breach yanks the craft out of GUIDED_NOGPS → last-resort only.

## Sources
ArduPilot master: mode.h, mode_guided.cpp, mode_guided_nogps.cpp, GCS_MAVLink_Copter.cpp,
Parameters.cpp, ReleaseNotes.txt; libraries AC_WPNav.cpp, AC_Fence.cpp, AP_RangeFinder_Params.cpp,
AP_NavEKF/AP_NavEKF_Source.cpp, GCS_MAVLink/GCS_Common.cpp. MAVLink common.xml (74, 82, 511, 512)
and standard.xml (33). (Agent read raw source via curl; WebSearch/WebFetch were unavailable.)
