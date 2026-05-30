# pi-fpv-companion — User Guide


---

## The one rule

**The 3-position switch is your steering wheel.** Flick it down and the companion
lets go — you have the sticks, instantly. Keep a finger near it the whole time.

| Switch | Mode | What happens |
|--------|------|--------------|
| **Down** | STANDBY | Companion is asleep. **You fly, normally.** |
| **Middle** | TRACK | It takes the sticks: yaws to center the target, follows, holds height. |
| **Up** | DIVE | It commits: noses down and drops onto the target. |

That's the whole mental model. Everything below is making it work and trusting it.

---

## Part 1 — Set it up (once)

### a. Put it on the aircraft
- Pi UART → flight controller UART (3 wires: **Pi pin 8 → FC RX, pin 10 → FC TX,
  GND↔GND**). Composite video out of the Pi → FC camera-in. Details + photos:
  `hardware.md`.

### b. Install the software on the Pi
```bash
bash scripts/install-pi.sh      # installs to /opt/pi-fpv-companion, sets up the service
```
Say yes to the IMX500 firmware and the boot config when asked, then **reboot**.

### c. Set up the flight controller (ArduCopter 4.6+)
In Mission Planner → Full Parameter List, set and reboot the FC:

| Param | Value | Meaning |
|-------|-------|---------|
| `SERIALn_PROTOCOL` / `_BAUD` | `2` / `115` | MAVLink on the UART wired to the Pi |
| `SRn_EXTRA2` | `≥ 5` | streams climb rate (the companion holds height with it) |
| `ANGLE_MAX` | `4500` | 45° max lean — must match `fc.angle_max_deg` |
| `RC7_OPTION` | `0` | leave ch7 alone — it's the companion's engage switch |
| `RC9_OPTION` | `0` | leave ch9 alone — it's the companion's target-select input |
| flight-mode switch | → **STABILIZE** | the mode the companion flies in |
| ch7 (a spare 3-pos switch) | STANDBY/TRACK/DIVE | your steering wheel (above) |
| ch9 (a spare input, e.g. rocker) | cycle target (tap) | maps in FreedomTX → ch9 |

Keep the FC's own **RC-loss and battery failsafes** configured — they're your
backstop if everything else fails.

### d. Point it at the right targets
Edit `config/imx500.yaml` on the Pi (`/opt/pi-fpv-companion/config/imx500.yaml`):
```yaml
camera: { type: imx500, hflip: false, vflip: false }   # flip to match your mount
fc:
  control_mode: stabilize     # diving mode (default). 'althold' = gentle/altitude-safe.
  rc_roll_sign: 1             # <- you'll confirm these on the bench (Part 2)
  rc_pitch_sign: 1
  rc_yaw_sign: 1
guidance:
  classes_of_interest: [person, car, truck, boat]   # what to lock onto
  dive_vrate_gain: 17.0       # closed-loop dive vertical homing (0 = DIVE just
                              # leans in). See dive-guidance.md.
```

> DIVE **closes onto a target below, level, or above you** — it commits a gentle
> forward lean and uses the throttle to hold the target's frame position, so the
> flight path follows the line of sight (constant-bearing homing). See
> `docs/dive-guidance.md`. The shipped `config/imx500.yaml` already enables the
> tuned dive; it needs `VFR_HUD` streaming (`SR*_EXTRA2`) to close the loop.

---

## Part 2 — The bench check (props OFF, 15 minutes, do not skip)

This is the difference between a good first flight and a crash. **Take the props
off.** Then run it by hand so you can watch it think:

```bash
sudo systemctl stop pi-fpv-companion      # free it up
/opt/pi-fpv-companion/.venv/bin/python -m pi_fpv_companion \
    --config /opt/pi-fpv-companion/config/imx500.yaml \
    --force-mode track     # pretend the switch is in TRACK
# the composited feed (bbox + HUD) goes out the analog composite / TV out
# open Mission Planner to watch the stick commands it sends
```

Walk an object (a person works) across the camera and check:

- 👁️ **It sees it** — a box tracks the object in the browser view.
- ↪️ **It aims the RIGHT way** *(the critical one)* — object to the **right** of
  center → it commands **yaw right** (toward it). Object **bigger/closer** → it
  **eases off**, doesn't lunge. If any axis goes the wrong way, flip that
  `rc_*_sign` (or fix `camera.hflip`) and re-check. *A wrong sign makes it chase
  away from the target, faster and faster — find it here, not in the air.*
- ✋ **You can take it back** — set the real switch to STANDBY → your sticks
  return at once. Then kill the program (Ctrl-C) mid-track → the FC falls back to
  your radio within a second or two.

Re-enable the service when done: `sudo systemctl start pi-fpv-companion`.

---

## Part 3 — How to actually fly it

Build up gently. First time, fly somewhere open with room to recover, and treat it
like a maiden flight.

**1. Power up and warm up.** Battery in, give it ~a minute. In your goggles you
should see the live feed with boxes on detected objects. Switch **down** (STANDBY).

**2. Take off and fly — normally.** It's just your STABILIZE quad right now; the
companion is asleep. Get comfortable, climb to a safe height with margin below you.

**3. Line up — and pick your target.** Put targets in the frame. With the
multi-target tracker (`tracker.type: multi_iou`, the IMX500 default) the HUD shows
**every** detection (faint boxes) in STANDBY, with the locked one bold. Tap your
**select input** (`fc.select_channel` — on the Tango 2 the top switches are full,
so map a spare input to **ch9**) to **cycle the lock** to the next target.
Selection works **only in STANDBY** — whatever is locked when you flick to TRACK
is **frozen** through TRACK and DIVE, so a stray bump can't swap targets
mid-engagement (and if your target is lost while committed it **holds**, it never
silently re-targets). Choose before you commit; to re-choose, flick back to
STANDBY. (No select channel wired? It auto-locks the highest-confidence detection.)

**4. Hand it the wheel — flick to TRACK (middle).** Now *let go of the sticks.*
The companion yaws to center the target, leans in to follow, and holds your
altitude by itself. Watch it turn **toward** the target. It should feel like a
smooth, hands-off chase that keeps the target the same size in frame.
> If it ever turns *away* or wanders off — flick to STANDBY immediately. Something's
> miscalibrated; land and recheck Part 2.

**5. Follow as long as you like.** Re-take the sticks anytime by flicking to
STANDBY. TRACK won't dive — it just follows and holds range.

**6. Commit — flick to DIVE (up).** It commits to the target and closes, moving
altitude onto it — **dives** onto a target below you, **holds** for one level
ahead, **climbs** toward one above you. It works this out from where the target
sits in the frame (constant-bearing homing; see `docs/dive-guidance.md`). With
`dive_vrate_gain` at 0 it only leans in. There is **no automatic pull-up** — *you*
end the dive.

> A fixed forward camera can only *see* a ground target once it's far enough
> ahead (shallow enough); something steeply below you is below the frame. Engage
> from a moderate altitude for the most reliable closure.

**7. Bail out / finish — flick to STANDBY (down).** Instant manual control. Pull
up, recover, fly home.

That's it. STANDBY → fly. TRACK → it follows. DIVE → it commits. STANDBY → you're
back.

### What each mode feels like
- **STANDBY** — nothing different; you're flying.
- **TRACK** — hands-off; it gently yaws and leans to keep the target centered and
  the same distance away, holding height. Following, not attacking.
- **DIVE** — committed and aggressive: closes on the target and moves altitude
  onto it (descend / hold / climb depending on where it is). Short and decisive —
  you pull out by going STANDBY.

---

## Part 4 — Make it behave how you want (tuning)

Edit `config/imx500.yaml`, restart the service (`sudo systemctl restart
pi-fpv-companion`). One change at a time.

| You want… | Change |
|-----------|--------|
| It turns the wrong way | a `rc_*_sign` is flipped — **fix before flying** (Part 2) |
| Snappier / calmer yaw | `guidance.yaw_p_gain` up / down |
| Lead a moving target (less tail-chase) | `guidance.lead_time_s` → 0.2–0.6 s |
| Pick among multiple targets | `tracker.type: multi_iou` + `fc.select_channel` (tap to cycle) |
| Follow closer / farther | `guidance.desired_bbox_frac` up (closer) / down |
| Gentler / harder approach | `guidance.max_pitch_deg` |
| DIVE to actually change altitude | `guidance.dive_vrate_gain` > 0 (0 = just leans); needs VFR_HUD |
| Faster / slower dive vertical | `dive_max_descent_mps` / `dive_max_climb_mps` |
| DIVE loses an above target out the top | lower `dive_forward_deg` (gentler lean) |
| DIVE won't descend | confirm `VFR_HUD` streams (`SR*_EXTRA2`); the rate loop needs it |
| It holds altitude poorly | confirm `SRn_EXTRA2` is streaming; nudge `stab_hover_throttle_us` |
| Altitude bounces/hunts | lower `stab_hover_learn_kp`, then `stab_hover_learn_gain` |
| Stop chasing wrong objects | trim `classes_of_interest`; raise `safety.min_track_quality` |
| Calm, altitude-safe mode | `fc.control_mode: althold` (won't dive hard, but holds height) |

---

## Part 5 — When something's off

**Abort, always:** switch to STANDBY, or change your flight-mode switch out of
STABILIZE. Either gives you full manual control immediately.

- **Watch it live:** the composited feed (bbox + HUD) is on the analog composite / TV out.
- **Read the logs:** `journalctl -u pi-fpv-companion -f`
- **Restart it:** `sudo systemctl restart pi-fpv-companion`

| Problem | Likely fix |
|---------|-----------|
| Black screen / no video | composite wiring; `video.tv_mode` (PAL vs NTSC) |
| Won't respond to the switch | check ch7 reaches the FC (Mission Planner radio cal); thresholds |
| "never armed" / no control | UART wiring + `SERIALn` params; FC must be armed |
| Log says `no fresh VFR_HUD.climb` | set `SRn_EXTRA2 > 0` on the FC |
| It flies the wrong way | stick signs / `camera.hflip` (Part 2) |
| Camera keeps restarting | check the IMX500 shows in `rpicam-hello --list-cameras` |

---

## Please remember

- **No GPS, and in STABILIZE no altitude floor** — a dive keeps descending until
  *you* stop it. Fly with height to spare and a finger on the switch.
- **The bench check (Part 2) is the real safety test.** A wrong stick sign is the
  one thing that turns this dangerous; SITL can't catch it for your airframe.
- **You are the safety system:** the engage switch, a flight-mode switch back to
  manual, and the FC's own failsafes. Keep all three.
- Proven in simulation (ArduCopter 4.6.3), **not yet on a real airframe** — your
  first flights are the real test. Go slow, go low, build up.
