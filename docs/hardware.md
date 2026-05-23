# Hardware notes

Tribal knowledge about the Pi Zero 2W setup that is easy to forget between sessions.

## CVBS (composite video) out

The Pi Zero 2W has **no RCA jack**. Composite video is exposed as two unpopulated
through-hole test points on the bottom of the board, silkscreened "TV":

- **Left pad (nearest the "TV" silkscreen text) = composite signal (+)**, drives the RCA centre / VTX video-in
- **Right pad = GND**, drives the RCA sleeve / VTX video-GND

The signal is 1 Vpp DC-coupled at the SoC. Modern analog VTX inputs and FC analog
cam-ins are AC-coupled internally, so **no external coupling cap, resistor, or buffer
is needed** for direct-solder to a 75 Ω load. (If you happen to be driving a DC-coupled
analog input, add a 10 µF series cap.) Reference: Adafruit / PiHut tutorials show
direct-solder TV pad → RCA with nothing in between.

Solder considerations: pads are small unprotected through-holes connected directly to
the SoC's VEC pin. Use a temperature-controlled iron at 315–350°C, pre-tin the pad,
keep dwell under 2 s, strain-relieve the wire — these pads will rip off if pulled.
ESD strap recommended. **Shorting the signal pad to a 3.3 V or 5 V rail will almost
certainly destroy the SoC's VEC.**

Keep the run to the FC cam-in pad as short as possible. Long unshielded runs near
ESCs introduce visible noise.

### `/boot/firmware/config.txt` entries (Bookworm and Trixie)

**Under the default vc4-kms-v3d driver, the legacy `enable_tvout=1`/`sdtv_mode`/`sdtv_aspect` options
are silently ignored.** The correct switch is the dtoverlay's `composite` parameter:

```
# pi-fpv-companion: composite output via vc4 KMS
dtoverlay=vc4-kms-v3d,composite
```

PAL vs NTSC selection happens in `/boot/firmware/cmdline.txt` (single line, space-separated):

```
... vc4.tv_norm=PAL    # or vc4.tv_norm=NTSC
```

**Trixie note**: the older `vc4-fkms-v3d` (firmware KMS) overlay is **deprecated** on Trixie+. Don't
fall back to it. The framebuffer device `/dev/fb0` will NOT auto-exist under modern KMS — output is
via DRM dumb buffers on `/dev/dri/card0`. This is **handled**: `video/drm_framebuffer.py`
(`DrmFramebuffer`, ctypes DRM dumb-buffer) is the default sink on Trixie, selected automatically by
`main.py` when `/dev/dri/card0` exists. `LinuxFramebuffer` (legacy `/dev/fb0`) is still used if that
device is present (older Pi OS / fkms); otherwise the DRM path takes over. No rewrite pending.

**Optional power/clock savings** (only after you confirm composite output works — disabling HDMI
removes a debug path):

```
hdmi_blanking=2
```

### Resolution

- NTSC: 720 × 480 interlaced (treat as 720 × 240 per field, or render 720 × 480 progressive and accept softness)
- PAL:  720 × 576 interlaced

We render at this resolution natively — no scaling. Downscale the camera frame to fit, draw overlay, push to framebuffer.

## UART to flight controller

Use GPIO 14 (TXD0, pin 8) and GPIO 15 (RXD0, pin 10), plus ground.

### `/boot/firmware/config.txt`

```
enable_uart=1
dtoverlay=disable-bt   # free the PL011 hardware UART for our use
```

### `/boot/firmware/cmdline.txt`

Remove `console=serial0,115200` if present — otherwise the kernel grabs the UART for its console.

### Disable the serial console service

```
sudo systemctl disable --now serial-getty@ttyAMA0.service
```

### Baud rates

Project standard: **115200** for both backends. Set ArduPilot's `SERIALn_BAUD = 115` on whichever UART you wire the Pi to; Betaflight MSP defaults to 115200 anyway.

ArduPilot's TELEM ports default higher (921600), but 115200 simplifies wiring (no level-shift or ringing concerns over the short Pi-to-FC run), shares a single setting across both firmware backends, and is plenty for HEARTBEAT + RC_CHANNELS at 10 Hz plus our outbound command stream.

## Camera (CSI)

Both camera options use the standard Zero 2W mini CSI connector (22-pin, 0.5 mm pitch). The Pi Zero 2W ribbon is the narrow one — needs the right adapter if the camera came with a 15-pin cable.

### Regular Pi Camera

Use `picamera2`. Bookworm 64-bit Lite ships it via apt:

```
sudo apt install -y python3-picamera2 python3-libcamera
```

Pip-install of picamera2 does not work; it is a system package.

### IMX500 AI Camera

Same CSI connector, same `picamera2` API. Additional pieces:

```
sudo apt install -y imx500-all
```

This pulls in:
- Firmware blobs for the on-sensor NPU
- Sample `.rpk` model packages
- `imx500-models` directory at `/usr/share/imx500-models/`

Models are packaged as `.rpk` files compiled through Sony's AITRIOS tooling. You load one with picamera2's `IMX500()` helper before starting the camera. The detector outputs land in the picamera2 metadata stream under the `CnnOutputTensor` key (or similar — verify against current picamera2 docs at run time).

The IMX500 has finite on-chip model memory (~8 MB). Pick a model that fits — the bundled `imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk` is the canonical starting point for object detection.

## Power

- Pi Zero 2W draws 350 mA idle, peaks above 1 A under load (boot + camera + Wi-Fi together is the worst case).
- Drone 5V BEC must have headroom. Shared with the FC's 5V rail is usually fine; sharing with a high-current servo rail is not.
- Add a 470 µF low-ESR cap across the Pi's 5V/GND as close to the board as possible — protects against VTX/ESC switching noise.

## Wiring summary

```
Pi Zero 2W                         FC
----------                         --
GPIO 14 (TXD0, pin 8)  ─────────▶  UARTx RX
GPIO 15 (RXD0, pin 10) ◀─────────  UARTx TX
GND     (pin 6 or 9)   ──────────  GND
5V      (pin 2 or 4)   ◀──────────  5V BEC

TV pad (signal)        ──────────▶ FC cam-in signal
TV pad (gnd)           ──────────  FC cam-in GND
```
