# Raspberry Pi 5 setup log

Setting up the Pi 5 for the Panda showcase (roles and spikes: `docs/panda-setup.md`), 2026-10-08. The
robot was not touched.

**Summary**
- Reachable as `ssh pi` (`dots@codex`), key-only, passwordless sudo. Existing OS kept, not reflashed.
- Inventory: Pi 5, 4 GB RAM, stock `PREEMPT` (not RT) kernel, gigabit full-duplex Ethernet, fan
  connected, no throttling.
- Timing (S3 preview): stock kernel, headless, worst wake-up latency **97 µs under load** against a
  1000 µs period. With the desktop running, one **782 µs** spike at idle.
- Camera (S8): **not done**, no camera plugged in.
- OpenAI (S13 preview): Decisions p50 **189 ms**, p95 **477 ms**; Responses p50 **1.26 s**, p95
  **2.28 s**. Measured from the Pi but **through the Mac** (SSH tunnel), not on the Pi's own Wi-Fi.
- Open problem: the Pi **cannot join the WPA3-only `MissionRobotics` Wi-Fi**; the guest network needs
  a captive-portal click-through. Internet currently comes through a reverse SOCKS tunnel from the
  Mac.

## 1. Finding the Pi

- Mac side: the USB-C adapter is `en2` ("USB 10/100/1000 LAN"), link at 1000baseT full duplex, with a
  manually set `192.168.1.10/24` already on it (no DHCP on this cable).
- The Pi had no IPv4 on the cable and `raspberrypi.local` did not answer ping. Found by mDNS
  (`dns-sd -B _workstation._tcp local.`) as `raspberrypi` on `en2`, IPv6 link-local
  `fe80::f019:a2e9:cbe9:4a3d%en2`, MAC `88:a2:9e:da:c2:ec`.
- Port 22 was closed. With the card in the Mac: the SD card held the stock **Raspberry Pi OS
  desktop** image (pi-gen stage4, 2026-09-15); its first-boot wizard had never run (no monitor).

## 2. SSH (existing OS kept, no reflash)

Only the boot partition was edited, from the Mac (originals saved in
`/boot/firmware/orig-cloudinit/`):

- `user-data`: `hostname: codex`, `ssh_pwauth: false`; `meta-data`: `instance_id` bumped so
  cloud-init treats the next boot as a new instance.
- Finding: on this image cloud-init applies the hostname but **does not create users** from
  `users:` (`dots` never appeared; `id dots` → no such user). `bootcmd` does run, as root, on every
  boot. A one-off `bootcmd` script created `dots` (`useradd`, password field `*`, so key-only),
  installed the Mac's `~/.ssh/id_ed25519.pub`, added `/etc/sudoers.d/010_dots-nopasswd`, and ran
  `systemctl enable ssh`. An empty `ssh` file on the boot partition also turns SSH on (consumed at
  boot). The bootstrap was removed from `user-data` once login worked.
- `eth0`: static `192.168.1.20/32` with a single route to the Mac (`192.168.1.10/32`), no default
  route, IPv6 link-local only:

  ```
  sudo nmcli con mod "Wired connection 1" ipv4.method manual ipv4.addresses 192.168.1.20/32 \
    ipv4.routes 192.168.1.10/32 ipv4.never-default yes ipv6.method link-local
  ```

  It was `/24` at first, which hid the guest network's portal at `192.168.1.1` behind the cable.
- Mac `~/.ssh/config`:

  ```
  Host pi codex
      HostName 192.168.1.20
      HostKeyAlias codex.local
      User dots
      IdentityFile ~/.ssh/id_ed25519
      IdentitiesOnly yes
      ServerAliveInterval 60
      ServerAliveCountMax 3
  ```

- No RTC battery: the Pi booted thinking it was 2026-09-14. Clock set from the Mac
  (`ssh pi sudo date -u -s "$(date -u +'%Y-%m-%d %H:%M:%S')"`); timezone `America/Los_Angeles`.
- Still there: the SSH banner "SSH may not work until a valid user has been set up"
  (`/etc/ssh/sshd_config.d/rename_user.conf`, from the unfinished desktop first-boot wizard). Harmless.

## 3. Internet for the Pi

**Wi-Fi.** `wlan0` was `unavailable` until the country was set
(`sudo raspi-config nonint do_wifi_country US`). Then:

- `MissionRobotics` is **WPA3-only** (SAE, FT/SAE). The password was entered by Anthony via
  `ssh -t pi sudo nmcli --ask ...`. Every attempt fails: `CTRL-EVENT-AUTH-REJECT auth_type=3
  auth_transaction=2 status_code=1`, i.e. the AP rejects the SAE *confirm*, then
  `ASSOC-REJECT status_code=16`. Requiring PMF (`802-11-wireless-security.pmf 3`) did not help.
  Chip: BCM4345/6, firmware 7.45.265 (2023-08-29), which does SAE in firmware. A confirm failure
  looks the same for a wrong password and for a client/AP WPA3 mismatch; not separated yet. Ways to
  tell: switch the SSID to WPA2/WPA3 transition or disable fast roaming (802.11r) briefly, or try a
  USB Wi-Fi adapter.
- `MissionRoboticsGuest` (WPA2/WPA3) joins at once (`192.168.2.12/24`), but it's a **UniFi captive
  portal** (`http://192.168.1.1:8880/guest/s/default/`, "Hotspot Portal", `auth: none`): HTTPS is
  intercepted with a self-signed `CN=UDR` certificate until the portal is accepted. Not accepted yet;
  it needs a browser click-through or authorizing the Pi's Wi-Fi MAC `88:a2:9e:da:c2:ed` in UniFi.
- Why it matters: under Option B the Ethernet port belongs to the robot, so Wi-Fi is the control
  box's only internet. A Pi that can't join a WPA3-only network needs a WPA2 network, a USB Wi-Fi
  adapter with good WPA3 support (e.g. MediaTek MT7921AU) or a second Ethernet port.

**What was used instead (temporary): reverse SOCKS through the Mac**, no Mac network settings changed.

```
# on the Mac (one background ssh process)
ssh -f -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -R 1080 pi
# on the Pi
/etc/apt/apt.conf.d/90robojev-proxy:  Acquire::http(s)::Proxy "socks5h://127.0.0.1:1080";
git config --global http.proxy socks5h://127.0.0.1:1080
curl -x socks5h://127.0.0.1:1080 ...   # uv install, API timing
```

Remove when the Pi has its own internet: `sudo rm /etc/apt/apt.conf.d/90robojev-proxy`,
`git config --global --unset http.proxy`, and kill the Mac's `ssh -f -N ... -R 1080 pi`.

## 4. Inventory

| Item | Value |
|---|---|
| Board | Raspberry Pi 5 Model B Rev 1.1, 4 cores |
| RAM | 4 GB (+2 GB zram swap) |
| Storage | 117 GB microSD, 6.7 GB used before the upgrade |
| OS | Raspberry Pi OS (desktop build), Debian 13.7 trixie |
| Kernel | `6.18.50+rpt-rpi-2712 #1 SMP PREEMPT` (stock, not PREEMPT_RT); no RT kernel package in the Pi repos |
| Bootloader | 2026-01-21 |
| Temperature | 45 °C idle, 60–63 °C after 5 min of `stress-ng`; `throttled=0x0` throughout |
| Cooler | `pwmfan` hwmon present, so a fan is connected (Active Cooler or case fan); 0 rpm at idle |
| `eth0` | 1000 Mb/s, full duplex, link detected (meets Option B's bar) |
| `wlan0` | BCM4345/6, see §3 |
| USB | Root hubs only: **no camera plugged in** |
| Default target | was `graphical.target` (desktop, incl. `piwiz`); now **`multi-user.target`** (headless) |

## 5. Baseline

```
sudo apt-get update && sudo apt-get -y upgrade        # no kernel change in this upgrade
sudo apt-get -y install git build-essential cmake python3-venv rt-tests stress-ng ethtool usbutils v4l-utils
curl -LsSf https://astral.sh/uv/install.sh | sh       # uv 0.12.24
git clone -b claude/franka https://github.com/alee792/robojev.git ~/robojev
```

## 6. Timing jitter on the stock kernel (S3 preview)

`sudo cyclictest -m -S -p 90 -i 1000 -D 5m -q`, idle, then with `stress-ng --cpu 4 --io 2 -t 5m`
alongside. Run twice: first with the desktop session still up, then headless after
`sudo systemctl set-default multi-user.target` and a reboot. Note: `-S` spreads threads 500 µs
apart, so cores 1–3 ran at 1500/2000/2500 µs intervals; core 0 is the one at the 1 ms rate.

Max wake-up latency, µs (avg was 2–3 µs everywhere):

| Run | Core 0 | Core 1 | Core 2 | Core 3 | Worst |
|---|---|---|---|---|---|
| Desktop, idle | **782** | 30 | 45 | 32 | 782 |
| Desktop, load | 58 | 29 | 38 | 30 | 58 |
| Headless, idle | 44 | 68 | 22 | 37 | 68 |
| Headless, load | 97 | 49 | 18 | 62 | 97 |

Read: on the stock `PREEMPT` kernel, headless, the worst wake-up was under 100 µs in 1.4 million
loops, against a 1000 µs period. The 782 µs desktop spike shows how much a stray process can cost
without RT; a control box should run headless. This is only a preview: libfranka's
`communication_test` (S3 proper) also includes the network round trip to the robot, which must fit
in the same millisecond, and 5 minutes is short next to a demo.

**PREEMPT_RT (not done, needs your OK).** Raspberry Pi doesn't ship an RT kernel package. PREEMPT_RT
has been in mainline since 6.12, so it's a config switch on the Pi kernel source:
build `raspberrypi/linux` `rpi-6.18.y` with `bcm2712_defconfig` + `CONFIG_PREEMPT_RT=y` (on the Pi,
about an hour), install the modules, copy the image to `/boot/firmware/kernel_2712_rt.img`, and boot
it with `kernel=kernel_2712_rt.img` in `config.txt` (the stock kernel stays as the fallback). Then
`isolcpus=3 nohz_full=3 rcu_nocbs=3` on `cmdline.txt` for the control-loop core, and rerun this
table. Alternative: Ubuntu 24.04 for Pi, which has a `linux-realtime` kernel through Ubuntu Pro.

## 7. Camera (S8 preview)

Not done: no camera was plugged in (`lsusb` shows only root hubs). `v4l-utils` is installed. When
it's connected: `lsusb` (Orbbec's USB vendor ID is `2bc5`), then Orbbec's OpenNI2 or OrbbecSDK
ARM64 build for depth.

## 8. Cloud latency (S13 preview)

- `OPENAI_API_KEY` copied from the Mac's shell environment to `~/.config/robojev/env` on the Pi (mode
  600), piped over SSH, never printed.
- Decisions docs read: `developers.openai.com/api/docs/guides/decisions` (public beta, `gpt-6-luna`
  only; `choice` questions return `choice`, per-option `probabilities` and `confidence`; images only as
  inline base64 `input_image`). The key has access.
- Calls: `curl` from the Pi, a fresh TLS connection each time (no keep-alive), so each time includes
  the handshake.
  - Responses: `{"model":"gpt-6-luna","input":"Reply with the single word OK.","reasoning":{"effort":"low"},"max_output_tokens":16}`
  - Decisions: one `choice` question ("What should the robot arm do right now?", `hold` / `continue`,
    input "A person says: actually, spell PANDA instead.").
    Example answer: `continue` 0.81, confidence 0.62.
- **Path:** the Pi had no internet of its own, so the Pi's calls went through the SOCKS tunnel: Pi →
  cable → Mac → the Mac's Wi-Fi (office LAN) → internet. This isn't the Pi on Wi-Fi, and it isn't a
  demo-day path (under Option B the Ethernet port is the robot's). The same calls straight from the
  Mac show what the tunnel costs.

20 calls each, all HTTP 200, ms:

| From | Call | p50 | p95 | min | max |
|---|---|---|---|---|---|
| Pi via Mac tunnel | Responses | 1259 | 2276 | 954 | 2585 |
| Pi via Mac tunnel | Decisions | 189 | 477 | 113 | 515 |
| Mac directly | Responses | 1409 | 1927 | 803 | 2353 |
| Mac directly | Decisions | 117 | 138 | 102 | 141 |

Read: Decisions is fast (~117 ms p50 from the Mac, under the ~150 ms the demo assumes); the tunnel
adds ~70 ms at p50 and a long tail. S13's bar (decision p95 under ~400 ms) is met from the Mac but
not quite through the tunnel; remeasure on the Pi's own Wi-Fi. The minimal Responses call is 1–2.5 s;
a real planner call with a plan in it will be longer.

Script (on the Pi at `~/s3/lat.sh`): loops `curl -sS -x socks5h://127.0.0.1:1080 -o /tmp/lat_body.json
-w '%{http_code} %{time_total}\n' -m 60 <url> -H "Authorization: Bearer $OPENAI_API_KEY" -d <body>`.
Raw cyclictest and timing outputs are in `~/s3/` on the Pi.
