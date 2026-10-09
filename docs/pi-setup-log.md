# Raspberry Pi 5 setup log

Setting up the Pi 5 for the Panda showcase (roles and spikes: `docs/panda-setup.md`). The robot was not
touched. Status: **in progress**: reachable and inventoried; blocked on internet for the Pi (steps 5–8).

## 1. Finding the Pi

- Mac side: the USB-C adapter is `en2` ("USB 10/100/1000 LAN"), link up at 1000baseT full duplex,
  with a static `192.168.1.10/24` already set (no DHCP on this cable).
- The Pi had no IPv4 on the cable. It was found by mDNS (`dns-sd -B _workstation._tcp`) as
  `raspberrypi` on `en2`, IPv6 link-local `fe80::f019:a2e9:cbe9:4a3d%en2`, MAC `88:a2:9e:da:c2:ec`.
- Port 22 was closed: the SD card held the stock **Raspberry Pi OS desktop** image (pi-gen stage4,
  2026-09-15) whose first-boot wizard never ran (no monitor).

## 2. SSH (without reflashing)

The existing OS was kept. With the card in the Mac, only the boot partition was edited (originals
saved in `/boot/firmware/orig-cloudinit/`):

- `user-data`: `hostname: codex`, `ssh_pwauth: false`; `meta-data`: `instance_id` bumped so
  cloud-init runs again.
- Finding: on this image cloud-init sets the hostname but **does not create users** from
  `users:`; `dots` never appeared. `bootcmd` does run (as root, every boot), so a one-off
  `bootcmd` script created `dots` (key-only, password `*`, passwordless sudo), installed the Mac's
  `id_ed25519.pub`, and enabled `ssh`. An empty `ssh` file on the boot partition also switches SSH
  on (consumed at boot). The bootstrap was removed from `user-data` once login worked.
- Pi `eth0` set to a static `192.168.1.20/24`, no default route (`ipv4.never-default yes`), so it
  stops waiting for DHCP and never takes the internet route from Wi-Fi.
- Mac `~/.ssh/config`:

  ```
  Host pi codex
      HostName 192.168.1.20
      HostKeyAlias codex.local
      User dots
      IdentityFile ~/.ssh/id_ed25519
      IdentitiesOnly yes
  ```

- Clock: no RTC battery and no network, so the Pi booted at 2026-09-14; set from the Mac
  (`sudo date -u -s ...`). Timezone `America/Los_Angeles`.

## 3. Internet for the Pi: Wi-Fi fails (an Option B finding)

- `wlan0` was `unavailable` until the Wi-Fi country was set (`raspi-config nonint do_wifi_country US`).
- `MissionRobotics` is **WPA3-only** (SAE, FT/SAE). The password was entered by Anthony via
  `nmcli --ask`. Every attempt fails: `CTRL-EVENT-ASSOC-REJECT bssid=00:00:00:00:00:00
  status_code=16` (local authentication timeout), then NM asks for new secrets. Requiring PMF
  (`802-11-wireless-security.pmf 3`) did not help. Wi-Fi chip: BCM4345/6, firmware 7.45.265
  (2023-08-29).
- Status 16 here looks the same for a wrong password and for an AP the chip's WPA3 firmware can't
  handshake with (e.g. H2E-only); not yet separated.
- Why it matters: under Option B the Ethernet port belongs to the robot, so Wi-Fi is the Pi's only
  internet. A Pi that can't join a WPA3-only venue network needs a WPA2 network, a USB Wi-Fi
  adapter or a second Ethernet port.

## 4. Inventory

| Item | Value |
|---|---|
| Board | Raspberry Pi 5 Model B Rev 1.1, 4 cores |
| RAM | 4 GB (+2 GB zram swap) |
| Storage | 117 GB microSD, 6.7 GB used |
| OS | Raspberry Pi OS (desktop), Debian 13.7 trixie, `graphical.target` |
| Kernel | `6.18.50+rpt-rpi-2712 #1 SMP PREEMPT` (stock, not PREEMPT_RT) |
| Bootloader | 2026-01-21 |
| Temperature | 45.0 °C idle; `throttled=0x0` |
| Cooler | `pwmfan` hwmon present (fan header in use: Active Cooler or case fan); 0 rpm at idle |
| `eth0` | 1000 Mb/s, full duplex, link detected (meets Option B's bar) |
| USB | Root hubs only: **no camera plugged in** |
| Running extras | Desktop session incl. `piwiz` (unfinished first-boot wizard), `wf-panel-pi`, `pcmanfm` |

## 5–8. Not done yet

Blocked on internet for the Pi: package updates, `rt-tests`/`stress-ng`, `uv`, the repo clone,
cyclictest (S3 preview), camera (S8, and no camera is attached) and OpenAI latency (S13).
