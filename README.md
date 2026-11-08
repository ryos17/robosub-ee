# Stanford RoboSub — Electrical Repo

Electrical subsystem for the Stanford RoboSub AUV. It includes PCB designs (KiCad), firmware, Python tools, and research.

🌐 [stanfordrobosub.org](https://www.stanfordrobosub.org/)

## Repository layout

- **`platformio.ini`** — PlatformIO config for building/flashing the Teensy firmware. `src_dir` points at one sketch folder; change it to pick which firmware gets flashed (see comments in the file).
- **`gen1/`** — First hardware generation: PCBs only. Deprecated; the sub no longer exists.
- **`gen2/`** — Second (current) generation: PCBs and firmware for the audio and Teensy boards.
- **`gen3/`** — Third generation (in progress): ongoing designs of PCBs with Micro Breakout, Battery Boards, Power Distribution.
- **`imported_parts/`** — Shared KiCad libraries: symbols, footprints, 3D models, import script, and the JLC parts order log.
- **`misc/`** — Older and experimental material.
- **`research/`** — Research work (whisper-based intervehicle communication).

## Sub setup (Jetson Orin)

### Fix the system clock

The Jetson has no battery-backed RTC, so after being powered off its clock can be wrong. A wrong date breaks TLS, so Claude Code, `git`, `pip`, etc. fail to connect. Fix it by syncing over the internet:

```bash
timedatectl                      # check: "System clock synchronized" and the date
sudo timedatectl set-ntp true    # sync the time over the internet
sudo systemctl restart systemd-timesyncd
timedatectl                      # the date should now be correct
```

### Connect to eduroam (Stanford)

Replace `SUNETID` and the password with your own (Stanford eduroam uses PEAP / MSCHAPv2 with `sunetid@stanford.edu` as the identity):

```bash
nmcli connection add type wifi ifname wlP1p1s0 con-name eduroam ssid eduroam \
  wifi-sec.key-mgmt wpa-eap 802-1x.eap peap 802-1x.phase2-auth mschapv2 \
  802-1x.identity SUNETID@stanford.edu 802-1x.password 'YOUR_PASSWORD'
nmcli connection up eduroam
```

If an `eduroam` connection already exists, update it instead of adding a new one:

```bash
nmcli connection modify eduroam 802-1x.identity SUNETID@stanford.edu 802-1x.password 'YOUR_PASSWORD'
nmcli connection up eduroam
```

Don't commit your password. Afterwards, run the clock fix above if Claude still can't connect.
