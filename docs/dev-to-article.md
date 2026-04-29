---
title: "SignalHop: Peer-to-Peer Communication via Sound Waves"
slug: signalhop-acoustic-mesh-networking
published_at: 2026-04-28T12:00:00Z
description: "Build an acoustic mesh networking stack that lets devices communicate using sound — no WiFi, no Bluetooth, no internet required. FSK modem, chirp sync, hop routing, AI noise cancellation."
tags: [networking, embedded, iot, sound, mesh-networking, esp32, arduino, python]
canonical_url: https://github.com/AmSach/SignalHop
---
# SignalHop: Peer-to-Peer Communication via Sound Waves

*What if your phone could talk to another phone without WiFi, Bluetooth, or cellular?*

## The Problem

Natural disasters knock out infrastructure. Conferences kill WiFi. Remote sensors need data offload in places with zero connectivity. Bluetooth pairing is painful and range-limited. LoRa is great but requires specialized hardware.

**Sound waves work everywhere.** Every phone has a speaker and microphone. Every embedded board can drive a transducer. And sound doesn't need a license.

## What SignalHop Is

SignalHop is a complete acoustic mesh networking stack:

- **Acoustic modem**: Encodes data as FSK at 18kHz/20kHz — ultrasonic, non-intrusive, 500 symbols/sec
- **Mesh networking**: Peer discovery via chirp beacons, hop-by-hop routing with TTL
- **AI noise cancellation**: Spectral subtraction + CNN-based demodulator for real-world robustness
- **Cross-platform**: Python (server/ML), C++ (ESP32), JavaScript (browser demo)

## How the Modem Works

**FSK (Frequency Shift Keying)** maps binary digits to tones:
- `0` → 18,000 Hz (low tone)
- `1` → 20,000 Hz (high tone)

At 500 symbols/sec, you get ~62 bytes/sec of raw throughput — enough for text messages, sensor readings, and emergency beacons.

**Frame structure:**
```
[Preamble: 4 up-chirps] [Header: 48 bytes] [Payload: ≤256 bytes] [CRC32: 4 bytes]
```

The chirp preamble lets receivers synchronize — even in noisy environments.

## Real-World Use Cases

| Scenario | Why Sound? |
|----------|-----------|
| **Emergency communication** | Infrastructure is down. Phones work. Sound propagates through walls. |
| **Underground/cave exploration** | RF is blocked. Sound isn't. |
| **IoT sensor offload** | Cheap ultrasonic transducers, no WiFi needed |
| **Disaster relief** | Rapid peer discovery via chirp beacons, no coordination required |

## The Tech Stack

- `core/modem.py` — FSK acoustic modem engine (Python)
- `core/mesh.py` — Peer discovery + hop routing
- `ai/noise_cancel.py` — Spectral subtraction + CNN denoiser
- `hardware/esp32/acoustic_modem.cpp` — ESP32 I2S acoustic driver
- `web/demo.html` — Browser-based acoustic chat demo (Web Audio API)

## Quick Start

```bash
git clone https://github.com/AmSach/SignalHop
cd SignalHop
python3 core/modem.py
```

Open `web/demo.html` in a browser to see the acoustic modem in action.

## What's Next

- TensorFlow Lite model for CNN-based demodulation
- Raspberry Pi Pico port
- GPS integration for location-annotated emergency beacons
- Mobile apps (React Native acoustic stack)

---

*Sound is the oldest protocol. We've just updated the spec.*