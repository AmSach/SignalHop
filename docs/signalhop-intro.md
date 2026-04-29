# SignalHop — Blog

**Category:** Networking / Embedded Systems / Open Source
**Date:** 2026-04-28
**Tags:** acoustic-modem, mesh-networking, esp32, arduino, iot, open-source, sound-communication, wireless

---

# SignalHop: Peer-to-Peer Communication via Sound Waves

*What if your phone could talk to another phone without WiFi, Bluetooth, or cellular?*

## The Problem

Natural disasters knock out infrastructure. Conferences kill WiFi. Remote sensors need data offload in places with zero connectivity. Bluetooth pairing is painful and range-limited. LoRa is great but requires specialized hardware.

**Sound waves work everywhere.** Every phone has a speaker and microphone. Every embedded board can drive a transducer. And sound doesn't need a license.

## What SignalHop Is

SignalHop is a complete acoustic mesh networking stack:

- **Acoustic modem**: Encodes data as FSK (Frequency Shift Keying) at 18kHz/20kHz — ultrasonic, non-intrusive, works in the 500 symbols/sec range
- **Mesh networking**: Peer discovery via chirp beacons, hop-by-hop routing with TTL
- **AI noise cancellation**: Spectral subtraction + CNN-based demodulator for real-world robustness
- **Cross-platform**: Python (server/ML), C++ (ESP32/Raspberry Pi Pico), JavaScript (browser demo)

```
[Your Phone]  --18kHz/20kHz FSK-->  [Another Phone]
     |                                    |
     └────── Sound waves (no internet) ──┘
```

## How the Modem Works

**FSK (Frequency Shift Keying)** maps binary digits to tones:
- `0` → 18,000 Hz (low tone)
- `1` → 20,000 Hz (high tone)

At 500 symbols/sec, you get ~62 bytes/sec of raw throughput — enough for text messages, sensor readings, and emergency beacons. Not Netflix. That's fine.

**Frame structure:**
```
[Preamble: 4 up-chirps] [Header: 48 bytes] [Payload: ≤256 bytes] [CRC32: 4 bytes]
```

The chirp preamble lets receivers synchronize — even in noisy environments. The header contains network ID, sequence number, TTL (time-to-live for mesh relay), and sender ID.

## Real-World Use Cases

| Scenario | Why Sound? |
|----------|-----------|
| **Emergency communication** | Infrastructure is down. Phones work. Sound propagates through walls. |
| **Underground/cave exploration** | RF is blocked. Sound isn't. |
| **IoT sensor offload** | Cheap ultrasonic transducers, no WiFi needed |
| **Disaster relief** | Rapid peer discovery via chirp beacons, no coordination required |
| **Privacy** | Sound doesn't go through walls well — naturally short-range |

## The Tech Stack

**Python core** (`core/modem.py`, `core/mesh.py`) — the acoustic modem engine and mesh routing layer.

**AI noise cancellation** (`ai/noise_cancel.py`) — spectral subtraction + a CNN-based demodulator. Trained on noisy acoustic environments, runs on-device with TensorFlow Lite.

**ESP32 driver** (`hardware/esp32/acoustic_modem.cpp`) — real-time acoustic transmit/receive using I2S audio interface. Handles the physical layer in C++ for timing precision.

**Browser demo** (`web/demo.html`) — Web Audio API-based acoustic chat demo. Encode text to ultrasound, visualize waveforms, simulate mesh peers.

## What Makes It Different

Most acoustic communication projects are toy demos. SignalHop is built for **production**:

1. **Proper framing** — CRC checksums, sequence numbers, TTL for mesh relay
2. **Noise robust** — Goertzel algorithm for single-tone detection, spectral subtraction denoising
3. **Mesh-ready** — Not just point-to-point; designed for multi-hop relay with loop detection
4. **Cross-platform** — From ESP32 to browser, with ML inference on the edge

## What's Next

- [ ] TensorFlow Lite model for CNN-based demodulation trained on real-world noise
- [ ] Raspberry Pi Pico port for lower-power embedded use
- [ ] Full mesh routing protocol (distance-vector with ETX metric)
- [ ] GPS integration for location-annotated emergency beacons
- [ ] Mobile apps (React Native acoustic stack)

## Try It

Clone the repo and run the Python modem:

```bash
git clone https://github.com/AmSach/SignalHop
cd SignalHop
python3 core/modem.py
```

Open `web/demo.html` in a browser to see the acoustic modem in action.

---

*Sound is the oldest protocol. We've just updated the spec.*