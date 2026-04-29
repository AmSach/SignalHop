# SignalHop — Acoustic Mesh Networking

**Mission:** Enable peer-to-device communication via sound waves. No internet, no WiFi, no Bluetooth required.

## What It Does
- **Acoustic modem**: Encode/decode data into sound (audible + ultrasonic)
- **Mesh networking**: Devices relay messages hop-by-hop using sound
- **AI noise cancellation**: Neural network compensates for echo/noise
- **Emergency beacon mode**: Broadcast SOS with location via chirp
- **IoT bridge**: Use as a last-resort comm channel for sensors in remote areas

## Architecture
```
SignalHop/
├── core/              # Core acoustic modem engine
│   ├── modem.py       # Encode/decode binary → sound waves
│   ├── mesh.py         # Peer discovery + routing
│   └── chirp.py        # Chirp sequence generation/detection
├── ai/                # ML models
│   ├── noise_cancel.py      # Audio enhancement
│   └── demodulate.py        # Neural demodulator
├── hardware/          # Embedded/robotics layer
│   ├── pico/          # Raspberry Pi Pico interface
│   └── esp32/         # ESP32 acoustic driver
├── web/               # Web interface (demo)
│   └── demo.html      # Browser-based acoustic chat demo
├── utils/             # Utilities
│   ├── spectrum.py    # Spectrogram analysis
│   └── beacon.py      # Emergency beacon
└── docs/              # Docs + blog posts
```

## Tech Stack
- Python (modem + ML)
- C++ (ESP32/Pico embedded)
- JavaScript (Web Audio API demo)
- TensorFlow Lite (on-device ML)
- Web Audio API (browser demo)

## Status
🟡 Building foundation — v0.1