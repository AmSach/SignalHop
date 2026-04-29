# SignalHop ESP32 Setup Guide

This guide gets you running SignalHop's acoustic modem on an ESP32 with I2S audio.

## Hardware Requirements

- **ESP32 DevKit** (or any ESP32 with I2S pins)
- **I2S Microphone** — e.g. INMP441, SPH0645, or WM8960
- **I2S Speaker** or amplifier — e.g. MAX98357A, or any I2S DAC
- **Jumper wires** and a 3.3V/5V breadboard

### Pin Wiring (ESP32 DevKit v1)

| Component   | ESP32 Pin | Notes |
|-------------|-----------|-------|
| MIC WS      | GPIO 25   | I2S Word Select |
| MIC SCK     | GPIO 26   | I2S Bit Clock |
| MIC SD      | GPIO 34   | I2S Data In (input) |
| SPK BCK     | GPIO 26   | Shared with mic SCK |
| SPK WS      | GPIO 25   | Shared with mic WS |
| SPK DIN     | GPIO 33   | I2S Data Out |
| 3.3V        | 3.3V      | Power mic |
| GND         | GND       | Common ground |

> **Note:** GPIO 34–39 are input-only and cannot drive outputs. Make sure your mic is powered separately if using these pins.

## Software Setup

### 1. Install ESP-IDF

```bash
# Clone ESP-IDF
git clone --recursive https://github.com/espressif/esp-idf.git
cd esp-idf
./install.sh
source export.sh
```

### 2. Create Project

```bash
cd ~/esp
idf.py create-project signalhop_modem
cd signalhop_modem
```

### 3. Copy the Acoustic Modem Driver

Copy `hardware/esp32/acoustic_modem.cpp` into your project:

```bash
cp /path/to/SignalHop/hardware/esp32/acoustic_modem.cpp main/
```

### 4. Configure I2S Pins

Edit `main/CMakeLists.txt` and ensure it includes:

```cmake
idf_component_register(SRCS "acoustic_modem.cpp" INCLUDE_DIRS ".")
```

Set your pin configuration in `main/acoustic_modem.cpp` if different from defaults:
```cpp
i2s_pin_config_t pins = {
    .bck_io_num     = 26,
    .ws_io_num      = 25,
    .data_out_num   = 33,
    .data_in_num    = 34,
};
```

### 5. Build and Flash

```bash
idf.py set-target esp32
idf.py build
idf.py flash monitor
```

### 6. Verify Operation

The modem will print on startup:
```
I2S initialized: 48000 Hz, 16-bit, TX+RX

=== SignalHop ESP32 Acoustic Modem ===
Echo test: encoded 20 bytes → 60288 samples
Header NETWORK_ID: SIGNALHOP_V1 (valid: 1)
Payload len field: 20
Decoded payload: "SignalHop ESP32 test!"
```

## How It Works

1. **`build_frame()`** — Constructs the full acoustic frame:
   - 4 up-chirp preambles (16kHz→22kHz sweep, 50ms each)
   - 41-byte header ( NETWORK_ID + length + routing fields )
   - Encoded payload bits (FSK at 18kHz/20kHz)
   - CRC32 checksum (zlib polynomial, big-endian)

2. **`fast_sin()`** — Minmax polynomial sine approximation. No ARM DSP library needed, works on any ESP32.

3. **`goertzel_energy()`** — Single-tone energy detector for FSK demodulation. Efficient enough for real-time operation at 48kHz.

4. **`detect_chirp()`** — Correlation-based preamble detection. Normalizes against signal energy to avoid false positives.

5. **`parse_frame()`** — Validates NETWORK_ID, extracts payload length, verifies CRC32.

## Modem Parameters

| Parameter | Value |
|-----------|-------|
| Sample Rate | 48,000 Hz |
| Low Tone | 18,000 Hz (binary 0) |
| High Tone | 20,000 Hz (binary 1) |
| Symbol Rate | 500 symbols/sec |
| Samples/Symbol | 96 |
| Preamble | 4 up-chirps × 2400 samples |
| Frame Overhead | 41 bytes header + 4 bytes CRC |
| Max Payload | 256 bytes |

## Using in Your App

```cpp
#include "acoustic_modem.h"

// Transmit a message
const char* msg = "Hello from ESP32!";
transmit((uint8_t*)msg, strlen(msg) + 1);

// Receive a message (call in a loop or task)
uint8_t payload[256];
int len = receive(payload, sizeof(payload));
if (len > 0) {
    printf("Received: %.*s\n", len, payload);
}
```

## Two-Node Chat Demo

1. Flash both ESP32 devices with this firmware
2. Position them 1–3 meters apart, speakers facing each other
3. Each device auto-transmits its beacon every 5 seconds
4. Received messages are printed to UART

## Troubleshooting

**No chirp detected:**
- Check wiring (mic Data In → GPIO 34, not GPIO 35/36/39)
- Ensure mic is powered at 3.3V (not 5V — may damage the INMP441)
- Increase `detect_chirp()` threshold from 0.6 to 0.4

**Weak range:**
- The ESP32's internal DAC is noisy — use an I2S DAC like MAX98357A
- Ultrasonic transducers project better in open space
- Reduce ambient noise (close windows, turn off fans)

**I2S conflicts:**
- GPIO 26/25 are shared between mic and speaker BCLK/WS
- This is correct for this design (single bus, half-duplex TDM)
- If using separate I2S instances, use I2S_NUM_0 and I2S_NUM_1

## Next Steps

- Add TensorFlow Lite noise cancellation model (see `ai/noise_cancel.py`)
- Integrate with mesh routing in `core/mesh.py`
- Connect GPS module for location-annotated emergency beacons