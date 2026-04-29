#include <Arduino.h>
/*
 * SignalHop — ESP32 Acoustic Driver
 * Handles ultrasound transmission/reception using I2S audio interface.
 * 
 * Wiring:
 *   ESP32 I2S DOUT → Amplifier → Speaker
 *   ESP32 I2S DIN  ← MEMS Microphone (SPM1423 or similar)
 *   GND → Common ground with amp/mic
 *
 * Protocol:
 *   - FSK modem: 18kHz (bit 0), 20kHz (bit 1)
 *   - 500 symbols/sec
 *   - Chirp sequence for sync (up-chirp, 50ms)
 *   - Max payload: 256 bytes
 */

// ─── Configuration ────────────────────────────────────────────────
#define SAMPLE_RATE     48000
#define SYMBOL_RATE     500
#define CARRIER_LOW     18000
#define CARRIER_HIGH    20000
#define CHIRP_DURATION_MS    50
#define PREAMBLE_CHIRPS 4
#define MAX_PAYLOAD     256
#define MAX_HOPS        8
#define GUARD_SAMPLES   (SAMPLE_RATE / SYMBOL_RATE * 0.1)

// ─── LED Status ─────────────────────────────────────────────────────
#define LED_TX          2   // TX activity
#define LED_RX          4   // RX activity
#define LED_MESH        18  // Mesh networking active
#define LED_ERROR       19  // Error condition

// ─── State Machine ─────────────────────────────────────────────────
enum ModemState { IDLE, SYNC, RECEIVING, TRANSMITTING, ERROR };
ModemState state = IDLE;

// ─── Globals ───────────────────────────────────────────────────────
static DRAM_ATTR float phase_acc = 0;
static DRAM_ATTR uint32_t tx_samples = 0;
static DRAM_ATTR uint32_t rx_samples = 0;

// ─── Frequency Generators ──────────────────────────────────────────
inline float gen_tone(float freq_hz, uint32_t sample_idx) {
    return sinf(2.0f * PI * freq_hz * sample_idx / SAMPLE_RATE);
}

// ─── Chirp Generator ───────────────────────────────────────────────
// Generates a linear frequency sweep (18kHz → 22kHz over CHIRP_DURATION_MS)
float* generate_up_chirp() {
    static float chirp[SAMPLE_RATE * CHIRP_DURATION_MS / 1000];
    uint32_t n = sizeof(chirp) / sizeof(float);
    for (uint32_t i = 0; i < n; i++) {
        float t = (float)i / SAMPLE_RATE;
        float freq = CARRIER_LOW + (2000.0f * t / (CHIRP_DURATION_MS / 1000.0f));
        chirp[i] = sinf(2.0f * PI * freq * t);
    }
    return chirp;
}

// ─── I2S Configuration ──────────────────────────────────────────────
void setup_i2s() {
    // I2S configuration for AC101 codec or UDA1334A
    // Note: ESP32 has two I2S controllers — using I2S_NUM_0
    I2S.begin(I2S_PHILIPS_MODE, SAMPLE_RATE, 16);
    pinMode(LED_TX, OUTPUT);
    pinMode(LED_RX, OUTPUT);
    pinMode(LED_MESH, OUTPUT);
    pinMode(LED_ERROR, OUTPUT);
    digitalWrite(LED_MESH, HIGH);  // Mesh LED on = networking active
}

// ─── TX: Encode payload to acoustic waveform ───────────────────────
void transmit_payload(const uint8_t* payload, size_t len) {
    if (len > MAX_PAYLOAD) return;

    state = TRANSMITTING;
    digitalWrite(LED_TX, HIGH);

    // Generate preamble (4 up-chirps)
    float* preamble = generate_up_chirp();
    size_t preamble_samples = sizeof(float) * (SAMPLE_RATE * CHIRP_DURATION_MS / 1000) * PREAMBLE_CHIRPS;

    // Encode payload as FSK
    // bit = 1 → CARRIER_HIGH, bit = 0 → CARRIER_LOW
    for (size_t b = 0; b < len * 8; b++) {
        uint8_t bit = (payload[b / 8] >> (7 - (b % 8))) & 1;
        float freq = bit ? CARRIER_HIGH : CARRIER_LOW;
        uint32_t symbol_samples = SAMPLE_RATE / SYMBOL_RATE;

        for (uint32_t s = 0; s < symbol_samples; s++) {
            float sample_val = sinf(2.0f * PI * freq * s / SAMPLE_RATE);
            // Apply cosine taper for smooth transitions
            float taper = 1.0f;
            if (s < 20) taper = s / 20.0f;
            if (s > symbol_samples - 20) taper = (symbol_samples - s) / 20.0f;

            // Would write to I2S buffer here
            // I2S.write(sample_val * 32767 * taper);
        }

        // Guard interval
        for (uint32_t g = 0; g < GUARD_SAMPLES; g++) {
            // I2S.write(0);  // silence
        }
    }

    digitalWrite(LED_TX, LOW);
    state = IDLE;
}

// ─── Goertzel Algorithm for Single-Tone Detection ────────────────────
class Goertzel {
public:
    float sample(int N, const float* samples, float target_freq) {
        float k = (float)N * target_freq / SAMPLE_RATE;
        int k_int = (int)(k + 0.5f);
        float omega = 2.0f * PI * k_int / N;
        float coeff = 2.0f * cosf(omega);

        float s = 0, s1 = 0, s2 = 0;
        for (int i = 0; i < N; i++) {
            s = samples[i] + coeff * s1 - s2;
            s2 = s1;
            s1 = s;
        }
        return s1 * s1 + s2 * s2 - coeff * s1 * s2;
    }
};

// ─── Chirp Sync: Cross-correlate with reference chirp ──────────────
bool detect_sync(const float* samples, size_t n, const float* chirp_ref, size_t chirp_len) {
    if (n < chirp_len) return false;
    float max_corr = 0;

    for (size_t i = 0; i < n - chirp_len; i++) {
        float corr = 0;
        for (size_t c = 0; c < chirp_len; c++) {
            corr += samples[i + c] * chirp_ref[c];
        }
        if (corr > max_corr) max_corr = corr;
    }

    // Threshold: correlation must exceed ~60% of max possible
    return max_corr > chirp_len * 0.6f;
}

// ─── Main Loop ─────────────────────────────────────────────────────
void loop() {
    switch (state) {
        case IDLE:
            // Listen for chirp preamble
            break;
        case SYNC:
            // Match chirp preamble, lock onto signal
            break;
        case RECEIVING:
            digitalWrite(LED_RX, HIGH);
            // Read I2S samples, run Goertzel, decode bits
            digitalWrite(LED_RX, LOW);
            break;
        case ERROR:
            digitalWrite(LED_ERROR, HIGH);
            delay(500);
            digitalWrite(LED_ERROR, LOW);
            state = IDLE;
            break;
    }
}

// ─── Boot Message ───────────────────────────────────────────────────
extern "C" void app_main() {
    setup_i2s();
    Serial.begin(115200);
    Serial.println("SignalHop ESP32 acoustic modem ready");
    Serial.printf("TX: %dHz/%dHz FSK @ %d sym/s\n", CARRIER_LOW, CARRIER_HIGH, SYMBOL_RATE);
}