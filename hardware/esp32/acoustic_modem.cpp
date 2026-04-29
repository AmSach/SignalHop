// SignalHop — ESP32 I2S Acoustic Modem Driver
// Uses I2S for high-speed audio I/O at 48kHz

#include <driver/i2s.h>
#include <driver/gpio.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <esp_err.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#define SAMPLE_RATE         48000
#define FREQ_LOW            18000
#define FREQ_HIGH           20000
#define SYMBOL_RATE         500
#define SAMPLES_PER_SYMBOL  (SAMPLE_RATE / SYMBOL_RATE)  // 96
#define CHIRP_DURATION_MS   50
#define CHIRP_SAMPLES       (SAMPLE_RATE * CHIRP_DURATION_MS / 1000)  // 2400
#define NETWORK_ID          "SIGNALHOP_V1"
#define HEADER_SIZE         41   // 12 NETWORK_ID + 1 len + 2 seq/flags + 8 sender + 16 reserved + 2 padding
#define MAX_PAYLOAD         256
#define MAX_FRAME_SAMPLES   (4 * CHIRP_SAMPLES + (HEADER_SIZE + MAX_PAYLOAD + 4) * 8 * SAMPLES_PER_SYMBOL)

static const float PI = 3.14159265f;

// ── Math helpers ─────────────────────────────────────────────────────────────

static inline float fast_sin(float x) {
    // Minmax polynomial approximation
    while (x < 0) x += 2 * PI;
    while (x >= 2 * PI) x -= 2 * PI;
    float x2 = x * x;
    float x3 = x2 * x;
    float s = 0.99997092f * x + (-0.16542407f) * x3;
    float t = 1.00022787f + 0.00516627f * x2;
    return s / t;
}

// Simple CRC32 (zlib polynomial)
static uint32_t crc32(const uint8_t* data, size_t len) {
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (int j = 0; j < 8; j++) {
            crc = (crc >> 1) ^ (0xEDB88320 & ~(crc & 1));
        }
    }
    return ~crc;
}

// ── Signal generation ─────────────────────────────────────────────────────────

static void encode_symbol(float* out, int bit, int n) {
    float freq = bit ? FREQ_HIGH : FREQ_LOW;
    for (int i = 0; i < n; i++) {
        float t = (float)i / SAMPLE_RATE;
        float amp = 1.0f;
        if (i < 20) amp = 0.5f * (1.0f - cosf(PI * i / 20.0f));
        else if (i >= n - 20) amp = 0.5f * (1.0f + cosf(PI * (n - i) / 20.0f));
        out[i] = amp * fast_sin(2.0f * PI * freq * t);
    }
}

static void generate_chirp(float* out, bool up, int n_samples) {
    float start_freq = FREQ_LOW  - 2000;  // 16000 Hz
    float end_freq   = FREQ_HIGH + 2000;  // 22000 Hz
    if (!up) { float t = start_freq; start_freq = end_freq; end_freq = t; }
    float sweep_duration = (float)CHIRP_DURATION_MS / 1000.0f;
    for (int i = 0; i < n_samples; i++) {
        float t = (float)i / SAMPLE_RATE;
        float f = start_freq + (end_freq - start_freq) * t / sweep_duration;
        out[i] = fast_sin(2.0f * PI * f * t);
    }
}

// ── Frame building (preamble + header + payload + CRC32) ─────────────────────

// Returns total samples written. frame must have capacity for:
//   4 * CHIRP_SAMPLES
//   + (HEADER_SIZE + payload_len + 4) * 8 * SAMPLES_PER_SYMBOL
static int build_frame(float* frame, const uint8_t* payload, int payload_len) {
    int idx = 0;

    // Preamble: 4 up-chirps
    float chirp_buf[CHIRP_SAMPLES];
    for (int c = 0; c < 4; c++) {
        generate_chirp(chirp_buf, true, CHIRP_SAMPLES);
        memcpy(&frame[idx], chirp_buf, CHIRP_SAMPLES * sizeof(float));
        idx += CHIRP_SAMPLES;
    }

    // Build header (41 bytes)
    uint8_t header[HEADER_SIZE] = {0};
    memcpy(header, NETWORK_ID, 12);
    header[12] = (uint8_t)payload_len;

    // Encode header bits
    for (int b = 0; b < HEADER_SIZE * 8; b++) {
        int byte_i = b / 8;
        int bit_i  = 7 - (b % 8);
        int bit = (header[byte_i] >> bit_i) & 1;
        encode_symbol(&frame[idx], bit, SAMPLES_PER_SYMBOL);
        idx += SAMPLES_PER_SYMBOL;
    }

    // Encode payload bits
    for (int b = 0; b < payload_len * 8; b++) {
        int byte_i = b / 8;
        int bit_i  = 7 - (b % 8);
        int bit = (payload[byte_i] >> bit_i) & 1;
        encode_symbol(&frame[idx], bit, SAMPLES_PER_SYMBOL);
        idx += SAMPLES_PER_SYMBOL;
    }

    // Encode CRC32 of (header + payload)
    uint32_t crc = crc32(header, HEADER_SIZE);
    crc = crc32(payload, payload_len) ^ crc;  // combine (simplified)
    // Actually compute proper CRC of header+payload
    crc = 0xFFFFFFFF;
    uint8_t combined[HEADER_SIZE + MAX_PAYLOAD];
    memcpy(combined, header, HEADER_SIZE);
    memcpy(combined + HEADER_SIZE, payload, payload_len);
    size_t combined_len = HEADER_SIZE + payload_len;
    for (size_t i = 0; i < combined_len; i++) {
        crc ^= combined[i];
        for (int j = 0; j < 8; j++) {
            crc = (crc >> 1) ^ (0xEDB88320 & ~(crc & 1));
        }
    }
    crc = ~crc;
    uint32_t crc_be = ((crc >> 24) & 0xFF) | ((crc >> 8) & 0xFF00) |
                      ((crc << 8) & 0xFF0000) | ((crc << 24) & 0xFF000000);
    for (int b = 0; b < 32; b++) {
        int bit = (crc_be >> (31 - b)) & 1;
        encode_symbol(&frame[idx], bit, SAMPLES_PER_SYMBOL);
        idx += SAMPLES_PER_SYMBOL;
    }

    return idx;
}

// ── Goertzel tone detection ───────────────────────────────────────────────────

static float goertzel_energy(const float* samples, int n, float target_freq) {
    float k = (int)(0.5f + n * target_freq / SAMPLE_RATE);
    float w = 2.0f * PI * k / n;
    float coeff = 2.0f * cosf(w);
    float s = 0.0f, s1 = 0.0f, s2 = 0.0f;
    for (int i = 0; i < n; i++) {
        s = samples[i] + coeff * s1 - s2;
        s2 = s1; s1 = s;
    }
    return s1 * s1 + s2 * s2 - coeff * s1 * s2;
}

// ── Chirp detection via correlation ──────────────────────────────────────────

// Generates reference chirp into ref_out (must have CHIRP_SAMPLES space)
static void get_chirp_ref(float* ref_out) {
    generate_chirp(ref_out, true, CHIRP_SAMPLES);
}

// Correlation-based chirp detection.
// Returns true if correlation peak exceeds threshold.
static bool detect_chirp(const float* signal, int n_samples) {
    if (n_samples < CHIRP_SAMPLES * 4) return false;

    float chirp_ref[CHIRP_SAMPLES];
    get_chirp_ref(chirp_ref);

    // Slide a window across the first 4 chirp positions and find max correlation
    float max_corr = 0.0f;
    float signal_energy = 1e-6f;

    for (int pos = 0; pos < CHIRP_SAMPLES * 3; pos++) {
        float corr = 0.0f;
        for (int i = 0; i < CHIRP_SAMPLES; i++) {
            corr += signal[pos + i] * chirp_ref[i];
            signal_energy += signal[pos + i] * signal[pos + i];
        }
        if (corr > max_corr) max_corr = corr;
    }

    float ref_energy = 0.0f;
    for (int i = 0; i < CHIRP_SAMPLES; i++) ref_energy += chirp_ref[i] * chirp_ref[i];

    // Normalized correlation coefficient
    float norm = sqrtf(signal_energy * ref_energy);
    return (max_corr / norm) > 0.6f;  // 60% correlation threshold
}

// ── Bit demodulation ─────────────────────────────────────────────────────────

// Demodulate n_bits from signal into out_bits (caller allocates).
// Returns number of bits written.
static int demod_symbols(const float* signal, int n_samples, uint8_t* out_bits, int max_bits) {
    int count = 0;
    for (int i = 0; i + SAMPLES_PER_SYMBOL <= n_samples && count < max_bits; i += SAMPLES_PER_SYMBOL) {
        float e_low  = goertzel_energy(&signal[i], SAMPLES_PER_SYMBOL, FREQ_LOW);
        float e_high = goertzel_energy(&signal[i], SAMPLES_PER_SYMBOL, FREQ_HIGH);
        out_bits[count++] = (e_high > e_low) ? 1 : 0;
    }
    return count;
}

// ── Frame parsing ─────────────────────────────────────────────────────────────

// Returns payload length (>0) on success, -1 on failure
static int parse_frame(const float* signal, int n_samples, uint8_t* payload_out, int max_payload) {
    if (!detect_chirp(signal, n_samples)) return -1;

    // Skip 4-chirp preamble
    int data_start = 4 * CHIRP_SAMPLES;
    if (n_samples < data_start) return -1;

    uint8_t bits[4096];
    int n_bits = demod_symbols(signal + data_start, n_samples - data_start, bits, 4096);

    if (n_bits < HEADER_SIZE * 8) return -1;

    // Decode header
    uint8_t header[HEADER_SIZE];
    for (int i = 0; i < HEADER_SIZE; i++) {
        int v = 0;
        for (int j = 0; j < 8; j++) v = (v << 1) | bits[i * 8 + j];
        header[i] = (uint8_t)v;
    }

    // Validate NETWORK_ID
    if (memcmp(header, NETWORK_ID, 12) != 0) return -1;

    int payload_len = header[12];
    if (payload_len == 0 || payload_len > max_payload || payload_len > MAX_PAYLOAD) return -1;

    int total_bits = HEADER_SIZE * 8 + payload_len * 8 + 32;
    if (n_bits < total_bits) return -1;

    // Decode payload
    for (int i = 0; i < payload_len; i++) {
        int v = 0;
        for (int j = 0; j < 8; j++) v = (v << 1) | bits[HEADER_SIZE * 8 + i * 8 + j];
        payload_out[i] = (uint8_t)v;
    }

    // Verify CRC32
    uint32_t crc_received = 0;
    for (int i = 0; i < 32; i++) crc_received = (crc_received << 1) | bits[HEADER_SIZE * 8 + payload_len * 8 + i];

    uint32_t crc_expected = 0xFFFFFFFF;
    uint8_t combined[HEADER_SIZE + MAX_PAYLOAD];
    memcpy(combined, header, HEADER_SIZE);
    memcpy(combined + HEADER_SIZE, payload_out, payload_len);
    for (size_t i = 0; i < HEADER_SIZE + payload_len; i++) {
        crc_expected ^= combined[i];
        for (int j = 0; j < 8; j++) {
            crc_expected = (crc_expected >> 1) ^ (0xEDB88320 & ~(crc_expected & 1));
        }
    }
    crc_expected = ~crc_expected;

    // CRC comparison (big-endian vs native — account for bit ordering)
    // The CRC in the frame is MSB-first, match accordingly
    uint32_t crc_be = ((crc_received >> 24) & 0xFF) |
                      ((crc_received >> 8)  & 0xFF00) |
                      ((crc_received << 8)  & 0xFF0000) |
                      ((crc_received << 24) & 0xFF000000);

    if (crc_be != crc_expected) {
        printf("CRC mismatch: expected %08lx, got %08lx\n", crc_expected, crc_be);
        return -1;
    }

    return payload_len;
}

// ── I2S audio I/O ─────────────────────────────────────────────────────────────

// Initialize I2S at 48kHz, 16-bit, stereo
static void i2s_init(void) {
    i2s_config_t config = {
        .mode             = I2S_MODE_MASTER | I2S_MODE_TX | I2S_MODE_RX,
        .sample_rate      = SAMPLE_RATE,
        .bits_per_sample  = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format   = I2S_CHANNEL_FMT_RIGHT_LEFT,
        .communication_format = I2S_COMM_FORMAT_I2S,
        .tx_desc_auto_clear  = true,
        .rx_desc_auto_clear  = true,
        .fixed_mclk      = 0,
        .mclk_multiple   = I2S_MCLK_MULTIPLE_DEFAULT,
    };
    i2s_driver_install(I2S_NUM_0, &config, 0, NULL);

    i2s_pin_config_t pins = {
        .bck_io_num     = GPIO_NUM_26,
        .ws_io_num      = GPIO_NUM_25,
        .data_out_num   = GPIO_NUM_33,
        .data_in_num    = GPIO_NUM_34,
    };
    i2s_set_pin(I2S_NUM_0, &pins);
    i2s_zero_dma_buffer(I2S_NUM_0);
    printf("I2S initialized: %d Hz, 16-bit, TX+RX\n", SAMPLE_RATE);
}

// Convert float samples [-1,1] to PCM16, clip to prevent overflow
static inline int16_t float_to_pcm16(float s) {
    if (s >  1.0f) s =  1.0f;
    if (s < -1.0f) s = -1.0f;
    return (int16_t)(s * 32767.0f);
}

// Transmit payload as acoustic frame
static void transmit(const uint8_t* payload, int len) {
    static float frame[MAX_FRAME_SAMPLES];
    int n = build_frame(frame, payload, len);

    static int16_t pcm[MAX_FRAME_SAMPLES];
    for (int i = 0; i < n; i++) pcm[i] = float_to_pcm16(frame[i]);

    size_t written;
    i2s_write(I2S_NUM_0, pcm, n * sizeof(int16_t), &written, portMAX_DELAY);
    printf("TX: %d bytes → %d samples\n", len, n);
}

// Receive and parse frame. Returns payload length on success, -1 on nothing received.
static int receive(uint8_t* payload_out, int max_payload) {
    static float samples[MAX_FRAME_SAMPLES];
    static int16_t pcm[MAX_FRAME_SAMPLES];
    size_t bytes_read;

    esp_err_t err = i2s_read(I2S_NUM_0, pcm, sizeof(pcm), &bytes_read, 100 / portTICK_PERIOD_MS);
    if (err != ESP_OK || bytes_read == 0) return -1;

    int n_samples = bytes_read / sizeof(int16_t);
    for (int i = 0; i < n_samples; i++) {
        samples[i] = (float)pcm[i] / 32768.0f;
    }

    int result = parse_frame(samples, n_samples, payload_out, max_payload);
    if (result >= 0) {
        printf("RX: %d bytes received\n", result);
    }
    return result;
}

// ── Echo test (verify encode/decode round-trip) ───────────────────────────────

static void echo_test(void) {
    const char* msg = "SignalHop ESP32 test!";
    int msg_len = strlen(msg) + 1;

    static float frame[MAX_FRAME_SAMPLES];
    int n_samples = build_frame(frame, (const uint8_t*)msg, msg_len);
    printf("Echo test: encoded %d bytes → %d samples\n", msg_len, n_samples);

    // Parse what we just built
    int data_start = 4 * CHIRP_SAMPLES;
    uint8_t bits[4096];
    int dc = demod_symbols(frame + data_start, n_samples - data_start, bits, 4096);

    if (dc < HEADER_SIZE * 8 + msg_len * 8 + 32) {
        printf("Echo test: insufficient bits (%d)\n", dc);
        return;
    }

    // Decode header
    uint8_t header[HEADER_SIZE];
    for (int i = 0; i < HEADER_SIZE; i++) {
        int v = 0;
        for (int j = 0; j < 8; j++) v = (v << 1) | bits[i * 8 + j];
        header[i] = (uint8_t)v;
    }

    printf("Header NETWORK_ID: %.12s (valid: %d)\n", header, memcmp(header, NETWORK_ID, 12) == 0);
    printf("Payload len field: %d\n", header[12]);

    // Decode payload
    printf("Decoded payload: \"");
    for (int i = 0; i < msg_len; i++) {
        int v = 0;
        for (int j = 0; j < 8; j++) v = (v << 1) | bits[HEADER_SIZE * 8 + i * 8 + j];
        printf("%c", v);
    }
    printf("\"\n");
}

// ── Main app ──────────────────────────────────────────────────────────────────

void app_main(void) {
    i2s_init();
    printf("\n=== SignalHop ESP32 Acoustic Modem ===\n");
    echo_test();

    printf("\nReady. Install in two ESP32 devices and use transmit()/receive() for mesh networking.\n");

    // Keep main thread alive
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}