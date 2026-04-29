// SignalHop — ESP32 I2S Acoustic Modem Driver
// Uses I2S for high-speed audio output at 48kHz

#include <driver/i2s.h>
#include <math.h>

#define SAMPLE_RATE   48000
#define FREQ_LOW      18000
#define FREQ_HIGH     20000
#define SYMBOL_RATE   500
#define SAMPLES_PER_SYMBOL (SAMPLE_RATE / SYMBOL_RATE)  // 96

static const float PI = 3.14159265;

// Generate FSK tone at given frequency
void generate_tone(float* buf, int n, float freq_hz) {
    for (int i = 0; i < n; i++) {
        buf[i] = arm_sin_f32(2.0 * PI * freq_hz * i / SAMPLE_RATE);
    }
}

// Generate one symbol (bit) as audio samples
void encode_symbol(float* out, int bit) {
    float freq = bit ? FREQ_HIGH : FREQ_LOW;
    generate_tone(out, SAMPLES_PER_SYMBOL, freq);
    // Cosine taper on edges
    for (int i = 0; i < 10; i++) {
        float taper = 0.5 * (1.0 - cos(PI * i / 10));
        out[i] *= taper;
        out[SAMPLES_PER_SYMBOL - 1 - i] *= taper;
    }
}

// Build complete frame: preamble + header + payload + crc
void build_frame(float* frame, const uint8_t* payload, int payload_len) {
    int idx = 0;

    // Preamble: 4 up-chirps
    for (int c = 0; c < 4; c++) {
        float start = FREQ_LOW - 2000;
        float end = FREQ_HIGH + 2000;
        for (int i = 0; i < SAMPLES_PER_SYMBOL * 5; i++) {
            float t = (float)i / SAMPLE_RATE;
            float freq = start + (end - start) * t / 0.05f;
            frame[idx++] = arm_sin_f32(2.0 * PI * freq * t);
        }
    }

    // Header: NETWORK_ID (12) + len/seq/ttl (13) + reserved (16) = 41 bytes
    // For now, skip header and just encode payload for demo

    // Payload bits
    for (int b = 0; b < payload_len * 8; b++) {
        int byte_i = b / 8;
        int bit_i = 7 - (b % 8);
        int bit = (payload[byte_i] >> bit_i) & 1;
        encode_symbol(&frame[idx], bit);
        idx += SAMPLES_PER_SYMBOL;
    }
}

// Goertzel algorithm for single-tone energy detection
float goertzel_energy(const float* samples, int n, float target_freq) {
    float k = (int)(0.5f + n * target_freq / SAMPLE_RATE);
    float w = 2.0 * PI * k / n;
    float coeff = 2.0f * cos(w);

    float s = 0, s1 = 0, s2 = 0;
    for (int i = 0; i < n; i++) {
        s = samples[i] + coeff * s1 - s2;
        s2 = s1;
        s1 = s;
    }
    return s1 * s1 + s2 * s2 - coeff * s1 * s2;
}

// Demodulate bits from audio samples
int demod_symbols(const float* signal, int n_samples, uint8_t* out_bits, int max_bits) {
    int n_sym = SAMPLES_PER_SYMBOL;
    int count = 0;

    for (int i = 0; i + n_sym <= n_samples && count < max_bits; i += n_sym) {
        float e_low = goertzel_energy(&signal[i], n_sym, FREQ_LOW);
        float e_high = goertzel_energy(&signal[i], n_sym, FREQ_HIGH);
        out_bits[count++] = (e_high > e_low) ? 1 : 0;
    }
    return count;
}

// I2S configuration
void i2s_init() {
    i2s_config_t config = {
        .mode = I2S_MODE_MASTER | I2S_MODE_TX,
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
        .communication_format = I2S_COMM_FORMAT_I2S,
        .tx_desc_auto_clear = true,
    };
    i2s_driver_install(I2S_NUM_0, &config, 0, NULL);

    i2s_pin_config_t pins = {
        .bck_io_num = GPIO_NUM_26,
        .ws_io_num = GPIO_NUM_25,
        .data_out_num = GPIO_NUM_33,
        .data_in_num = GPIO_NUM_34,
    };
    i2s_set_pin(I2S_NUM_0, &pins);
}

// Transmit a frame
void transmit(const uint8_t* payload, int len) {
    static float frame[48000];  // 1 second max
    build_frame(frame, payload, len);
    size_t bytes_written;
    i2s_write(I2S_NUM_0, frame, sizeof(frame), &bytes_written, portMAX_DELAY);
}

// Receive and demodulate
int receive(float* samples, uint8_t* out_bits) {
    size_t bytes_read;
    i2s_read(I2S_NUM_0, samples, sizeof(float) * 48000, &bytes_read, portMAX_DELAY);
    int n_samples = bytes_read / sizeof(float);
    return demod_symbols(samples, n_samples, out_bits, 4096);
}

// Echo test
void echo_test() {
    uint8_t msg[] = "SignalHop test!";
    float frame[48000];
    build_frame(frame, msg, sizeof(msg));

    uint8_t bits[4096];
    int count = demod_symbols(frame, sizeof(frame)/sizeof(float), bits, 4096);
    printf("Demodulated %d bits\n", count);
}