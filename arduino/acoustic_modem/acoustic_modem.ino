// SignalHop — Arduino Ultrasonic Acoustic Modem
// Works with any Arduino-compatible board with enough RAM (Uno: 2KB RAM — very tight)
// Best with Arduino Nano 33 BLE or ESP32

#include <arduino.h>
#include <math.h>

#define SAMPLE_RATE  8000    // Arduino ADC/DAC limits (use external I2S for higher)
#define FREQ_LOW     8000    // Maximum feasible with Arduino PWM
#define FREQ_HIGH    10000
#define SYMBOL_RATE  500
#define SAMPLES_PER_SYMBOL (SAMPLE_RATE / SYMBOL_RATE)  // 16

static float phase_accum = 0.0;

void generate_tone(float* buf, int n, float freq_hz) {
  for (int i = 0; i < n; i++) {
    buf[i] = sin(2.0 * PI * freq_hz * i / SAMPLE_RATE);
  }
}

void encode_symbol(float* out, int bit) {
  float freq = bit ? FREQ_HIGH : FREQ_LOW;
  generate_tone(out, SAMPLES_PER_SYMBOL, freq);
}

void transmit_message(const char* msg) {
  // In a real implementation, transmit via I2S DAC
  // Arduino Uno can't really do this — need external audio shield
  Serial.println(msg);
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'T') {
      digitalWrite(LED_BUILTIN, HIGH);
      float sym[SAMPLES_PER_SYMBOL];
      encode_symbol(sym, 1);
      // Would need I2S output here
      digitalWrite(LED_BUILTIN, LOW);
    }
  }
}