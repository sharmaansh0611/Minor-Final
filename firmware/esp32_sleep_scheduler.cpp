/* esp32_sleep_scheduler.cpp
 * Firmware skeleton for ESP32 deep-sleep + TinyML policy.
 * This is a template / stub; it prints flow and where to insert TFLite Micro code.
 */

#include <Arduino.h>
#include "esp_sleep.h"

// Use RTC data to persist small state across deep-sleep
RTC_DATA_ATTR int boot_count = 0;

void setup() {
  Serial.begin(115200);
  delay(100);
  ++boot_count;
  Serial.printf("Boot #%d\n", boot_count);

  // 1) Read sensors (pseudo)
  float sensor0 = 22.0; // placeholder
  int sensor1 = 0; // placeholder

  // 2) Run local TinyML inference (pseudo placeholder)
  // Insert TensorFlow Lite Micro inference here and decide `sleep_seconds`
  int sleep_seconds = 60; // default

  Serial.printf("Deciding to sleep for %d s\n", sleep_seconds);

  // 3) Configure esp32 deep sleep
  esp_sleep_enable_timer_wakeup((uint64_t)sleep_seconds * 1000000ULL);
  Serial.println("Going to deep sleep now.");
  esp_deep_sleep_start();
}

void loop() {
  // shouldn't reach here
}
