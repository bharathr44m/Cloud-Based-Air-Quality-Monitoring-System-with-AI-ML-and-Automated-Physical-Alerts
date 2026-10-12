/*
 * Stage 3: ESP32 Air Quality Monitoring Firmware
 * Project: Cloud-Based Air Quality Monitoring System with Automated Physical Alert
 *
 * Hardware Pinout:
 * - MQ135: GPIO34 (ADC1_CH6 via 10k + 15k voltage divider)
 * - DHT11: GPIO16
 * - ST7735 TFT: CS=15, RESET=2, DC=4, MOSI=23, SCK=18
 * - Active Buzzer: GPIO17 (3.3V VCC, Active HIGH)
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ST7735.h>
#include <SPI.h>
#include <DHT.h>
#include <ArduinoJson.h>

// ====================================================================
// USER CONFIGURATION (FILL IN YOUR CREDENTIALS & PREFERENCES)
// ====================================================================
const char* WIFI_SSID     = "a";       // <--- WiFi SSID
const char* WIFI_PASSWORD = "123abc123";   // <--- WiFi Password

const char* SUPABASE_URL  = "https://ppvtqsyhiuabvxjtxtyp.supabase.co/rest/v1/sensor_data";
const char* SUPABASE_KEY  = "sb_publishable_DJxaBgoKqZaKefyxaZJ8jQ_UwPO3br6";

#define AQI_ALARM_THRESHOLD 650.0f  // AQI PPM threshold to sound local buzzer (Normal air ~400 PPM)
#define BUZZER_ACTIVE_HIGH  false   // Active LOW buzzer: LOW = Sound ON, HIGH = Quiet

// ====================================================================
// HARDWARE PIN DEFINITIONS
// ====================================================================
#define MQ135_PIN        34   // Input-only pin (ADC1)
#define DHTPIN           16   // DHT11 Data Pin
#define DHTTYPE          DHT11
#define BUZZER_PIN       5    // Active Buzzer Pin

// ST7735 TFT Pins (Hardware VSPI)
#define TFT_CS           15
#define TFT_RST          2
#define TFT_DC           4
#define TFT_MOSI         23
#define TFT_SCLK         18

// ====================================================================
// SYSTEM CONSTANTS & TIMING INTERVALS (millis)
// ====================================================================
const unsigned long SENSOR_INTERVAL   = 2000;   // Read sensors every 2 sec
const unsigned long DISPLAY_INTERVAL  = 1000;   // Refresh display every 1 sec
const unsigned long SUPABASE_INTERVAL = 12000;  // POST to Supabase every 12 sec

// ====================================================================
// GLOBAL OBJECTS & STATE VARIABLES
// ====================================================================
Adafruit_ST7735 tft = Adafruit_ST7735(TFT_CS, TFT_DC, TFT_RST);
DHT dht(DHTPIN, DHTTYPE);

unsigned long lastSensorTime   = 0;
unsigned long lastDisplayTime  = 0;
unsigned long lastSupabaseTime = 0;

float currentAQI = 0.0f;
float currentTemp = 0.0f;
float currentHumidity = 0.0f;
bool buzzerTriggered = false;

// ====================================================================
// HELPER FUNCTIONS
// ====================================================================

// MQ135 ADC to PPM conversion with Voltage Divider adjustment (10k + 15k)
float readMQ135PPM() {
    int rawADC = analogRead(MQ135_PIN);
    
    // ESP32 ADC: 0 - 4095 for 0 - 3.3V
    float voltageAtPin = (rawADC / 4095.0f) * 3.3f;
    
    // Voltage divider ratio: V_pin = V_sensor * (15k / (10k + 15k)) = V_sensor * 0.6
    // Therefore V_sensor = V_pin / 0.6
    float sensorVoltage = voltageAtPin / 0.6f;
    
    // Approximate PPM curve mapping for MQ135 Air Quality PPM
    // (Standard ambient air 400ppm baseline to 1000+ ppm high contamination)
    float ppm = 400.0f + (sensorVoltage * 250.0f);
    if (ppm < 0) ppm = 0;
    return ppm;
}

// Draw static UI background once to avoid flicker
void initDisplayLayout() {
    tft.initR(INITR_BLACKTAB);  // Standard 1.8" ST7735 screen tab
    tft.setRotation(1);         // Landscape orientation
    tft.fillScreen(ST7735_BLACK);
    
    // Header Banner
    tft.fillRect(0, 0, 160, 20, ST7735_BLUE);
    tft.setTextColor(ST7735_WHITE);
    tft.setTextSize(1);
    tft.setCursor(15, 6);
    tft.print("AIR QUALITY MONITOR");

    // Labels (Drawn once)
    tft.setTextColor(ST7735_GREEN);
    tft.setCursor(8, 30);
    tft.print("AQI PPM:");

    tft.setCursor(8, 55);
    tft.print("Temp   :");

    tft.setCursor(8, 80);
    tft.print("Humidity:");

    tft.setCursor(8, 105);
    tft.print("WiFi   :");

    // Divider Line (16-bit RGB565 Dark Grey: 0x7BEF)
    tft.drawFastHLine(0, 98, 160, 0x7BEF);
}

// Update value fields without full screen redraw (flicker-free)
void updateDisplayValues() {
    tft.setTextSize(1);

    // 1. AQI PPM Value
    if (currentAQI >= AQI_ALARM_THRESHOLD) {
        tft.setTextColor(ST7735_RED, ST7735_BLACK);
    } else {
        tft.setTextColor(ST7735_YELLOW, ST7735_BLACK);
    }
    tft.setCursor(80, 30);
    tft.print(currentAQI, 1);
    tft.print(" ppm   ");

    // 2. Temperature Value
    tft.setTextColor(ST7735_CYAN, ST7735_BLACK);
    tft.setCursor(80, 55);
    tft.print(currentTemp, 1);
    tft.print(" C   ");

    // 3. Humidity Value
    tft.setTextColor(ST7735_CYAN, ST7735_BLACK);
    tft.setCursor(80, 80);
    tft.print(currentHumidity, 1);
    tft.print(" %   ");

    // 4. WiFi Status
    tft.setCursor(80, 105);
    if (WiFi.status() == WL_CONNECTED) {
        tft.setTextColor(ST7735_GREEN, ST7735_BLACK);
        tft.print("Connected   ");
    } else {
        tft.setTextColor(ST7735_RED, ST7735_BLACK);
        tft.print("Connecting..");
    }
}

// Connect or maintain WiFi connection
void checkWiFiConnection() {
    if (WiFi.status() == WL_CONNECTED) return;

    Serial.print("[WiFi] Connecting to: ");
    Serial.println(WIFI_SSID);

    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    int timeout = 0;
    while (WiFi.status() != WL_CONNECTED && timeout < 20) { // Wait up to 10 seconds
        delay(500);
        Serial.print(".");
        timeout++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\n[WiFi] Connected successfully!");
        Serial.print("[WiFi] IP Address: ");
        Serial.println(WiFi.localIP());
    } else {
        Serial.print("\n[WiFi] Connection timed out. Status code: ");
        Serial.println(static_cast<int>(WiFi.status()));
        Serial.println("[WiFi] Check that the hotspot is 2.4 GHz and credentials match exactly.");

        int networkCount = WiFi.scanNetworks();
        Serial.printf("[WiFi] Nearby networks found: %d\n", networkCount);
        for (int networkIndex = 0; networkIndex < networkCount; networkIndex++) {
            Serial.printf(
                "[WiFi] %s | RSSI %d dBm | channel %d | encryption %d\n",
                WiFi.SSID(networkIndex).c_str(),
                WiFi.RSSI(networkIndex),
                WiFi.channel(networkIndex),
                static_cast<int>(WiFi.encryptionType(networkIndex))
            );
        }
        WiFi.scanDelete();
    }
}

// Send HTTPS POST to Supabase REST API
void postToSupabase() {
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[Supabase] Skipped POST: No WiFi connection");
        return;
    }

    WiFiClientSecure client;
    client.setInsecure(); // Skip SSL certificate verification for lower RAM footprint

    HTTPClient http;
    if (http.begin(client, SUPABASE_URL)) {
        http.addHeader("Content-Type", "application/json");
        http.addHeader("apikey", SUPABASE_KEY);
        http.addHeader("Authorization", String("Bearer ") + SUPABASE_KEY);
        http.addHeader("Prefer", "return=minimal");

        // Build JSON Payload using ArduinoJson
        StaticJsonDocument<200> doc;
        doc["aqi_ppm"]          = currentAQI;
        doc["temperature"]      = currentTemp;
        doc["humidity"]         = currentHumidity;
        doc["buzzer_triggered"] = buzzerTriggered;

        String jsonPayload;
        serializeJson(doc, jsonPayload);

        Serial.print("[Supabase] POSTing payload: ");
        Serial.println(jsonPayload);

        int httpResponseCode = http.POST(jsonPayload);

        if (httpResponseCode > 0) {
            Serial.printf("[Supabase] Response code: %d\n", httpResponseCode);
        } else {
            Serial.printf("[Supabase] POST failed, error: %s\n", http.errorToString(httpResponseCode).c_str());
        }

        http.end();
    } else {
        Serial.println("[Supabase] Unable to connect to server endpoint");
    }
}

// ====================================================================
// SETUP FUNCTION
// ====================================================================
void setup() {
    Serial.begin(115200);
    delay(500);
    Serial.println("\n--- Starting Air Quality ESP32 System ---");

    // Initialize Pins
    pinMode(BUZZER_PIN, OUTPUT);
    digitalWrite(BUZZER_PIN, BUZZER_ACTIVE_HIGH ? LOW : HIGH);
    pinMode(MQ135_PIN, INPUT);

    // Initialize Sensors & Display
    dht.begin();
    initDisplayLayout();

    // Start WiFi
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);
    checkWiFiConnection();
}

// ====================================================================
// MAIN LOOP (NON-BLOCKING millis() TIMING)
// ====================================================================
void loop() {
    unsigned long currentMillis = millis();

    // Loop Task 1: Sensor Readings (Every 2 seconds)
    if (currentMillis - lastSensorTime >= SENSOR_INTERVAL) {
        lastSensorTime = currentMillis;

        // 1. Read MQ135
        currentAQI = readMQ135PPM();

        // 2. Read DHT11 with NaN protection
        float tempVal = dht.readTemperature();
        float humVal  = dht.readHumidity();

        if (!isnan(tempVal)) {
            currentTemp = tempVal;
        } else {
            Serial.println("[DHT11] Warning: Temperature read error (NaN), retaining previous value");
        }

        if (!isnan(humVal)) {
            currentHumidity = humVal;
        } else {
            Serial.println("[DHT11] Warning: Humidity read error (NaN), retaining previous value");
        }

        // 3. Immediate local alert check
        if (currentAQI >= AQI_ALARM_THRESHOLD) {
            digitalWrite(BUZZER_PIN, BUZZER_ACTIVE_HIGH ? HIGH : LOW);
            buzzerTriggered = true;
            Serial.println("[ALERT] AQI Threshold EXCEEDED! Buzzer ACTIVE.");
        } else {
            digitalWrite(BUZZER_PIN, BUZZER_ACTIVE_HIGH ? LOW : HIGH);
            buzzerTriggered = false;
        }
    }

    // Loop Task 2: Update TFT Screen (Every 1 second)
    if (currentMillis - lastDisplayTime >= DISPLAY_INTERVAL) {
        lastDisplayTime = currentMillis;
        updateDisplayValues();
    }

    // Loop Task 3: POST to Supabase (Every 12 seconds)
    if (currentMillis - lastSupabaseTime >= SUPABASE_INTERVAL) {
        lastSupabaseTime = currentMillis;
        checkWiFiConnection();
        postToSupabase();
    }
}
