#include <ESP32Servo.h>
#include <WiFi.h>
#include <Firebase_ESP_Client.h>
#include <time.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/semphr.h>
#include <freertos/queue.h>
#include "addons/TokenHelper.h"
#include "addons/RTDBHelper.h"

/* ===== FIREBASE CONFIGURATION ===== */
#define WIFI_SSID "EintsWaveX"
#define WIFI_PASSWORD "EintsWaveX"
#define FIREBASE_HOST "iwss-iotxaixes-default-rtdb.firebaseio.com"
#define FIREBASE_DB_SECRET "PASTE_YOUR_DATABASE_SECRET_HERE"

FirebaseData fbdo;
FirebaseConfig config;
FirebaseAuth auth;

// Database references
String appStatusPath = "/app_status";
String latestInferencePath = "/inference_data/latest_inference";

/* ===== WEIGHT AVERAGING CONFIG ===== */
#define WEIGHT_SAMPLES 10
#define WEIGHT_SAMPLE_INTERVAL 200

/* ===== FREERTOS VARIABLES ===== */
TaskHandle_t WeightTaskHandle = NULL;
TaskHandle_t FirebaseTaskHandle = NULL;
TaskHandle_t ServoTaskHandle = NULL;

// Queues
QueueHandle_t xCommandQueue = NULL;

// Semaphores
SemaphoreHandle_t xWeightBufferSemaphore = NULL;
SemaphoreHandle_t xServoBusySemaphore = NULL;

// Shared variables
volatile bool currentAppStatus = false;
volatile bool weightBufferReady = false;
volatile float currentAverageWeight = 0.0;
volatile unsigned long pollInterval = 2900; // Default interval from your data
volatile bool servoBusy = false;

// Track last processed inference
String lastProcessedTimestamp = "";
unsigned long lastProcessedTime = 0;

/* ===== SERVO ===== */
Servo servo1;
Servo servo2;
Servo servo3;
const int servo1Pin = 26;
const int servo2Pin = 27;
const int servo3Pin = 14;

/* ===== WEIGHT BUFFER ===== */
float weightBuffer[WEIGHT_SAMPLES];
volatile int weightBufferIndex = 0;

/* ===== STRUCT FOR INFERENCE DATA ===== */
typedef struct {
  String category;
  String timestamp;
  float confidence;
  bool valid;
  bool processed;
} InferenceData_t;

/* ===== TASK FUNCTION PROTOTYPES ===== */
void WeightTask(void *parameter);
void FirebaseTask(void *parameter);
void ServoTask(void *parameter);
void connectToWiFi();
void initFirebase();
float calculateAverageWeight();
bool getLatestInference(InferenceData_t &data);
void updateInferenceWeight(float weight);
void markInferenceProcessed();

/* ===== SETUP ===== */
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n\n=== SINGLE INFERENCE FreeRTOS SYSTEM STARTING ===\n");

  // Initialize weight buffer
  for (int i = 0; i < WEIGHT_SAMPLES; i++) {
    weightBuffer[i] = 0.0;
  }

  // Initialize FreeRTOS objects
  xCommandQueue = xQueueCreate(5, sizeof(int));
  xWeightBufferSemaphore = xSemaphoreCreateMutex();
  xServoBusySemaphore = xSemaphoreCreateMutex();

  if (xCommandQueue == NULL || xWeightBufferSemaphore == NULL || xServoBusySemaphore == NULL) {
    Serial.println("ERROR: Failed to create FreeRTOS objects!");
    while (1);
  }

  // Initialize hardware
  servo1.attach(servo1Pin);
  servo2.attach(servo2Pin);
  servo3.attach(servo3Pin);
  servo1.write(0);
  servo2.write(0);
  servo3.write(0);

  Serial.println("Hardware initialized");

  // WiFi & Firebase
  connectToWiFi();
  initFirebase();

  // Create FreeRTOS tasks
  xTaskCreatePinnedToCore(
    WeightTask,
    "WeightTask",
    4096,
    NULL,
    2,
    &WeightTaskHandle,
    0);

  xTaskCreatePinnedToCore(
    FirebaseTask,
    "FirebaseTask",
    8192,
    NULL,
    2,
    &FirebaseTaskHandle,
    1);

  xTaskCreatePinnedToCore(
    ServoTask,
    "ServoTask",
    4096,
    NULL,
    2,
    &ServoTaskHandle,
    1);

  Serial.println("✓ FreeRTOS tasks created successfully");
  Serial.println("✓ System running on dual-core ESP32");
  Serial.println("✓ Single inference entry mode");
}

void loop() {
  // Simple status display
  static unsigned long lastStatus = 0;
  if (millis() - lastStatus > 1000) {
    lastStatus = millis();

    Serial.println("\n=== SYSTEM STATUS ===");
    Serial.print("App running: ");
    Serial.println(currentAppStatus ? "YES" : "NO");
    Serial.print("Poll interval: ");
    Serial.print(pollInterval);
    Serial.println("ms");
    Serial.print("Servo busy: ");
    Serial.println(servoBusy ? "YES" : "NO");

    if (xSemaphoreTake(xWeightBufferSemaphore, pdMS_TO_TICKS(100)) == pdTRUE) {
      Serial.print("Weight samples: ");
      Serial.print(weightBufferIndex);
      Serial.print("/");
      Serial.print(WEIGHT_SAMPLES);
      if (weightBufferReady) {
        Serial.print(" (READY - Avg: ");
        Serial.print(currentAverageWeight);
        Serial.print("g)");
      }
      Serial.println();
      xSemaphoreGive(xWeightBufferSemaphore);
    }

    Serial.print("Last processed: ");
    if (lastProcessedTimestamp != "") {
      Serial.print(lastProcessedTimestamp);
      Serial.print(" at ");
      Serial.print((millis() - lastProcessedTime) / 1000);
      Serial.println("s ago");
    } else {
      Serial.println("None");
    }

    Serial.print("Command queue: ");
    Serial.print(uxQueueMessagesWaiting(xCommandQueue));
    Serial.println("/5");
    Serial.println("===================\n");
  }

  vTaskDelay(pdMS_TO_TICKS(1000));
}

/* ===== TASK 1: WEIGHT READING ===== */
void WeightTask(void *parameter) {
  Serial.println("WeightTask started on Core 0");

  TickType_t xLastWakeTime = xTaskGetTickCount();

  for (;;) {
    // Simulated weight
    float weight = 100.0 + (rand() % 100) / 10.0;

    if (xSemaphoreTake(xWeightBufferSemaphore, pdMS_TO_TICKS(500)) == pdTRUE) {
      weightBuffer[weightBufferIndex] = weight;
      int tempIndex = weightBufferIndex;
      tempIndex++;
      weightBufferIndex = tempIndex;

      if (weightBufferIndex >= WEIGHT_SAMPLES) {
        currentAverageWeight = calculateAverageWeight();
        weightBufferReady = true;
        weightBufferIndex = 0;
      }

      xSemaphoreGive(xWeightBufferSemaphore);
    }

    vTaskDelayUntil(&xLastWakeTime, pdMS_TO_TICKS(WEIGHT_SAMPLE_INTERVAL));
  }
}

/* ===== TASK 2: FIREBASE POLLING ===== */
void FirebaseTask(void *parameter) {
  Serial.println("FirebaseTask started on Core 1");

  // Initial delay
  vTaskDelay(pdMS_TO_TICKS(5000));

  unsigned long lastPollTime = 0;
  unsigned long lastStatusCheck = 0;
  bool firebaseConnected = false;

  // Get initial app status and interval
  Serial.println("[FirebaseTask] Getting initial app status...");
  if (Firebase.RTDB.getJSON(&fbdo, appStatusPath)) {
    FirebaseJson &json = fbdo.jsonObject();
    FirebaseJsonData result;
    
    // Get app running status
    if (json.get(result, "on_running")) {
      currentAppStatus = result.boolValue;
      Serial.print("[FirebaseTask] Initial app status: ");
      Serial.println(currentAppStatus ? "ON" : "OFF");
    }
    
    // Get poll interval
    if (json.get(result, "interval")) {
      pollInterval = result.intValue;
      Serial.print("[FirebaseTask] Initial poll interval: ");
      Serial.print(pollInterval);
      Serial.println("ms");
    }
  }

  for (;;) {
    unsigned long currentTime = millis();

    // Check WiFi
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("[FirebaseTask] WiFi disconnected!");
      firebaseConnected = false;
      vTaskDelay(pdMS_TO_TICKS(5000));
      continue;
    }

    // Check Firebase connection periodically
    if (currentTime - lastStatusCheck > 1000) {
      lastStatusCheck = currentTime;

      // Check app status (running/not running)
      if (Firebase.RTDB.get(&fbdo, appStatusPath + "/on_running")) {
        bool newStatus = fbdo.boolData();

        if (newStatus != currentAppStatus) {
          currentAppStatus = newStatus;
          Serial.print("[FirebaseTask] App status: ");
          Serial.println(currentAppStatus ? "ON" : "OFF");
        }

        if (!firebaseConnected) {
          firebaseConnected = true;
          Serial.println("[FirebaseTask] ✓ Firebase connected");
        }
      } else {
        if (firebaseConnected) {
          Serial.print("[FirebaseTask] ✗ Connection lost: ");
          Serial.println(fbdo.errorReason());
        }
        firebaseConnected = false;
      }

      // Check for interval updates
      if (Firebase.RTDB.getInt(&fbdo, appStatusPath + "/interval")) {
        unsigned long newInterval = fbdo.intData();
        if (newInterval != pollInterval && newInterval >= 1000) {
          pollInterval = newInterval;
          Serial.print("[FirebaseTask] Poll interval updated: ");
          Serial.print(pollInterval);
          Serial.println("ms");
        }
      }
    }

    // Only proceed if connected and app is running
    if (!firebaseConnected || !currentAppStatus) {
      vTaskDelay(pdMS_TO_TICKS(1000));
      continue;
    }

    // Wait if servo is busy
    if (servoBusy) {
      vTaskDelay(pdMS_TO_TICKS(100));
      continue;
    }

    // Poll for new inferences based on interval (AFTER servo finishes)
    if (currentTime - lastPollTime > pollInterval) {
      lastPollTime = currentTime;
      
      Serial.println("[FirebaseTask] Polling for latest inference...");

      // Get the latest inference data
      InferenceData_t inference;
      if (getLatestInference(inference)) {
        // Check if it's valid and not processed yet
        if (inference.valid && !inference.processed && 
            inference.timestamp != lastProcessedTimestamp) {
          
          Serial.print("[FirebaseTask] New inference found: ");
          Serial.print(inference.category);
          Serial.print(" (");
          Serial.print(inference.confidence);
          Serial.println(")");

          // Send command based on category
          int command = 0;
          if (inference.category == "plastic") command = 1;
          else if (inference.category == "paper") command = 2;
          else if (inference.category == "metal") command = 3;

          if (command > 0) {
            if (xQueueSend(xCommandQueue, &command, 0) == pdTRUE) {
              Serial.print("[FirebaseTask] Sent command: ");
              Serial.println(command);

              // Update weight if available
              if (weightBufferReady) {
                updateInferenceWeight(currentAverageWeight);
              }

              // Mark as processed
              markInferenceProcessed();

              // Update tracking
              lastProcessedTimestamp = inference.timestamp;
              lastProcessedTime = currentTime;
            }
          }
        } else if (inference.processed) {
          // Update last processed timestamp even if already processed
          if (inference.timestamp != lastProcessedTimestamp) {
            lastProcessedTimestamp = inference.timestamp;
            Serial.print("[FirebaseTask] Already processed inference: ");
            Serial.println(inference.timestamp);
          }
        }
      }
    }

    // Handle manual commands
    if (Serial.available()) {
      char input = Serial.read();

      if (input == '1' || input == '2' || input == '3') {
        int command = input - '0';
        if (xQueueSend(xCommandQueue, &command, 0) == pdTRUE) {
          Serial.print("[FirebaseTask] Manual command: ");
          Serial.println(command);
        }
      } else if (input == 's') {
        Serial.println("\n[FirebaseTask] === STATUS ===");
        Serial.print("Firebase: ");
        Serial.println(firebaseConnected ? "Connected" : "Disconnected");
        Serial.print("App running: ");
        Serial.println(currentAppStatus ? "YES" : "NO");
        Serial.print("Poll interval: ");
        Serial.println(pollInterval);
        Serial.print("Last processed: ");
        Serial.println(lastProcessedTimestamp);
        Serial.print("Weight ready: ");
        Serial.println(weightBufferReady ? "YES" : "NO");
        Serial.print("Servo busy: ");
        Serial.println(servoBusy ? "YES" : "NO");
        Serial.println("==============\n");
      } else if (input == 'p') {
        // Poll now
        Serial.println("[FirebaseTask] Manual poll...");
        lastPollTime = 0;
      } else if (input == 'i') {
        // Check inference now
        Serial.println("[FirebaseTask] Checking latest inference...");
        InferenceData_t inference;
        if (getLatestInference(inference)) {
          Serial.println("=== LATEST INFERENCE ===");
          Serial.print("Category: ");
          Serial.println(inference.category);
          Serial.print("Confidence: ");
          Serial.println(inference.confidence);
          Serial.print("Timestamp: ");
          Serial.println(inference.timestamp);
          Serial.print("Valid: ");
          Serial.println(inference.valid ? "YES" : "NO");
          Serial.print("Processed: ");
          Serial.println(inference.processed ? "YES" : "NO");
          Serial.println("=====================\n");
        }
      } else if (input == 't') {
        // Write test inference
        Serial.println("[FirebaseTask] Writing test inference...");
        
        FirebaseJson json;
        json.set("category", "plastic");
        json.set("confidence", 0.85);
        json.set("valid", true);
        json.set("processed", false);

        // Create timestamp
        char timestamp[20];
        time_t now = time(nullptr);
        strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&now));
        json.set("timestamp", timestamp);

        if (Firebase.RTDB.setJSON(&fbdo, latestInferencePath, &json)) {
          Serial.println("[FirebaseTask] Test inference created");
        } else {
          Serial.print("[FirebaseTask] Failed: ");
          Serial.println(fbdo.errorReason());
        }
      }
    }

    vTaskDelay(pdMS_TO_TICKS(100));
  }
}

/* ===== TASK 3: SERVO CONTROL ===== */
void ServoTask(void *parameter) {
  Serial.println("ServoTask started on Core 1");

  int command = 0;

  for (;;) {
    if (xQueueReceive(xCommandQueue, &command, pdMS_TO_TICKS(50)) == pdTRUE) {
      // Set servo busy flag
      if (xSemaphoreTake(xServoBusySemaphore, pdMS_TO_TICKS(500)) == pdTRUE) {
        servoBusy = true;
        xSemaphoreGive(xServoBusySemaphore);
      }

      Serial.print("[ServoTask] Executing: ");
      Serial.println(command);

      switch (command) {
        case 1:  // Plastic
          servo2.write(0);
          servo3.write(0);
          for (int pos = 0; pos <= 180; pos++) {
            servo1.write(pos);
            vTaskDelay(pdMS_TO_TICKS(20));
          }
          servo1.write(0);
          break;

        case 2:  // Paper
          servo2.write(90);
          servo3.write(0);
          for (int pos = 0; pos <= 180; pos++) {
            servo1.write(pos);
            vTaskDelay(pdMS_TO_TICKS(20));
          }
          servo1.write(0);
          vTaskDelay(pdMS_TO_TICKS(1000));
          servo2.write(0);
          break;

        case 3:  // Metal
          servo2.write(90);
          servo3.write(90);
          for (int pos = 0; pos <= 180; pos++) {
            servo1.write(pos);
            vTaskDelay(pdMS_TO_TICKS(20));
          }
          servo1.write(0);
          vTaskDelay(pdMS_TO_TICKS(1000));
          servo2.write(0);
          servo3.write(0);
          break;
      }

      // Clear servo busy flag
      if (xSemaphoreTake(xServoBusySemaphore, pdMS_TO_TICKS(500)) == pdTRUE) {
        servoBusy = false;
        xSemaphoreGive(xServoBusySemaphore);
      }

      Serial.println("[ServoTask] Command completed");
      command = 0;
    }

    vTaskDelay(pdMS_TO_TICKS(50));
  }
}

/* ===== FIREBASE HELPER FUNCTIONS ===== */
bool getLatestInference(InferenceData_t &data) {
  if (!Firebase.RTDB.getJSON(&fbdo, latestInferencePath)) {
    // Path might not exist yet
    Serial.print("[FirebaseTask] No inference data: ");
    Serial.println(fbdo.errorReason());
    return false;
  }

  FirebaseJson &json = fbdo.jsonObject();
  FirebaseJsonData result;

  // Get category
  if (!json.get(result, "category")) {
    Serial.println("[FirebaseTask] No category field");
    return false;
  }
  data.category = result.stringValue;

  // Get timestamp
  if (json.get(result, "timestamp")) {
    data.timestamp = result.stringValue;
  } else {
    data.timestamp = "";
  }

  // Get confidence
  if (json.get(result, "confidence")) {
    data.confidence = result.floatValue;
  } else {
    data.confidence = 0.0;
  }

  // Get valid flag
  if (json.get(result, "valid")) {
    data.valid = result.boolValue;
  } else {
    data.valid = false;
  }

  // Get processed flag
  if (json.get(result, "processed")) {
    data.processed = result.boolValue;
  } else {
    data.processed = false;
  }

  return true;
}

void updateInferenceWeight(float weight) {
  String weightPath = latestInferencePath + "/weight";

  if (Firebase.RTDB.setFloat(&fbdo, weightPath, weight)) {
    Serial.print("[FirebaseTask] Weight updated: ");
    Serial.print(weight);
    Serial.println("g");
  } else {
    Serial.print("[FirebaseTask] Weight update failed: ");
    Serial.println(fbdo.errorReason());
  }
}

void markInferenceProcessed() {
  String processedPath = latestInferencePath + "/processed";

  if (Firebase.RTDB.setBool(&fbdo, processedPath, true)) {
    Serial.println("[FirebaseTask] Marked as processed");
  } else {
    Serial.print("[FirebaseTask] Failed to mark as processed: ");
    Serial.println(fbdo.errorReason());
  }
}

/* ===== NETWORK FUNCTIONS ===== */
void connectToWiFi() {
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");

  for (int i = 0; i < 30; i++) {
    if (WiFi.status() == WL_CONNECTED) {
      Serial.println("\n✓ WiFi connected");
      Serial.print("IP: ");
      Serial.println(WiFi.localIP());

      // Configure time for timestamps
      configTime(0, 0, "pool.ntp.org");
      return;
    }
    delay(500);
    Serial.print(".");
  }

  Serial.println("\n✗ WiFi failed!");
}

void initFirebase() {
  Serial.println("\nInitializing Firebase...");

  config.database_url = "https://" + String(FIREBASE_HOST);
  config.signer.tokens.legacy_token = FIREBASE_DB_SECRET;
  config.token_status_callback = tokenStatusCallback;

  // Reasonable timeouts
  config.timeout.serverResponse = 30 * 1000;
  config.timeout.wifiReconnect = 10 * 1000;

  fbdo.setResponseSize(16384);

  Firebase.begin(&config, &auth);
  Firebase.reconnectWiFi(true);

  Serial.print("Testing connection");
  for (int i = 0; i < 10; i++) {
    if (Firebase.ready()) {
      Serial.println("\n✓ Firebase ready");
      return;
    }
    delay(1000);
    Serial.print(".");
  }

  Serial.println("\n⚠ Firebase not ready (will retry in task)");
}

/* ===== WEIGHT FUNCTIONS ===== */
float calculateAverageWeight() {
  float sum = 0.0;
  int count = 0;

  for (int i = 0; i < WEIGHT_SAMPLES; i++) {
    if (weightBuffer[i] > 0.1) {
      sum += weightBuffer[i];
      count++;
    }
  }

  return count > 0 ? sum / count : 0.0;
}