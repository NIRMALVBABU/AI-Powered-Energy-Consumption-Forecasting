# 🤖 AI-Powered Predictive Maintenance for IoT Devices

> An AI system that predicts failures in IoT devices and machinery **before they happen**, preventing costly downtime and improving operational efficiency.

---

## 📌 Project Overview

| Field | Details |
|-------|---------|
| **Project Title** | AI-Powered Predictive Maintenance for IoT Devices |
| **Objective** | Build an AI system that predicts machine failures using real-time IoT sensor data |
| **Industry Relevance** | Used by Siemens, Bosch, GE, IBM Watson IoT, AWS IoT, Microsoft Azure IoT, Tesla |
| **Approach** | Dataset-based virtual simulation (no physical hardware required) |

---

## 🔍 Problem Statement

Traditional maintenance is **reactive** — machines are fixed only after they break. This leads to:
- Unplanned downtime and production loss
- High emergency repair costs
- Safety risks for workers

**Predictive maintenance** uses AI and IoT sensor data (temperature, vibration, current) to detect early failure signals and act **before** breakdowns occur.

---

## 📊 Market Facts

- The global predictive maintenance market was valued at **$7.85 billion in 2022**, projected to reach **$60.13 billion by 2030** (CAGR: 29.5%)
- Reduces maintenance costs by **5–10%**, unplanned downtime by **15%**, and boosts labor productivity by **5–20%**
- Extends aging asset lifespan by **20%** and reduces safety/environmental risks by **14%**

**Real-world adoption:**
- **McDonald's** — AI predicts equipment failures (e.g., ice cream machines) across 43,000+ restaurants
- **Air France-KLM** — Google Cloud AI predicts aircraft maintenance needs
- **U.S. Navy (USS Fitzgerald)** — Monitors 10,000 sensor readings per second using AI

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.x |
| ML Framework | Scikit-learn, TensorFlow / Keras |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| API Deployment | Flask |
| Cloud/IoT | MQTT / Firebase / ESP8266 |
| Model Persistence | Joblib |

---

## 🔌 Hardware Components (for physical deployment)

| Component | Purpose |
|-----------|---------|
| Raspberry Pi / Arduino | Main controller — collects sensor data and runs AI |
| DHT11 / DHT22 Temperature Sensor | Detects overheating in machines |
| Vibration Sensor (SW-420) | Detects abnormal vibrations in motors |
| Current Sensor (ACS712) | Monitors power fluctuations |
| Wi-Fi Module (ESP8266 / ESP32) | Sends data to the cloud for real-time monitoring |
| Motor & Fan | Simulates a running machine to generate real data |

> **Note:** This project can be fully executed using datasets and virtual simulation without physical hardware.

---

## 📁 Folder Structure

```
AI-Predictive-Maintenance-IoT/
│
├── data/                  # Raw and processed sensor datasets
├── notebooks/             # Jupyter notebooks for EDA and model training
├── src/                   # Source Python scripts (modular)
│   ├── data_collection.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── predict.py
├── models/                # Saved trained model files (.pkl)
├── outputs/               # Generated graphs, reports, confusion matrices
├── images/                # Screenshots and visualizations for README
├── docs/                  # Additional documentation
├── app.py                 # Flask API for real-time prediction
├── main.py                # Entry point to run the full pipeline
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/AI-Predictive-Maintenance-IoT.git
cd AI-Predictive-Maintenance-IoT

# 2. Create a virtual environment
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`
```
pandas
numpy
scikit-learn
tensorflow
flask
matplotlib
seaborn
joblib
requests
```

---

## 🚀 How to Run

### Step 1 — Train the Model
```bash
python src/train_model.py
```

### Step 2 — Start the Flask Prediction API
```bash
python app.py
```

### Step 3 — Test a Prediction (via Postman or curl)
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"temperature": 75, "vibration": 4.5, "current": 12.3}'
```

**Expected Response:**
```json
{"Prediction": "Machine is running normally."}
```

---

## 🧠 Core Code

### 1. Sensor Data Collection (Simulated / Raspberry Pi)

```python
import Adafruit_DHT
import RPi.GPIO as GPIO
import spidev
import serial
import time
import requests

GPIO.setmode(GPIO.BCM)

# DHT11 Temperature Sensor
DHT_SENSOR = Adafruit_DHT.DHT11
DHT_PIN = 4

# SW-420 Vibration Sensor
VIBRATION_PIN = 17
GPIO.setup(VIBRATION_PIN, GPIO.IN)

# MCP3008 (Current Sensor via ADC)
spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1350000

def read_adc(channel):
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    data = ((adc[1] & 3) << 8) + adc[2]
    return data

def get_sensor_data():
    humidity, temperature = Adafruit_DHT.read(DHT_SENSOR, DHT_PIN)
    if temperature is None:
        temperature = 0
    vibration_detected = GPIO.input(VIBRATION_PIN)
    current_value = read_adc(0)
    return temperature, humidity, vibration_detected, current_value

def send_data_to_ai(temperature, vibration, current):
    api_url = "http://127.0.0.1:5000/predict"
    payload = {"temperature": temperature, "vibration": vibration, "current": current}
    response = requests.post(api_url, json=payload)
    if response.status_code == 200:
        prediction = response.json()["Prediction"]
        print(f"AI Prediction: {prediction}")

try:
    while True:
        temperature, humidity, vibration, current = get_sensor_data()
        print(f"Sensor Data: Temp={temperature}°C, Vibration={vibration}, Current={current} mA")
        send_data_to_ai(temperature, vibration, current)
        time.sleep(5)
except KeyboardInterrupt:
    GPIO.cleanup()
```

---

### 2. Model Training (`src/train_model.py`)

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load sensor dataset
data = pd.read_csv("data/iot_sensor_data.csv")

# Features and target
X = data[['temperature', 'vibration', 'current']]
y = data['failure']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model
joblib.dump(model, "models/predictive_maintenance_model.pkl")
print("Model saved successfully.")
```

---

### 3. Flask API for Real-Time Predictions (`app.py`)

```python
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("models/predictive_maintenance_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([[data["temperature"], data["vibration"], data["current"]]])
    prediction = model.predict(features)
    result = "Machine failure predicted!" if prediction[0] == 1 else "Machine is running normally."
    return jsonify({"Prediction": result})

if __name__ == '__main__':
    app.run(debug=True)
```

---

## 📈 Project Workflow

```
IoT Sensor Data
      ↓
Data Preprocessing (cleaning, normalization)
      ↓
Feature Engineering (temperature, vibration, current)
      ↓
Model Training (Random Forest / LSTM)
      ↓
Failure Prediction (failure / no failure)
      ↓
Alert Generation + Visualization Dashboard
```

---

## 📷 Expected Outputs

- Dataset preview (head, info, describe)
- Preprocessing output (cleaned data)
- Model training logs (accuracy, classification report)
- Prediction output per sensor reading
- Failure detection graph (time vs. sensor readings)
- Confusion matrix
- GitHub repository with all code and visuals

---

## 🗓️ 7-Day Build Plan

| Day | Task |
|-----|------|
| Day 1 | Environment setup, repo creation |
| Day 2 | Dataset loading and exploration |
| Day 3 | Data cleaning and preprocessing |
| Day 4 | Model training and evaluation |
| Day 5 | Flask API development |
| Day 6 | Visualization and output generation |
| Day 7 | GitHub upload, README polish, final commit |

---

## 🎯 Learning Outcomes

- Understanding IoT sensor data and predictive analytics
- Hands-on experience with scikit-learn and ML pipelines
- Building and deploying a Flask REST API
- Industry-aligned project for placements and internship portfolios
- GitHub project documentation and version control best practices

---

## 🏢 Industry Use Cases

- Manufacturing plant maintenance automation
- Factory floor equipment monitoring
- Power plant turbine health tracking
- Automotive assembly line diagnostics
- Aviation predictive servicing

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙋 Author

Built as a proof-of-work portfolio project for placements and internship opportunities.
