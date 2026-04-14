# ⚡ AI-Powered Energy Consumption Forecasting

> Forecast electricity usage in homes and buildings using Machine Learning to support smart cities and climate tech.

---

## 📌 Problem Statement

| Problem | Solution |
|---|---|
| Power grids fail to balance supply and demand, causing blackouts | AI predicts when and how much energy will be consumed so supply can be planned |
| Buildings and factories use energy inefficiently during peak hours | Model identifies usage trends to optimize operations |
| People and businesses overpay due to unpredictable usage | AI forecasts help avoid peak-time penalties |
| Overuse of energy increases carbon emissions | Better planning supports sustainable energy and net-zero goals |
| Traditional monitoring is manual, slow, and error-prone | AI automates predictions in real-time, reducing human error |

---

## 🧰 Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.9+ |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn (MLPRegressor) |
| Model Persistence | Joblib |
| API Deployment | Flask |
| Environment | Google Colab / Jupyter Notebook |

---

## 🗂️ Folder Structure

```
AI-Energy-Forecasting/
│
├── data/                   # Raw and processed datasets
│   └── energy.csv
├── notebooks/              # Jupyter/Colab notebooks
│   └── energy_forecasting.ipynb
├── src/                    # Core Python modules
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   └── evaluate.py
├── models/                 # Saved trained models
│   └── energy_forecast_model.pkl
├── outputs/                # Generated graphs and result files
├── images/                 # Screenshots for README
├── docs/                   # Project documentation
├── app.py                  # Flask prediction API
├── main.py                 # Entry point to run full pipeline
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.9 or above
- pip

### Step 1 — Clone the repository
```bash
git clone https://github.com/your-username/AI-Energy-Forecasting.git
cd AI-Energy-Forecasting
```

### Step 2 — Create and activate a virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### `requirements.txt`
```
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
flask
```

---

## 🚀 How to Run

### Run the full ML pipeline
```bash
python main.py
```

### Launch the Flask prediction API
```bash
python app.py
```

### Test the API (Postman or curl)
**URL:** `http://127.0.0.1:5000/predict`  
**Method:** POST  
**Body (JSON):**
```json
{
  "hour": 14,
  "day": 2
}
```
**Response:**
```json
{
  "predicted_energy": 3.87
}
```

---

## 📐 Project Architecture

```
[ Raw Energy CSV ]
        │
        ▼
[ Data Loading & Resampling ]  ── pandas resample to hourly
        │
        ▼
[ Preprocessing ]  ── fill missing values, clean outliers
        │
        ▼
[ Feature Engineering ]  ── extract hour, day_of_week
        │
        ▼
[ Model Training ]  ── MLPRegressor (64x64 hidden layers)
        │
        ▼
[ Evaluation ]  ── MAE, RMSE, R² Score
        │
        ▼
[ Model Saved ]  ── energy_forecast_model.pkl (joblib)
        │
        ▼
[ Flask API ]  ── /predict endpoint → real-time JSON response
        │
        ▼
[ Visualization ]  ── Actual vs Predicted plots saved to /outputs
```

---

## 💻 Core Code

### Step 1 — Load & Visualize Dataset
```python
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data/energy.csv', parse_dates=['Datetime'], index_col='Datetime')
data = data.resample('H').mean()
data = data.fillna(method='ffill')

data['Energy'].plot(figsize=(15, 5), title="Energy Consumption Over Time")
plt.savefig('outputs/energy_trend.png')
plt.show()
```

### Step 2 — Feature Engineering
```python
data['hour'] = data.index.hour
data['day'] = data.index.dayofweek
```

### Step 3 — Train the MLP Model
```python
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

X = data[['hour', 'day']]
y = data['Energy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae:.4f}")
```

### Step 4 — Save the Trained Model
```python
import joblib

joblib.dump(model, 'models/energy_forecast_model.pkl')
print("Model saved successfully.")
```

### Step 5 — Flask Prediction API (`app.py`)
```python
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('models/energy_forecast_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([[data['hour'], data['day']]])
    prediction = model.predict(features)
    return jsonify({'predicted_energy': round(float(prediction[0]), 4)})

if __name__ == '__main__':
    app.run(debug=True)
```

---

## 📊 Expected Outputs

| Output | Description |
|---|---|
| `outputs/energy_trend.png` | Historical energy usage line chart |
| `outputs/actual_vs_predicted.png` | Model prediction accuracy graph |
| `outputs/metrics.txt` | MAE, RMSE, R² score |
| `models/energy_forecast_model.pkl` | Saved trained model |
| Flask API response | Real-time JSON predictions |

---

## 📅 7-Day Build Plan

| Day | Task | Commit Message |
|---|---|---|
| Day 1 | Project setup, folder structure, virtual env | `feat: initialize project structure` |
| Day 2 | Load and explore dataset | `feat: add dataset loading and EDA` |
| Day 3 | Data cleaning and preprocessing | `feat: add preprocessing pipeline` |
| Day 4 | Feature engineering and model training | `feat: train MLP forecasting model` |
| Day 5 | Model evaluation and metrics | `feat: add evaluation metrics` |
| Day 6 | Visualization and output graphs | `feat: add visualization outputs` |
| Day 7 | Flask API + GitHub polish + README | `feat: deploy prediction API and finalize README` |

---

## 🌍 Real-World Impact

- **Google** saved 40% of data center cooling energy using AI forecasting
- The global AI energy forecasting market is projected to reach **$60 Billion by 2030**
- The U.S. Department of Energy has funded over **$100 million** in AI-based energy R&D
- **70% of India's power loss** is attributed to poor forecasting and mismanagement
- Smart cities worldwide are being built on AI energy management systems

---

## 🎓 Learning Outcomes

- Time-series data preprocessing with Pandas
- Feature engineering from datetime indices
- Training and evaluating a neural network regressor
- Saving and loading ML models with Joblib
- Building and testing a REST API with Flask
- Structuring and publishing an industry-level GitHub project

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

Dataset sourced from publicly available smart grid energy consumption logs.  
Built as a portfolio project to demonstrate applied machine learning in the clean-tech domain.
