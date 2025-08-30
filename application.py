from flask import Flask, render_template, request
import joblib
import os
import numpy as np
import pandas as pd
from src.logger import get_logger

app = Flask(__name__)
logger = get_logger(__name__)


model = joblib.load('artifacts/models/model.pkl')
scaler = joblib.load('artifacts/processed/scaler.pkl')


FEATURE_NAMES = [
    'Operation_Mode',
    'Temperature_C',
    'Vibration_Hz',
    'Power_Consumption_kW',
    'Network_Latency_ms',
    'Packet_Loss_%',
    'Quality_Control_Defect_Rate_%',
    'Production_Speed_units_per_hr',
    'Predictive_Maintenance_Score',
    'Error_Rate_%',
    'Year',
    'Month',
    'Day',
    'Hour'
]

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = np.array([
            int(request.form['operation_mode']),
            float(request.form['temperature']),
            float(request.form['vibration']),
            float(request.form['power_consumption']),
            float(request.form['network_latency']),
            float(request.form['packet_loss']),
            float(request.form['quality_control']),
            float(request.form['production_speed']),
            float(request.form['maintenance_score']),
            float(request.form['error_rate']),
            int(request.form['year']),
            int(request.form['month']),
            int(request.form['day']),
            int(request.form['hour'])
        ]).reshape(1, -1)

        features_df = pd.DataFrame(features, columns=FEATURE_NAMES)

        features_scaled = scaler.transform(features_df)

        prediction = model.predict(features_scaled)[0]

        prediction_text = "Efficient" if prediction == 1 else "Inefficient"

        logger.info(f"Prediction made successfully: {prediction_text}")

        return render_template('index.html', prediction=prediction_text)

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return render_template('index.html', prediction="Error occurred during prediction")

if __name__ == '__main__':
    print("\n✨ Flask server is starting...")
    print("🌐 Access the application at: http://localhost:5000")
    print("🔍 Press CTRL+C to quit\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
