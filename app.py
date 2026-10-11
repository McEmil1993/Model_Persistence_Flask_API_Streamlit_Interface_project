"""
Exercise 2 — Flask API
Run after training the model: python app.py
POST JSON to http://127.0.0.1:5000/predict
Example: {"study_hours": 6}
"""
from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "student_pass_model.joblib"

app = Flask(__name__)

def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Model file not found. Run 'python train_model.py' first."
        )
    return joblib.load(MODEL_FILE)

model = load_model()

@app.get("/")
def home():
    return jsonify({
        "message": "Student Pass Prediction API",
        "usage": "Send a POST request to /predict with study_hours."
    })

@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be valid JSON."}), 400

    if "study_hours" not in data:
        return jsonify({"error": "study_hours is required."}), 400

    try:
        study_hours = float(data["study_hours"])
    except (TypeError, ValueError):
        return jsonify({"error": "study_hours must be a numeric value."}), 400

    if study_hours < 0:
        return jsonify({"error": "study_hours cannot be negative."}), 400

    sample = pd.DataFrame([{"study_hours": study_hours}])
    predicted_class = int(model.predict(sample)[0])
    result = "PASS" if predicted_class == 1 else "FAIL"

    probability = None
    if hasattr(model, "predict_proba"):
        probability = round(float(model.predict_proba(sample)[0][predicted_class]), 4)

    return jsonify({
        "study_hours": study_hours,
        "prediction": result,
        "predicted_class": predicted_class,
        "confidence_for_predicted_class": probability
    })

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)