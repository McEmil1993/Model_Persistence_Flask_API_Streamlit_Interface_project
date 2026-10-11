from math import isfinite
from pathlib import Path

import joblib
from flask import Flask, jsonify, request

MODEL_PATH = Path(__file__).resolve().parent / "learnlens_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError("Run train_model.py before starting the API.")

model = joblib.load(MODEL_PATH)
app = Flask(__name__)


@app.get("/")
def home():
    return jsonify({
        "name": "LearnLens Learning Progress API",
        "method": "POST",
        "endpoint": "/predict",
        "required_fields": ["study_hours", "attendance_percentage"]
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object."}), 400

    missing = [
        key for key in ("study_hours", "attendance_percentage")
        if key not in data
    ]
    if missing:
        return jsonify({"error": "Missing required fields.", "fields": missing}), 400

    try:
        hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({"error": "Both values must be numbers."}), 400

    if not isfinite(hours) or not 0 <= hours <= 24:
        return jsonify({"error": "study_hours must be a finite number from 0 to 24."}), 400
    if not isfinite(attendance) or not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be a finite number from 0 to 100."}), 400

    result = str(model.predict([[hours, attendance]])[0])
    return jsonify({
        "study_hours": hours,
        "attendance_percentage": attendance,
        "learning_progress": result,
        "message": "Prediction generated from a small synthetic classroom dataset."
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
