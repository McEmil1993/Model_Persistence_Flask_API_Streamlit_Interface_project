from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)
MODEL_FILE = Path(__file__).with_name("student_pass_model.joblib")
model = joblib.load(MODEL_FILE)
FEATURES = ["study_hours", "attendance_percentage"]


def read_percentage(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("attendance_percentage must be a number from 0 to 100")
    if not 0 <= number <= 100:
        raise ValueError("attendance_percentage must be between 0 and 100")
    return number


@app.get("/")
def index():
    return jsonify({
        "message": "Student pass prediction API",
        "usage": "POST /predict with JSON: {study_hours, attendance_percentage}",
    })


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send a JSON object in the request body"}), 400

    try:
        study_hours = float(payload["study_hours"])
        attendance = read_percentage(payload["attendance_percentage"])
        if study_hours < 0:
            raise ValueError("study_hours must be zero or greater")
    except KeyError as exc:
        return jsonify({"error": f"Missing required field: {exc.args[0]}"}), 400
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400

    features = pd.DataFrame([[study_hours, attendance]], columns=FEATURES)
    passed = bool(model.predict(features)[0])
    confidence = float(model.predict_proba(features)[0].max())
    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": "Pass" if passed else "Needs improvement",
        "passed": passed,
        "confidence": round(confidence, 3),
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
