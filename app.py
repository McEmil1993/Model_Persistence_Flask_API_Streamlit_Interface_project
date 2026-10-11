"""Flask API for student pass/fail predictions."""

import math
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request


MODEL_PATH = Path(__file__).resolve().parent / "student_pass_model.joblib"

app = Flask(__name__)
model = joblib.load(MODEL_PATH)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400

    required_fields = ("study_hours", "attendance_percentage")
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify(
            {
                "error": "Missing required inputs",
                "missing_fields": missing_fields,
            }
        ), 400

    try:
        hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({"error": "Both inputs must be numeric"}), 400

    if not math.isfinite(hours) or not math.isfinite(attendance):
        return jsonify({"error": "Inputs must be finite numbers"}), 400
    if not 0 <= hours <= 24:
        return jsonify({"error": "study_hours must be between 0 and 24"}), 400
    if not 0 <= attendance <= 100:
        return jsonify(
            {"error": "attendance_percentage must be between 0 and 100"}
        ), 400

    features = pd.DataFrame(
        [
            {
                "study_hours": hours,
                "attendance_percentage": attendance,
            }
        ]
    )
    prediction = int(model.predict(features)[0])
    pass_class_index = list(model.classes_).index(1)
    pass_probability = float(model.predict_proba(features)[0][pass_class_index])

    return jsonify(
        {
            "status": "success",
            "study_hours": hours,
            "attendance_percentage": attendance,
            "prediction": "Pass" if prediction == 1 else "Fail",
            "pass_probability": round(pass_probability, 4),
        }
    ), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
