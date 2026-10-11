"""Exercise 2 - Flask API for the student pass predictor.

Accepts 'study_hours' and 'attendance_percentage', validates them, and returns
both values plus the model prediction as JSON.

Note: the saved model was trained on study hours only, so attendance is
validated and returned but does not change the prediction unless the model is
retrained with attendance as an input feature.
"""

import math
import os
from pathlib import Path

import joblib
from flask import Flask, jsonify, request

MODEL_PATH = Path(__file__).resolve().parent / "student_pass_model.joblib"

ATTENDANCE_NOTE = (
    "Prediction is based on study hours only. Attendance does not influence "
    "the result unless the model is retrained with attendance as an input "
    "feature."
)

app = Flask(__name__)
model = joblib.load(MODEL_PATH)  # Loaded once at startup; no retraining needed.


def error(message: str):
    """Return a 400 Bad Request response with a JSON error message."""
    return jsonify({"error": message}), 400


@app.get("/")
def home():
    return jsonify(
        {
            "message": "Student Pass Prediction API",
            "endpoint": "/predict",
            "method": "POST",
        }
    )


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return error("Please provide a valid JSON object.")

    # --- study_hours: required numeric value between 0 and 24 ----------------
    try:
        study_hours = float(data["study_hours"])
    except (KeyError, TypeError, ValueError):
        return error("Provide numeric 'study_hours'.")
    if not math.isfinite(study_hours) or not 0 <= study_hours <= 24:
        return error("'study_hours' must be between 0 and 24.")

    # --- attendance_percentage: required numeric value between 0 and 100 -----
    try:
        attendance = float(data["attendance_percentage"])
    except (KeyError, TypeError, ValueError):
        return error("Provide numeric 'attendance_percentage'.")
    if not math.isfinite(attendance) or not 0 <= attendance <= 100:
        return error("'attendance_percentage' must be between 0 and 100.")

    # --- Prediction ----------------------------------------------------------
    prediction = int(model.predict([[study_hours]])[0])

    return jsonify(
        {
            "study_hours": study_hours,
            "attendance_percentage": attendance,
            "prediction": prediction,
            "result": "PASS" if prediction == 1 else "FAIL",
            "note": ATTENDANCE_NOTE,
        }
    )


if __name__ == "__main__":
    app.run(debug=False, port=int(os.environ.get("PORT", "5000")))
