"""
Exercise 2 - Flask API

Endpoint:  /predict   (POST with JSON, or GET with query parameters)

Inputs
    study_hours             number, >= 0
    attendance_percentage   number, between 0 and 100 (inclusive)

Output (JSON)
    study_hours, attendance_percentage, prediction (0/1), result ("Pass"/"Fail"),
    pass_probability

The API uses student_pass_attendance_model.joblib, which was trained on BOTH
features by train_model.py, so attendance genuinely affects the prediction.

Run:  python train_model.py     (once, to create the model files)
      python app.py
"""

import math
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_pass_attendance_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH.name}. Run 'python train_model.py' first."
    )

# Loaded ONCE when the server starts - the model is never retrained here.
model = joblib.load(MODEL_PATH)

app = Flask(__name__)


def parse_number(payload, field, errors):
    """Return payload[field] as a finite float, or None (and record an error)."""
    value = payload.get(field)

    if value is None or value == "":
        errors.append(f"'{field}' is required.")
        return None
    # In Python, True/False are ints, so reject them explicitly.
    if isinstance(value, bool):
        errors.append(f"'{field}' must be a number.")
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"'{field}' must be a number.")
        return None
    if not math.isfinite(number):  # blocks NaN and Infinity
        errors.append(f"'{field}' must be a finite number.")
        return None
    return number


@app.route("/", methods=["GET"])
def index():
    return jsonify(
        message="Student pass prediction API",
        usage="POST /predict with JSON: {\"study_hours\": 6, \"attendance_percentage\": 80}",
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    # JSON body for POST; query string (?study_hours=6&...) works too.
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        payload = request.values

    errors = []
    study_hours = parse_number(payload, "study_hours", errors)
    attendance = parse_number(payload, "attendance_percentage", errors)

    # Validation rules
    if study_hours is not None and study_hours < 0:
        errors.append("'study_hours' must be 0 or greater.")
    if attendance is not None and not 0 <= attendance <= 100:
        errors.append("'attendance_percentage' must be between 0 and 100.")

    if errors:
        return jsonify(errors=errors), 400

    features = pd.DataFrame(
        {"study_hours": [study_hours], "attendance_percentage": [attendance]}
    )
    prediction = int(model.predict(features)[0])
    pass_probability = float(model.predict_proba(features)[0][1])

    return jsonify(
        study_hours=study_hours,
        attendance_percentage=attendance,
        prediction=prediction,
        result="Pass" if prediction == 1 else "Fail",
        pass_probability=round(pass_probability, 4),
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)