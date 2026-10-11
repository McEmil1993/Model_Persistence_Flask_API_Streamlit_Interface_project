"""
app.py

Exercise 2 - Flask API
Accepts study_hours AND attendance_percentage, validates them, and returns
both values plus the model prediction as JSON.

Run:  python app.py        (run train_model.py first to create the model file)
Test: send POST http://127.0.0.1:5000/predict from Postman (Body > raw > JSON)
"""

import math
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

BASE_DIR = Path(__file__).parent
MODEL_FILE = BASE_DIR / "student_pass_model_attendance.joblib"
LABELS = {0: "Fail", 1: "Pass"}

app = Flask(__name__)

# Load the saved model ONCE when the server starts. No training happens here.
if not MODEL_FILE.exists():
    raise SystemExit(f"{MODEL_FILE.name} not found. Run train_model.py first.")
model = joblib.load(MODEL_FILE)


def read_number(data, field):
    """Return (value, error). Accepts ints/floats only, not text or true/false."""
    if field not in data:
        return None, f"'{field}' is required."
    value = data[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, f"'{field}' must be a number."
    try:
        value = float(value)
    except OverflowError:  # a huge integer such as 10**400 cannot become a float
        return None, f"'{field}' must be a finite number."
    if not math.isfinite(value):  # rejects NaN and Infinity
        return None, f"'{field}' must be a finite number."
    return value, None


@app.get("/")
def home():
    return jsonify(
        message="Student pass prediction API",
        usage="POST /predict with JSON: {study_hours, attendance_percentage}",
    )


@app.post("/predict")
def predict():
    # 415 = wrong content type, 400 = bad body. Different problems, different codes.
    if not request.is_json:
        return jsonify(errors=["Content-Type must be application/json (in Postman: Body > raw > JSON)."]), 415
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(errors=['Body must be a valid JSON object, e.g. {"study_hours": 6, "attendance_percentage": 90}.']), 400

    errors = []
    study_hours, err = read_number(data, "study_hours")
    if err:
        errors.append(err)
    elif study_hours < 0:
        errors.append("'study_hours' cannot be negative.")

    attendance, err = read_number(data, "attendance_percentage")
    if err:
        errors.append(err)
    elif not 0 <= attendance <= 100:
        errors.append("'attendance_percentage' must be between 0 and 100.")

    if errors:
        return jsonify(errors=errors), 400

    # Column names must match the ones used in train_model.py.
    features = pd.DataFrame(
        {"study_hours": [study_hours], "attendance_percentage": [attendance]}
    )
    predicted_class = int(model.predict(features)[0])
    pass_probability = float(model.predict_proba(features)[0][1])

    return jsonify(
        study_hours=study_hours,
        attendance_percentage=attendance,
        prediction=LABELS[predicted_class],
        pass_probability=round(pass_probability, 3),
    )


@app.errorhandler(404)
def not_found(_error):
    return jsonify(errors=["Route not found. Use POST /predict."]), 404


@app.errorhandler(405)
def method_not_allowed(_error):
    return jsonify(errors=["Method not allowed. Use POST /predict with a JSON body."]), 405


if __name__ == "__main__":
    # debug=True is handy while learning (auto-reload, readable errors) but must be
    # turned off for a real deployment. host 127.0.0.1 keeps it reachable only from this PC.
    app.run(host="127.0.0.1", port=5000, debug=True)
