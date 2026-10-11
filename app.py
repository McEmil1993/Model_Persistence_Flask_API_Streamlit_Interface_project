# Exercise 2 - Flask API (study hours + attendance percentage)
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request

MODEL_PATH = Path(__file__).parent.resolve() / "student_pass_model_v2.joblib"

app = Flask(__name__)
classifier = joblib.load(MODEL_PATH)   # uses study hours AND attendance


class InputError(Exception):
    """Raised when the request data is not valid."""


def parse_request(data):
    # Task 1: both study hours and attendance percentage are required
    if not isinstance(data, dict) or not data:
        raise InputError("Send study_hours and attendance_percentage")
    for key in ("study_hours", "attendance_percentage"):
        if key not in data:
            raise InputError(f"{key} is required")
    try:
        hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        raise InputError("Both values must be numbers")

    if hours < 0:
        raise InputError("study_hours cannot be negative")
    # Task 2: attendance must be between 0 and 100
    if attendance < 0 or attendance > 100:
        raise InputError("attendance_percentage must be between 0 and 100")
    return hours, attendance


@app.errorhandler(InputError)
def handle_input_error(err):
    return jsonify({"error": str(err)}), 400


@app.route("/")
def home():
    return jsonify({"message": "Student Pass Prediction API is running"})


@app.route("/predict", methods=["GET", "POST"])
def predict():
    raw = request.get_json(silent=True) if request.method == "POST" else request.args.to_dict()
    hours, attendance = parse_request(raw)

    # Task 3: return both values and the prediction in JSON
    guess = classifier.predict(np.array([[hours, attendance]]))[0]
    return jsonify({
        "study_hours": hours,
        "attendance_percentage": attendance,
        "prediction": "Pass" if guess == 1 else "Fail",
    })


if __name__ == "__main__":
    app.run(debug=True)
