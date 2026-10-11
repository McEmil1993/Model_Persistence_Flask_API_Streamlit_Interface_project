# app.py - Exercise 2: Flask API (study hours + attendance percentage)
import os
import joblib
from flask import Flask, jsonify, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "student_pass_model_v2.joblib")

api = Flask(__name__)
pass_model = joblib.load(MODEL_PATH)  # loaded once when the server starts


def check_inputs(payload):
    """Return (hours, attendance, error_message)."""
    if not payload:
        return None, None, "Please send JSON data"
    if "study_hours" not in payload or "attendance_percentage" not in payload:
        return None, None, "study_hours and attendance_percentage are required"
    try:
        hours = float(payload["study_hours"])
        attendance = float(payload["attendance_percentage"])
    except (TypeError, ValueError):
        return None, None, "Both values must be numbers"
    if hours < 0:
        return None, None, "study_hours cannot be negative"
    if not 0 <= attendance <= 100:
        return None, None, "attendance_percentage must be between 0 and 100"
    return hours, attendance, None


@api.route("/")
def index():
    return jsonify({"message": "Student Pass Prediction API is running"})


@api.route("/predict", methods=["POST"])
def predict():
    hours, attendance, error = check_inputs(request.get_json(silent=True))
    if error:
        return jsonify({"error": error}), 400

    result = pass_model.predict([[hours, attendance]])[0]
    return jsonify({
        "study_hours": hours,
        "attendance_percentage": attendance,
        "prediction": "Pass" if result == 1 else "Fail",
    })


if __name__ == "__main__":
    api.run(debug=True)