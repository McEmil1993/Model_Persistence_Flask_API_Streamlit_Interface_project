# Exercise 2 - Flask API (study hours + attendance percentage)
import os

import joblib
import pandas as pd
from flask import Blueprint, Flask, jsonify, request

BASE = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE, "student_pass_model_v2.joblib"))

api = Blueprint("api", __name__)

# Task 2: allowed range for each value
LIMITS = {
    "study_hours": (0, 24),
    "attendance_percentage": (0, 100),
}


def error(message):
    return jsonify({"error": message}), 400


@api.route("/")
def home():
    return jsonify({"message": "Student Pass Prediction API is running"})


@api.route("/predict", methods=["GET", "POST"])
def predict():
    source = request.get_json(silent=True) if request.method == "POST" else request.args.to_dict()
    if not isinstance(source, dict) or not source:
        return error("Send study_hours and attendance_percentage")

    values = {}
    # Task 1: accept both study hours and attendance percentage
    for field, (low, high) in LIMITS.items():
        if field not in source:
            return error(f"{field} is required")
        try:
            number = float(source[field])
        except (TypeError, ValueError):
            return error("Both values must be numbers")
        # Task 2: validate the range (attendance must be 0 - 100)
        if not low <= number <= high:
            return error(f"{field} must be between {low} and {high}")
        values[field] = number

    row = pd.DataFrame({
        "study_hours": [values["study_hours"]],
        "attendance": [values["attendance_percentage"]],
    })
    outcome = model.predict(row)[0]

    # Task 3: return both inputs and the prediction as JSON
    return jsonify({
        "study_hours": values["study_hours"],
        "attendance_percentage": values["attendance_percentage"],
        "prediction": "Pass" if outcome == 1 else "Fail",
    })


app = Flask(__name__)
app.register_blueprint(api)

if __name__ == "__main__":
    app.run(debug=True)

