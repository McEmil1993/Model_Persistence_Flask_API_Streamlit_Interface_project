"""Exercise 2 - Flask API with study_hours AND attendance_percentage.

Run:  python app.py
"""
import joblib
import numpy as np
from flask import Flask, jsonify, request

import model_lib  # noqa: F401  (lets joblib rebuild the model class)

app = Flask(__name__)
model = joblib.load("student_pass_model_attendance.joblib")


@app.route("/")
def home():
    return jsonify({"message": "Student pass predictor API. POST to /predict"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    # Both fields are required
    for field in ("study_hours", "attendance_percentage"):
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    try:
        hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({"error": "study_hours and attendance_percentage must be numbers"}), 400

    if hours < 0:
        return jsonify({"error": "study_hours cannot be negative"}), 400
    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100"}), 400

    X = np.array([[hours, attendance]])
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0])

    return jsonify({
        "study_hours": hours,
        "attendance_percentage": attendance,
        "prediction": "pass" if pred == 1 else "fail",
        "pass_probability": round(prob, 3),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
