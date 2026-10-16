from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request

MODEL_PATH = Path(__file__).resolve().parent / "student_pass_model_v2.joblib"

app = Flask(__name__)
model = joblib.load(MODEL_PATH)   # trained on study hours AND attendance percentage

TOPIC = {
    "topic": "Flask API",
    "definition": (
        "Flask API is an application programming interface built using Flask, a "
        "lightweight Python web framework. It allows other applications to send data "
        "to a Machine Learning model and receive predictions as responses."
    ),
}


def reject(message):
    return jsonify({"error": message}), 400


@app.route("/")
def home():
    return jsonify({"message": "Student Pass Prediction API is running"})


@app.route("/about")
def about():
    return jsonify(TOPIC)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    # The API receives input data (JSON for POST, URL values for GET)
    data = request.get_json(silent=True) if request.method == "POST" else request.args.to_dict()
    if not isinstance(data, dict) or not data:
        return reject("Send study_hours and attendance_percentage")

    for field in ("study_hours", "attendance_percentage"):
        if field not in data:
            return reject(f"{field} is required")
    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return reject("Both values must be numbers")

    if study_hours < 0:
        return reject("study_hours cannot be negative")

    if not 0 <= attendance <= 100:
        return reject("attendance_percentage must be between 0 and 100")

    label = model.predict(np.array([[study_hours, attendance]]))[0]

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": "Pass" if label == 1 else "Fail",
    })


if __name__ == "__main__":
    app.run(debug=True)