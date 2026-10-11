
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")
model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Student Prediction API is running.",
        "endpoint": "/predict",
        "method": "POST",
        "required_fields": [
            "study_hours",
            "attendance_percentage"
        ]
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Send a valid JSON object."
        }), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": (
                "Provide numeric study_hours and "
                "attendance_percentage."
            )
        }), 400

    if not 0 <= study_hours <= 24:
        return jsonify({
            "error": "Study hours must be between 0 and 24."
        }), 400

    if not 0 <= attendance <= 100:
        return jsonify({
            "error": "Attendance must be between 0 and 100."
        }), 400

    student = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance_percentage": attendance
    }])

    prediction = model.predict(student)[0]
    probabilities = model.predict_proba(student)[0]
    classes = list(model.classes_)

    return jsonify({
        "prediction": str(prediction),
        "pass_probability": round(
            float(probabilities[classes.index("PASS")]), 4
        )
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
