
from pathlib import Path
import math

import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")

# Load the two-feature model when Flask starts
model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Student Pass Prediction API",
        "endpoint": "/predict",
        "method": "POST",
        "features": [
            "study_hours",
            "attendance_percentage"
        ]
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Please send valid JSON data."
        }), 400

    if "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify({
            "error": (
                "study_hours and attendance_percentage are required."
            )
        }), 400

    study_hours = data["study_hours"]
    attendance = data["attendance_percentage"]

    # Validate study hours
    if (
        isinstance(study_hours, bool)
        or not isinstance(study_hours, (int, float))
        or not math.isfinite(study_hours)
        or not 0 <= study_hours <= 24
    ):
        return jsonify({
            "error": "study_hours must be a number from 0 to 24."
        }), 400

    # Validate attendance
    if (
        isinstance(attendance, bool)
        or not isinstance(attendance, (int, float))
        or not math.isfinite(attendance)
        or not 0 <= attendance <= 100
    ):
        return jsonify({
            "error": (
                "attendance_percentage must be a number from 0 to 100."
            )
        }), 400

    # IMPORTANT: Keep the same feature order used in training.
    features = [[study_hours, attendance]]

    prediction = int(model.predict(features)[0])
    result = "Pass" if prediction == 1 else "Fail"

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": result,
        "model_features": [
            "study_hours",
            "attendance_percentage"
        ],
        "note": (
            "Demonstration model trained using illustrative "
            "rule-generated data. Predictions are not validated "
            "against real student outcomes."
        )
    })


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
