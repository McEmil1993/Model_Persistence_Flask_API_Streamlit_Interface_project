
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import math

app = Flask(__name__)

model = joblib.load("student_pass_model.joblib")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Student Prediction API is running",
        "endpoint": "/predict",
        "method": "POST"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Send a valid JSON object."
        }), 400

    if "study_hours" not in data:
        return jsonify({
            "error": "study_hours is required."
        }), 400

    if "attendance_percentage" not in data:
        return jsonify({
            "error": "attendance_percentage is required."
        }), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({
            "error": "Inputs must be numeric."
        }), 400

    if not math.isfinite(study_hours) or not 0 <= study_hours <= 24:
        return jsonify({
            "error": "study_hours must be between 0 and 24."
        }), 400

    if not math.isfinite(attendance) or not 0 <= attendance <= 100:
        return jsonify({
            "error": "attendance_percentage must be between 0 and 100."
        }), 400

    input_data = pd.DataFrame({
        "StudyHours": [study_hours]
    })

    prediction = model.predict(input_data)[0]

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": str(prediction),
        "note": "The current model uses study hours only."
    })


if __name__ == "__main__":
    app.run(debug=False, port=5000)
