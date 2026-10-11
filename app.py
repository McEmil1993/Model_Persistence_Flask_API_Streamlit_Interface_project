
from flask import Flask, request, jsonify
import joblib
import math
import pandas as pd

app = Flask(__name__)

# Load the newly trained model
model = joblib.load("student_pass_model.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Please provide a valid JSON object."
        }), 400

    if "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify({
            "error": "Both study_hours and attendance_percentage are required."
        }), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({
            "error": "Study hours and attendance must be numbers."
        }), 400

    if not math.isfinite(study_hours) or not math.isfinite(attendance):
        return jsonify({
            "error": "Inputs must be finite numbers."
        }), 400

    if study_hours < 0:
        return jsonify({
            "error": "Study hours cannot be negative."
        }), 400

    if not 0 <= attendance <= 100:
        return jsonify({
            "error": "Attendance must be between 0 and 100."
        }), 400

    # The model now predicts using BOTH features
    student = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance_percentage": attendance
    }])

    prediction = model.predict(student)[0]

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": str(prediction)
    })


if __name__ == "__main__":
    app.run(debug=True)
