from pathlib import Path
from math import isfinite

import joblib
from flask import Flask, jsonify, request

from student_model import StudentPerformanceModel

app = Flask(__name__)
MODEL_PATH = Path(__file__).resolve().parent / "student_pass_model.joblib"
model = joblib.load(MODEL_PATH)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
    else:
        payload = request.args.to_dict()

    if not isinstance(payload, dict):
        return jsonify({"error": "Provide a JSON object."}), 400

    study_hours = payload.get("study_hours")
    attendance_percentage = payload.get("attendance_percentage")

    if study_hours is None or attendance_percentage is None:
        return jsonify({"error": "study_hours and attendance_percentage are required."}), 400

    try:
        study_hours = float(study_hours)
    except (TypeError, ValueError):
        return jsonify({"error": "study_hours must be a number."}), 400

    try:
        attendance_percentage = float(attendance_percentage)
    except (TypeError, ValueError):
        return jsonify({"error": "attendance_percentage must be a number."}), 400

    if not isfinite(study_hours) or study_hours < 0:
        return jsonify({"error": "study_hours must be finite and nonnegative."}), 400

    if not 0 <= attendance_percentage <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100."}), 400

    prediction = float(model.predict([[study_hours, attendance_percentage]])[0])
    status = "Pass" if prediction >= 60 else "Fail"

    return jsonify(
        {
            "study_hours": study_hours,
            "attendance_percentage": attendance_percentage,
            "prediction_score": round(prediction, 2),
            "predicted_class": status,
            "note": "This model uses study hours and attendance, trained on synthetic classroom data."
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
