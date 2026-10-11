from pathlib import Path

import joblib
from flask import Flask, jsonify, request


MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")

app = Flask(__name__)
model = joblib.load(MODEL_PATH)


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify(error="The request body must contain valid JSON."), 400

    if "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify(
            error="study_hours and attendance_percentage are required."
        ), 400

    try:
        study_hours = float(data["study_hours"])
        attendance_percentage = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify(error="Both inputs must be numbers."), 400

    if study_hours < 0:
        return jsonify(error="study_hours cannot be negative."), 400

    if not 0 <= attendance_percentage <= 100:
        return jsonify(
            error="attendance_percentage must be between 0 and 100."
        ), 400

    predicted_class = int(model.predict([[study_hours]])[0])
    prediction = "Pass" if predicted_class == 1 else "Fail"

    return jsonify(
        study_hours=study_hours,
        attendance_percentage=attendance_percentage,
        prediction=prediction,
    )


if __name__ == "__main__":
    app.run(debug=True)
