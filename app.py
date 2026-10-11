"""Flask API for predicting student pass/fail results."""
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)
MODEL_PATH = Path(__file__).with_name("student_pass_model_v2.joblib")
model = joblib.load(MODEL_PATH)


@app.get("/")
def index():
    return jsonify(
        app="Student Pass Predictor API",
        status="online",
        endpoints={
            "predict": "/predict",
            "health": "/health",
        },
    )


@app.get("/health")
def health():
    return jsonify(status="ok")


def parse_numeric_value(value, field_name):
    if value is None or value == "":
        raise ValueError(f"Please provide {field_name}.")

    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number.") from exc

    if numeric_value != numeric_value:
        raise ValueError(f"{field_name} must be a valid number.")

    return numeric_value


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        hours = request.args.get("study_hours")
        attendance = request.args.get("attendance_percentage")
    else:
        payload = request.get_json(silent=True) or request.form or {}
        hours = payload.get("study_hours")
        attendance = payload.get("attendance_percentage")

    try:
        hours = parse_numeric_value(hours, "study_hours")
        attendance = parse_numeric_value(attendance, "attendance_percentage")
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    if hours < 0:
        return jsonify(error="study_hours cannot be negative."), 400
    if not 0 <= attendance <= 100:
        return jsonify(error="attendance_percentage must be between 0 and 100."), 400

    features = pd.DataFrame(
        {"study_hours": [hours], "attendance_percentage": [attendance]}
    )
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])

    return jsonify(
        study_hours=hours,
        attendance_percentage=attendance,
        prediction=prediction,
        probability=probability,
        pass_probability=round(probability * 100, 2),
        result="Pass" if prediction == 1 else "Fail",
        confidence="High" if probability >= 0.7 else "Medium" if probability >= 0.5 else "Low",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
