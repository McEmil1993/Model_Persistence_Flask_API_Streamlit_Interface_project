from math import isfinite
from pathlib import Path
from typing import Any

import joblib
from flask import Flask, Response, jsonify, request

from student_model import StudentPassModel


MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")
app = Flask(__name__)


def load_model() -> StudentPassModel:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Run train_model.py first."
        )
    return joblib.load(MODEL_PATH)


def parse_number(payload: dict[str, Any], field: str) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number.")
    if not isfinite(value):
        raise ValueError(f"{field} must be a finite number.")
    return float(value)


@app.post("/predict")
def predict() -> Response | tuple[Response, int]:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Request body must be a JSON object."), 400

    try:
        study_hours = parse_number(payload, "study_hours")
        attendance_percentage = parse_number(payload, "attendance_percentage")
        if study_hours < 0:
            raise ValueError("study_hours must be zero or greater.")
        if not 0 <= attendance_percentage <= 100:
            raise ValueError("attendance_percentage must be between 0 and 100.")
    except ValueError as error:
        return jsonify(error=str(error)), 400

    try:
        model = load_model()
    except FileNotFoundError as error:
        return jsonify(error=str(error)), 503

    prediction = int(model.predict([[study_hours]])[0])
    return jsonify(
        study_hours=study_hours,
        attendance_percentage=attendance_percentage,
        prediction=prediction,
        predicted_class="Pass" if prediction == 1 else "Fail",
        prediction_note="Attendance is returned for reference but is not used by this model.",
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
