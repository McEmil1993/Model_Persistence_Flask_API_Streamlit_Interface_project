"""Exercise 2 — Flask API (2-feature model).

Accepts study_hours + attendance_percentage, validates attendance 0-100,
and returns both values with the model prediction as JSON.
"""
from pathlib import Path

import joblib
from flask import Flask, jsonify, request

MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")
model = joblib.load(MODEL_PATH)

app = Flask(__name__)


def _parse_float(value, field_name):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{field_name}' must be a number.")
    return result


@app.get("/")
def index():
    return jsonify(
        {
            "message": "Student pass API. POST JSON to /predict.",
            "expects": {"study_hours": "number >= 0", "attendance_percentage": "0-100"},
        }
    )


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be JSON."}), 400

    if "study_hours" not in data or "attendance_percentage" not in data:
        return (
            jsonify(
                {"error": "Both 'study_hours' and 'attendance_percentage' are required."}
            ),
            400,
        )

    try:
        study_hours = _parse_float(data["study_hours"], "study_hours")
        attendance = _parse_float(data["attendance_percentage"], "attendance_percentage")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if study_hours < 0:
        return jsonify({"error": "'study_hours' must be >= 0."}), 400
    if not 0 <= attendance <= 100:
        return jsonify({"error": "'attendance_percentage' must be between 0 and 100."}), 400

    proba_pass = float(model.predict_proba([[study_hours, attendance]])[0][1])
    prediction = int(proba_pass >= 0.5)

    return jsonify(
        {
            "study_hours": study_hours,
            "attendance_percentage": attendance,
            "prediction": prediction,
            "result": "PASS" if prediction == 1 else "FAIL",
            "probability_pass": round(proba_pass, 4),
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
