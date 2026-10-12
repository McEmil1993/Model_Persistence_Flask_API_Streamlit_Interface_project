"""Student pass/fail HTTP service.

Exposes a small Flask API over the persisted sklearn pipeline.
Expects two numeric inputs and returns the model's verdict as JSON.
"""

import os
from pathlib import Path

import joblib
from flask import Flask, jsonify, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = Path(os.getenv("PASS_MODEL", BASE_DIR / "student_pass_model.joblib"))

_classifier = joblib.load(MODEL_FILE)


def to_number(raw, label):
    """Convert an incoming JSON value to float or raise a friendly error."""
    try:
        return float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"Field '{label}' must be numeric.")


def check_inputs(hours, attendance):
    """Return an error string when values are out of range, else None."""
    if hours < 0:
        return "Field 'study_hours' must be 0 or greater."
    if not 0 <= attendance <= 100:
        return "Field 'attendance_percentage' must stay within 0-100."
    return None


def create_app():
    server = Flask(__name__)

    @server.get("/")
    def home():
        return jsonify(
            {
                "service": "student-pass-api",
                "usage": "Send POST /predict with study_hours and attendance_percentage.",
                "example": {"study_hours": 4.5, "attendance_percentage": 80},
            }
        )

    @server.post("/predict")
    def score_student():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Body must be a JSON object."}), 400
        if "study_hours" not in payload or "attendance_percentage" not in payload:
            return (
                jsonify(
                    {
                        "error": "Provide both 'study_hours' and "
                        "'attendance_percentage'."
                    }
                ),
                400,
            )

        try:
            hours = to_number(payload["study_hours"], "study_hours")
            attendance = to_number(
                payload["attendance_percentage"], "attendance_percentage"
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        problem = check_inputs(hours, attendance)
        if problem:
            return jsonify({"error": problem}), 400

        chance = float(_classifier.predict_proba([[hours, attendance]])[0][1])
        passed = chance >= 0.5
        return jsonify(
            {
                "study_hours": hours,
                "attendance_percentage": attendance,
                "prediction": int(passed),
                "result": "PASS" if passed else "FAIL",
                "probability_pass": round(chance, 4),
            }
        )

    return server


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
