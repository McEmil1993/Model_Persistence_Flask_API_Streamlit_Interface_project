import math
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

MODEL_PATH = Path(__file__).parent / "student_pass_model.joblib"

if not MODEL_PATH.exists():
    raise SystemExit(
        f"Model file not found: {MODEL_PATH}\n"
        "Run `python train_model.py` first to create it."
    )

model = joblib.load(MODEL_PATH)
app = Flask(__name__)


def read_number(payload, field, errors, minimum=None, maximum=None):
    raw = payload.get(field)

    if raw is None or raw == "":
        errors[field] = "is required"
        return None

    if isinstance(raw, bool):
        errors[field] = "must be a number"
        return None

    try:
        value = float(raw)
    except (TypeError, ValueError):
        errors[field] = "must be a number"
        return None

    if not math.isfinite(value):
        errors[field] = "must be a finite number"
        return None

    if minimum is not None and maximum is not None:
        if not minimum <= value <= maximum:
            errors[field] = f"must be between {minimum:g} and {maximum:g}"
            return None
    elif minimum is not None and value < minimum:
        errors[field] = f"must be {minimum:g} or more"
        return None

    return value


@app.get("/")
def index():
    return jsonify(
        message="Student pass prediction API",
        usage={
            "POST /predict": {
                "study_hours": "number >= 0",
                "attendance_percentage": "number between 0 and 100",
            }
        },
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.is_json:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Request body must be a valid JSON object"), 400
    else:
        payload = request.values

    errors = {}
    study_hours = read_number(payload, "study_hours", errors, minimum=0)
    attendance = read_number(
        payload, "attendance_percentage", errors, minimum=0, maximum=100
    )

    if errors:
        return jsonify(error="Invalid input", details=errors), 400

    features = pd.DataFrame(
        {"study_hours": [study_hours], "attendance_percentage": [attendance]}
    )
    predicted_class = int(model.predict(features)[0])
    pass_probability = float(model.predict_proba(features)[0, 1])

    return jsonify(
        study_hours=study_hours,
        attendance_percentage=attendance,
        prediction=predicted_class,
        prediction_label="pass" if predicted_class == 1 else "fail",
        pass_probability=round(pass_probability, 4),
    )


if __name__ == "__main__":
    app.run(debug=True)