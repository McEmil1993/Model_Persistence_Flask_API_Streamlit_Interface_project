
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request

MODEL_FILENAME = "student_pass_model.joblib"

# Anchor to this script's folder so the server finds the model regardless of
# the directory it was launched from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

app = Flask(__name__)


def load_model():
    """Load the trained model once, so requests don't pay the disk cost."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_FILENAME} not found. Run `python train_model.py` first."
        )
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except FileNotFoundError as error:
    # Fail loudly at startup rather than on the first request.
    raise SystemExit(f"Error: {error}")


def validate_payload(payload):
    """Return (values, error_message). values is None when validation fails.

    Both inputs must be present and numeric; study hours must be non-negative
    (a value per day) and attendance must fall within 0-100.
    """
    if not isinstance(payload, dict):
        return None, "Request body must be a JSON object."

    for field in ("study_hours", "attendance_percentage"):
        if field not in payload:
            return None, f"Missing required field: '{field}'."

    try:
        study_hours = float(payload["study_hours"])
        attendance = float(payload["attendance_percentage"])
    except (TypeError, ValueError):
        return None, "'study_hours' and 'attendance_percentage' must be numbers."

    if study_hours < 0:
        return None, "'study_hours' cannot be negative."

    if not 0 <= attendance <= 100:
        return None, "'attendance_percentage' must be between 0 and 100."

    return (study_hours, attendance), None


@app.route("/", methods=["GET"])
def index():
    return jsonify(
        {
            "service": "Student Pass Predictor",
            "usage": "POST /predict with JSON "
            '{"study_hours": number, "attendance_percentage": number}',
            "note": "The current model was trained on study_hours only, so "
            "attendance_percentage is validated and echoed back but does not "
            "affect the prediction. Retrain the model to change that.",
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    values, error = validate_payload(payload)

    if error:
        return jsonify({"error": error}), 400

    study_hours, attendance = values

    row = pd.DataFrame({"study_hours": [study_hours]})
    prediction = int(model.predict(row)[0])
    result = "Pass" if prediction == 1 else "Fail"

    return jsonify(
        {
            "study_hours": study_hours,
            "attendance_percentage": attendance,
            "prediction": prediction,
            "result": result,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
