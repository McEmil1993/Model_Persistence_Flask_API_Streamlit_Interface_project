"""Flask API for student pass predictions."""

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_pass_model.joblib"
app = Flask(__name__)


def load_model():
    """Load the persisted model, training it once if it has not been created."""
    if not MODEL_PATH.exists():
        from train_model import train_and_save_model

        train_and_save_model()
    return joblib.load(MODEL_PATH)


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send a JSON object with study_hours and attendance_percentage."}), 400

    try:
        study_hours = float(payload["study_hours"])
        attendance = float(payload["attendance_percentage"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "study_hours and attendance_percentage must be numeric values."}), 400

    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100."}), 400
    if study_hours < 0:
        return jsonify({"error": "study_hours must be zero or greater."}), 400

    model = load_model()
    features = pd.DataFrame(
        {
            "study_hours": [study_hours],
            "attendance_percentage": [attendance],
        }
    )
    prediction = int(model.predict(features)[0])
    return jsonify(
        {
            "study_hours": study_hours,
            "attendance_percentage": attendance,
            "prediction": "Pass" if prediction == 1 else "Fail",
            "prediction_note": "The model was trained using both study_hours and attendance_percentage.",
        }
    )


@app.get("/")
def home():
    return jsonify({"message": "Student pass prediction API", "usage": "POST /predict with study_hours and attendance_percentage"})


if __name__ == "__main__":
    app.run(debug=True)
