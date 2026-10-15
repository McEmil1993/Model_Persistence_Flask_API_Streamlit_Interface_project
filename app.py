"""Flask API for the persisted student outcome model."""

from pathlib import Path

import joblib
from flask import Flask, jsonify, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_pass_model.joblib"
app = Flask(__name__)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found. Run train_model.py first.")
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL_PATH.exists()})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "study_hours and attendance_percentage must be numbers"}), 400

    if study_hours < 0:
        return jsonify({"error": "study_hours must be zero or greater"}), 400
    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100"}), 400

    model = load_model()
    prediction = int(model.predict([[study_hours, attendance]])[0])
    label = "Pass" if prediction == 1 else "Fail"
    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": label,
    })


if __name__ == "__main__":
    app.run(debug=True)
