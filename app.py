"""Flask API for student pass predictions (based on study hours only)."""

from pathlib import Path

import joblib
from flask import Flask, jsonify, request


MODEL_PATH = Path(__file__).resolve().parent / "student_pass_model.joblib"
app = Flask(__name__)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python train_model.py` first."
        )
    return joblib.load(MODEL_PATH)


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object with study_hours and attendance_percentage."}), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "study_hours and attendance_percentage must be numeric values."}), 400

    if study_hours < 0:
        return jsonify({"error": "study_hours must be 0 or greater."}), 400
    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100."}), 400

    model = load_model()
    prediction = int(model.predict([[study_hours]])[0])
    return jsonify(
        {
            "study_hours": study_hours,
            "attendance_percentage": attendance,
            "prediction": prediction,
            "predicted_class": "Pass" if prediction == 1 else "Fail",
            "note": "This model currently uses study_hours only; attendance is returned but does not affect the prediction.",
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
