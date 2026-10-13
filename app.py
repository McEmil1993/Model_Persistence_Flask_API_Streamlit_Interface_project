"""Flask API for student pass predictions."""

from pathlib import Path

import joblib
from flask import Flask, jsonify, request


app = Flask(__name__)
MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")


def get_model():
    """Load the persisted model, creating it once if it does not exist yet."""
    if not MODEL_PATH.exists():
        from train_model import train_and_save_model

        train_and_save_model()
    return joblib.load(MODEL_PATH)


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Send a JSON object."), 400

    try:
        study_hours = float(data["study_hours"])
        attendance_percentage = float(data["attendance_percentage"])
    except (KeyError, TypeError, ValueError):
        return jsonify(error="study_hours and attendance_percentage must be numbers."), 400

    if study_hours < 0:
        return jsonify(error="study_hours must be zero or greater."), 400
    if not 0 <= attendance_percentage <= 100:
        return jsonify(error="attendance_percentage must be between 0 and 100."), 400

    model = get_model()
    prediction = int(model.predict([[study_hours]])[0])
    predicted_class = "Pass" if prediction == 1 else "Fail"

    return jsonify(
        study_hours=study_hours,
        attendance_percentage=attendance_percentage,
        prediction=prediction,
        predicted_class=predicted_class,
        note="This model currently uses study_hours only; attendance is validated and returned but does not affect the prediction.",
    )


if __name__ == "__main__":
    app.run(debug=True)
