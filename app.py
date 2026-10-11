"""Flask JSON API for student pass predictions."""

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "student_pass_model.joblib"
app = Flask(__name__)


def get_model():
    """Load the persisted model when the server starts handling a request."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"{MODEL_PATH.name} is missing. Run `python train_model.py` first."
        )
    return joblib.load(MODEL_PATH)


@app.post("/predict")
def predict():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(error="Send a JSON object in the request body."), 400

    try:
        hours = float(body["study_hours"])
        attendance = float(body["attendance_percentage"])
    except (KeyError, TypeError, ValueError):
        return jsonify(
            error="study_hours and attendance_percentage must both be numbers."
        ), 400

    if not 0 <= attendance <= 100:
        return jsonify(error="attendance_percentage must be between 0 and 100."), 400
    if hours < 0:
        return jsonify(error="study_hours must be zero or greater."), 400

    try:
        model = get_model()
    except FileNotFoundError as exc:
        return jsonify(error=str(exc)), 503

    sample = pd.DataFrame(
        [[hours, attendance]],
        columns=["study_hours", "attendance_percentage"],
    )
    predicted_class = int(model.predict(sample)[0])
    return jsonify(
        study_hours=hours,
        attendance_percentage=attendance,
        predicted_class=predicted_class,
        prediction="Pass" if predicted_class == 1 else "Needs support",
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
