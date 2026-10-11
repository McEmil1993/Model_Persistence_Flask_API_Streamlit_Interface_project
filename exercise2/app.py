"""Exercise 2 - Flask API (Rayla)

Exposes the saved model as an API at /api/predict that accepts study_hours
and attendance_percentage and returns a JSON response.

Run:  python app.py      (then test with Postman or curl.exe)
"""

from pathlib import Path

from flask import Flask, request, jsonify

import joblib
import numpy as np

app = Flask(__name__)

# Use a local model file if present, otherwise the one built in exercise1
MODEL_FILE = Path(__file__).resolve().parent / "rayla_pass_model.joblib"
if not MODEL_FILE.exists():
    MODEL_FILE = (
        Path(__file__).resolve().parent.parent / "exercise1" / "rayla_pass_model.joblib"
    )

model = joblib.load(MODEL_FILE)  # loaded once at startup, never retrained


@app.post("/api/predict")
def api_predict():
    body = request.get_json(silent=True) or {}

    # 1. are both fields present?
    missing = [key for key in ("study_hours", "attendance_percentage") if key not in body]
    if missing:
        return jsonify(error=f"Missing field(s): {', '.join(missing)}"), 400

    # 2. are both values numbers?
    try:
        hours = float(body["study_hours"])
        attendance = float(body["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify(error="Both values must be numeric."), 400

    # 3. is attendance inside the allowed range?
    if attendance < 0 or attendance > 100:
        return jsonify(error="attendance_percentage must be between 0 and 100."), 400

    # NOTE: the model only learned from study hours, so attendance is
    # echoed back but does not change the prediction unless you retrain.
    features = np.array([[hours]], dtype=float)
    label = "Pass" if int(model.predict(features)[0]) == 1 else "Fail"

    return jsonify(
        study_hours=hours,
        attendance_percentage=attendance,
        prediction=label,
    ), 200


if __name__ == "__main__":
    app.run(debug=True, port=5001)
