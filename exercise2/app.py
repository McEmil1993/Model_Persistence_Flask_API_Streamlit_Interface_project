"""Exercise 2 - Flask API (Rusel)

Exposes the saved model as an API that accepts study_hours and
attendance_percentage and returns a JSON response.

Run:  python app.py      (then test with Postman or curl.exe)
"""

from pathlib import Path

from flask import Flask, request, jsonify

import joblib
import pandas as pd

app = Flask(__name__)

# Use a local model file if present, otherwise the one built in exercise1
MODEL_FILE = Path(__file__).resolve().parent / "rusel_pass_model.joblib"
if not MODEL_FILE.exists():
    MODEL_FILE = (
        Path(__file__).resolve().parent.parent / "exercise1" / "rusel_pass_model.joblib"
    )

model = joblib.load(MODEL_FILE)  # loaded once at startup, never retrained


def check_input(payload):
    """Return an error message, or None when the input is valid."""
    if payload is None:
        return "Request body must be JSON."

    for field in ("study_hours", "attendance_percentage"):
        if field not in payload:
            return f"Missing field: {field}"
        try:
            float(payload[field])
        except (TypeError, ValueError):
            return f"{field} must be numeric."

    if not 0 <= float(payload["attendance_percentage"]) <= 100:
        return "attendance_percentage must be between 0 and 100."

    return None


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)

    error = check_input(payload)
    if error:
        return jsonify({"error": error}), 400

    hours = float(payload["study_hours"])
    attendance = float(payload["attendance_percentage"])

    # NOTE: the model only learned from study hours, so attendance is
    # echoed back but does not change the prediction unless you retrain.
    result = model.predict(pd.DataFrame({"study_hours": [hours]}))[0]

    return jsonify({
        "study_hours": hours,
        "attendance_percentage": attendance,
        "prediction": "Pass" if result == 1 else "Fail",
    }), 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)
