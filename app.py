import math
from pathlib import Path

import joblib
from flask import Flask, jsonify, request


MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")
if not MODEL_PATH.exists():
    raise RuntimeError("Model file not found. Run train_model.py before starting the API.")

model = joblib.load(MODEL_PATH)
app = Flask(__name__)


def read_number(payload, field, minimum, maximum):
    value = payload.get(field)
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a number.")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a number.") from None
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number.")
    if not minimum <= number <= maximum:
        raise ValueError(f"{field} must be between {minimum} and {maximum}.")
    return number


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    try:
        study_hours = read_number(payload, "study_hours", 0, 24)
        attendance_percentage = read_number(payload, "attendance_percentage", 0, 100)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    predicted_class = int(model.predict([[study_hours]])[0])
    return jsonify(
        {
            "study_hours": study_hours,
            "attendance_percentage": attendance_percentage,
            "prediction": {
                "class": predicted_class,
                "label": "Pass" if predicted_class == 1 else "Fail",
            },
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)