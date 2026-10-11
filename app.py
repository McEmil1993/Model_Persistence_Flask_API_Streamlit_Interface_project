import math
from pathlib import Path

import joblib
from flask import Flask, jsonify, request


MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")
app = Flask(__name__)
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def read_number(payload: dict, field: str, minimum: float, maximum: float | None = None) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")

    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{field} must be at most {maximum}")
    return number


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400

    try:
        study_hours = read_number(payload, "study_hours", minimum=0)
        attendance_percentage = read_number(
            payload, "attendance_percentage", minimum=0, maximum=100
        )
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    if model is None:
        return jsonify({"error": "Model not found. Run train_model.py first."}), 503

    prediction = int(model.predict([[study_hours]])[0])
    return jsonify(
        {
            "study_hours": study_hours,
            "attendance_percentage": attendance_percentage,
            "prediction": prediction,
            "prediction_label": "Pass" if prediction else "Not yet",
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)