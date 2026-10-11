"""Exercise 2 - Flask API

POST /predict  with JSON: {"study_hours": 6, "attendance_percentage": 85}
Returns both inputs and the model prediction as JSON.

Run train_model_v2.py first so student_pass_model_v2.joblib exists.
"""

import os

import joblib
from flask import Flask, jsonify, request

MODEL_FILE = "student_pass_model_v2.joblib"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# Load once at startup - no retraining when the API handles requests.
model = joblib.load(os.path.join(BASE_DIR, MODEL_FILE))


def parse_number(data, key):
    """Return (value, error_message). Rejects missing, non-numeric, bool values."""
    if key not in data:
        return None, f"'{key}' is required"
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        # allow numeric strings such as "6.5"
        try:
            if isinstance(value, bool):
                raise ValueError
            value = float(value)
        except (TypeError, ValueError):
            return None, f"'{key}' must be a number"
    return float(value), None


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Student pass prediction API",
        "usage": "POST /predict with JSON {study_hours, attendance_percentage}",
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400

    study_hours, err = parse_number(data, "study_hours")
    if err:
        return jsonify({"error": err}), 400
    if study_hours < 0:
        return jsonify({"error": "'study_hours' must be 0 or greater"}), 400

    attendance, err = parse_number(data, "attendance_percentage")
    if err:
        return jsonify({"error": err}), 400
    # Validation required by the exercise: attendance must be 0-100
    if not 0 <= attendance <= 100:
        return jsonify({"error": "'attendance_percentage' must be between 0 and 100"}), 400

    features = [[study_hours, attendance]]
    label = int(model.predict(features)[0])
    prob_pass = float(model.predict_proba(features)[0][1])

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": "pass" if label == 1 else "fail",
        "prediction_label": label,
        "pass_probability": round(prob_pass, 4),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
