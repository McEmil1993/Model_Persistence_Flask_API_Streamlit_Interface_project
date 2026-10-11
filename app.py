import os

import joblib
from flask import Flask, jsonify, request

MODEL_FILENAME = "student_pass_model.joblib"

app = Flask(__name__)
model = joblib.load(MODEL_FILENAME)


def parse_number(payload, key):
    if key not in payload:
        return None, f"'{key}' is required"
    try:
        return float(payload[key]), None
    except (TypeError, ValueError):
        return None, f"'{key}' must be a number"


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "Student Pass Prediction API",
        "endpoint": "POST /predict",
        "payload": {
            "study_hours": "number, required",
            "attendance_percentage": "number between 0 and 100, required",
        },
    })


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be JSON"}), 400

    study_hours, study_error = parse_number(payload, "study_hours")
    if study_error:
        return jsonify({"error": study_error}), 400

    attendance, attendance_error = parse_number(payload, "attendance_percentage")
    if attendance_error:
        return jsonify({"error": attendance_error}), 400

    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100"}), 400

    prediction = int(model.predict([[study_hours]])[0])

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": "pass" if prediction == 1 else "fail",
        "model_features": ["study_hours"],
        "note": (
            "The current model was trained on study hours only. Attendance is "
            "validated and returned but will not affect the prediction until the "
            "model is retrained with attendance as a feature."
        ),
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5001)), debug=True)
