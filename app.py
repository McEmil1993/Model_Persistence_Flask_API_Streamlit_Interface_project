# Flask API for Student Pass Prediction
from pathlib import Path
import math

import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model file not found. Run 'python train_model.py' first."
    )

model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Student Pass Prediction API is running.",
        "endpoint": "POST /predict",
        "required_inputs": ["study_hours", "attendance_percentage"],
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object in the request body."}), 400

    try:
        study_hours = float(data.get("study_hours"))
    except (TypeError, ValueError):
        return jsonify({"error": "study_hours must be a number."}), 400

    try:
        attendance_percentage = float(data.get("attendance_percentage"))
    except (TypeError, ValueError):
        return jsonify({"error": "attendance_percentage must be a number."}), 400

    if not math.isfinite(study_hours) or not 0 <= study_hours <= 24:
        return jsonify({"error": "study_hours must be between 0 and 24."}), 400

    if not math.isfinite(attendance_percentage) or not 0 <= attendance_percentage <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100."}), 400

    # Keep the feature order the same as in train_model.py:
    # [study_hours, attendance_percentage]
    prediction_value = int(model.predict([[study_hours, attendance_percentage]])[0])
    prediction = "Pass" if prediction_value == 1 else "Fail"

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance_percentage,
        "prediction": prediction,
        "model_inputs": ["study_hours", "attendance_percentage"],
        "note": "Both study hours and attendance are used by the trained sample model.",
    })


if __name__ == "__main__":
    app.run(debug=True)
