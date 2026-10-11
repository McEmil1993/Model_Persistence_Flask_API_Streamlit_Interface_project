
from flask import Flask, request, jsonify
import joblib
from pathlib import Path
import math

app = Flask(__name__)

# Load the model created by train_model.py
MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model not found. Run train_model.py first."
    )

model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Student Pass/Fail Prediction API is running",
        "inputs": {
            "study_hours": "Number from 0 to 24",
            "attendance_percentage": "Number from 0 to 100"
        }
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Please provide valid JSON data."
        }), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": (
                "Provide numeric study_hours and "
                "attendance_percentage."
            )
        }), 400

    if not math.isfinite(study_hours) or not math.isfinite(attendance):
        return jsonify({
            "error": "Both inputs must be valid finite numbers."
        }), 400

    if not 0 <= study_hours <= 24:
        return jsonify({
            "error": "Study hours must be between 0 and 24."
        }), 400

    if not 0 <= attendance <= 100:
        return jsonify({
            "error": "Attendance must be between 0 and 100."
        }), 400

    # IMPORTANT:
    # Feature order must match training:
    # StudyHours, then Attendance.
    prediction = model.predict([
        [study_hours, attendance]
    ])[0]

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": str(prediction),
        "message": (
            f"Predicted Result: {prediction}"
        )
    })


if __name__ == "__main__":
    app.run(debug=True)