"""
Student Pass Prediction - Flask API
Exercise 2: Accept study_hours and attendance_percentage.
"""

from pathlib import Path
import joblib
from flask import Flask, jsonify, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_pass_model.joblib"

app = Flask(__name__)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model file not found. Run 'python train_model.py' before starting the API."
    )

model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Student Pass Prediction API is running.",
        "endpoint": "POST /predict",
        "required_json": {
            "study_hours": 5,
            "attendance_percentage": 85
        }
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be valid JSON."}), 400

    if "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify({
            "error": "Both study_hours and attendance_percentage are required."
        }), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({
            "error": "study_hours and attendance_percentage must be numbers."
        }), 400

    if study_hours < 0:
        return jsonify({"error": "study_hours cannot be negative."}), 400

    if not 0 <= attendance <= 100:
        return jsonify({
            "error": "attendance_percentage must be between 0 and 100."
        }), 400

    prediction = int(model.predict([[study_hours]])[0])
    result = "Pass" if prediction == 1 else "Fail"

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": prediction,
        "result": result,
        "note": (
            "The current model predicts using study_hours only. "
            "Attendance is validated and returned but does not affect the prediction."
        )
    })


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
