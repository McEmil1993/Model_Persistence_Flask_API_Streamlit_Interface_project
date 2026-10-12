from pathlib import Path
import joblib
from flask import Flask, jsonify, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_pass_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "student_pass_model.joblib was not found. Run train_model.py first."
    )

model = joblib.load(MODEL_PATH)
app = Flask(__name__)


def error(message, status_code=400):
    return jsonify({"error": message}), status_code


@app.get("/")
def home():
    return jsonify({
        "message": "Student Pass Prediction API",
        "endpoint": "POST /predict",
        "inputs": ["study_hours", "attendance_percentage"],
        "note": "The current model uses study hours only; attendance is validated and returned but does not affect the prediction."
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return error("Request body must be valid JSON.")

    if "study_hours" not in data:
        return error("study_hours is required.")
    if "attendance_percentage" not in data:
        return error("attendance_percentage is required.")

    try:
        study_hours = float(data["study_hours"])
    except (TypeError, ValueError):
        return error("study_hours must be a number.")

    try:
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return error("attendance_percentage must be a number.")

    if attendance < 0 or attendance > 100:
        return error("attendance_percentage must be between 0 and 100.")

    if study_hours < 0:
        return error("study_hours cannot be negative.")

    prediction = int(model.predict([[study_hours]])[0])
    prediction_label = "Pass" if prediction == 1 else "Fail"

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": prediction,
        "prediction_label": prediction_label,
        "explanation": "The current trained model uses study hours only. Attendance is accepted and validated but does not influence this prediction."
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
