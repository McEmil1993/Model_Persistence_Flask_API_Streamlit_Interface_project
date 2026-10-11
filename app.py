"""
Exercise 2 — Flask API
Run after training the model: python app.py
Endpoint: POST http://127.0.0.1:5000/predict
Example JSON: {"study_hours": 5, "attendance_percentage": 85}
"""
from pathlib import Path

import joblib
from flask import Flask, jsonify, request

MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")
app = Flask(__name__)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Run 'python train_model.py' before starting the API."
        )
    return joblib.load(MODEL_PATH)


model = load_model()


@app.get("/")
def home():
    return jsonify({
        "message": "Student Pass Prediction API",
        "usage": "Send a POST request to /predict with study_hours and attendance_percentage."
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be valid JSON."}), 400

    study_hours = data.get("study_hours")
    attendance = data.get("attendance_percentage")

    try:
        study_hours = float(study_hours)
    except (TypeError, ValueError):
        return jsonify({"error": "study_hours must be a number."}), 400

    try:
        attendance = float(attendance)
    except (TypeError, ValueError):
        return jsonify({"error": "attendance_percentage must be a number from 0 to 100."}), 400

    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100."}), 400
    if not 0 <= study_hours <= 24:
        return jsonify({"error": "study_hours must be between 0 and 24."}), 400

    prediction = int(model.predict([[study_hours]])[0])
    label = "Pass" if prediction == 1 else "Fail"

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": label,
        "note": "This sample model uses study hours only. Attendance is validated and returned, but does not affect the prediction."
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
