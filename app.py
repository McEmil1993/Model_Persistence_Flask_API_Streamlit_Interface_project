"""Flask API for student pass prediction."""
from pathlib import Path

import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)
MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model file not found. Run `python train_model.py` before starting the API."
    )
model = joblib.load(MODEL_PATH)


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object with study_hours and attendance_percentage."}), 400

    if "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify({
            "error": "Both study_hours and attendance_percentage are required."
        }), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({"error": "study_hours and attendance_percentage must be numbers."}), 400

    if study_hours < 0:
        return jsonify({"error": "study_hours cannot be negative."}), 400
    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100."}), 400

    # Current model was trained with study_hours only. Attendance is validated
    # and returned, but it does not affect the prediction yet.
    prediction = int(model.predict([[study_hours]])[0])
    probability = float(model.predict_proba([[study_hours]])[0][1])

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "predicted_class": prediction,
        "prediction": "Pass" if prediction == 1 else "Fail",
        "pass_probability": round(probability, 4),
        "note": "Attendance is returned for reference but does not affect this model's prediction."
    })


@app.get("/")
def home():
    return jsonify({
        "message": "Student Pass Prediction API",
        "usage": "POST /predict with JSON: study_hours and attendance_percentage"
    })


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
