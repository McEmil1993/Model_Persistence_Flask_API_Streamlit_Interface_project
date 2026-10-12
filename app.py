# Exercise 2 — Flask API
# This example model uses study hours only. Attendance is validated and
# returned in the response, but does not affect the prediction unless the
# model is retrained using both features.

from pathlib import Path
import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)
MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


@app.get("/")
def home():
    return jsonify({
        "message": "Student result prediction API",
        "usage": "POST /predict with JSON: {\"study_hours\": 5, \"attendance_percentage\": 90}"
    })


@app.post("/predict")
def predict():
    if model is None:
        return jsonify({"error": "Model file not found. Run python train_model.py first."}), 500

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be valid JSON."}), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": "Provide numeric study_hours and attendance_percentage."
        }), 400

    if study_hours < 0:
        return jsonify({"error": "study_hours must be 0 or greater."}), 400
    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100."}), 400

    prediction = str(model.predict([[study_hours]])[0])
    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": prediction,
        "note": "Attendance is validated and returned but does not influence this model's prediction."
    })


if __name__ == "__main__":
    app.run(debug=True)
