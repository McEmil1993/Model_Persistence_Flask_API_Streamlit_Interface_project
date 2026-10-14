"""
Exercise 2 - Flask API
"""
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("student_pass_model.joblib")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Student Pass Prediction API is running."})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify({"error": "study_hours and attendance_percentage required"}), 400

    study_hours = float(data["study_hours"])
    attendance  = float(data["attendance_percentage"])

    if attendance < 0 or attendance > 100:
        return jsonify({"error": "attendance must be 0-100"}), 400

    pred = int(model.predict(np.array([[study_hours]]))[0])
    label = "Pass" if pred == 1 else "Fail"

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": label,
        "note": "Model uses study_hours only."
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
