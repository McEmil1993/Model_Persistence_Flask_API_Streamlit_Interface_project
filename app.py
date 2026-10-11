import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)
model = joblib.load("student_pass_model_attendance.joblib")


@app.route("/")
def home():
    return jsonify({"message": "Student Pass/Fail Prediction API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    if "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify({"error": "study_hours and attendance_percentage are required"}), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({"error": "Both values must be numbers"}), 400

    if study_hours < 0:
        return jsonify({"error": "study_hours cannot be negative"}), 400
    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100"}), 400

    X = pd.DataFrame({
        "study_hours": [study_hours],
        "attendance_percentage": [attendance],
    })
    result = int(model.predict(X)[0])

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": "Pass" if result == 1 else "Fail",
    })


if __name__ == "__main__":
    app.run(debug=True)