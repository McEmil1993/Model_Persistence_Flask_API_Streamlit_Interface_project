"""Exercise 2 - Flask API
POST /predict  with JSON: {"study_hours": 5, "attendance_percentage": 90}
"""
import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)
model = joblib.load("student_pass_model_attendance.joblib")


@app.route("/")
def home():
    return jsonify({"message": "Student Pass Prediction API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    # --- check required fields ---
    for field in ("study_hours", "attendance_percentage"):
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    # --- convert to numbers ---
    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({"error": "study_hours and attendance_percentage must be numbers"}), 400

    # --- validation ---
    if study_hours < 0:
        return jsonify({"error": "study_hours cannot be negative"}), 400
    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100"}), 400

    # --- predict ---
    features = pd.DataFrame({
        "study_hours": [study_hours],
        "attendance_percentage": [attendance],
    })
    pred = int(model.predict(features)[0])
    prob = float(model.predict_proba(features)[0][1])

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": "Pass" if pred == 1 else "Fail",
        "pass_probability": round(prob, 3),
    })


if __name__ == "__main__":
    app.run(debug=True)
