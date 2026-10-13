"""
Exercise 2 - Flask API
POST /predict  with JSON: {"study_hours": 6, "attendance_percentage": 85}
"""
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)
model = joblib.load("student_pass_model_attendance.joblib")  # loaded once, no retraining


@app.route("/")
def home():
    return jsonify({"message": "Student Pass/Fail API is running. Use POST /predict"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Send a JSON body."}), 400

    # --- check both fields exist ---
    if "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify({"error": "Both 'study_hours' and 'attendance_percentage' are required."}), 400

    # --- check they are numbers ---
    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({"error": "Both values must be numbers."}), 400

    # --- validate ranges ---
    if study_hours < 0:
        return jsonify({"error": "study_hours cannot be negative."}), 400
    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100."}), 400

    features = pd.DataFrame([[study_hours, attendance]],
                            columns=["study_hours", "attendance_percentage"])
    pred = int(model.predict(features)[0])
    prob = float(model.predict_proba(features)[0][1])

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": "Pass" if pred == 1 else "Fail",
        "pass_probability": round(prob, 2),
    })


if __name__ == "__main__":
    app.run(debug=True)
