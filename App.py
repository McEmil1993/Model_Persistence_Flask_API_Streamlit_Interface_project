"""Exercise 2 - Flask API with study_hours and attendance_percentage."""
import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

# Model trained on BOTH study hours and attendance (see train_model.py)
model = joblib.load("student_pass_model_attendance.joblib")


@app.route("/")
def home():
    return jsonify({"message": "Student pass prediction API. Use POST /predict"})


@app.route("/predict", methods=["POST", "GET"])
def predict():
    # Accept JSON body (POST) or query parameters (GET)
    payload = request.get_json(silent=True) or request.args

    hours_raw = payload.get("study_hours")
    attendance_raw = payload.get("attendance_percentage")

    if hours_raw is None or attendance_raw is None:
        return jsonify({"error": "Both 'study_hours' and 'attendance_percentage' are required."}), 400

    try:
        study_hours = float(hours_raw)
        attendance = float(attendance_raw)
    except (TypeError, ValueError):
        return jsonify({"error": "Both values must be numbers."}), 400

    if study_hours < 0:
        return jsonify({"error": "study_hours must be 0 or greater."}), 400

    # Validation: attendance must be between 0 and 100
    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100."}), 400

    features = pd.DataFrame(
        {"study_hours": [study_hours], "attendance_percentage": [attendance]}
    )
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])

    return jsonify(
        {
            "study_hours": study_hours,
            "attendance_percentage": attendance,
            "prediction": "pass" if prediction == 1 else "fail",
            "prediction_class": prediction,
            "pass_probability": round(probability, 3),
        }
    )


if __name__ == "__main__":
    app.run(debug=True)