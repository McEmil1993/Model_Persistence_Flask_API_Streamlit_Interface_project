"""Exercise 2 - Flask API

POST /predict  with JSON: {"study_hours": 6, "attendance_percentage": 85}
"""
import joblib
import pandas as pd
from flask import Flask, jsonify, request

MODEL_FILE = "student_pass_model_attendance.joblib"

app = Flask(__name__)
model = joblib.load(MODEL_FILE)  # loaded once at startup


@app.get("/")
def home():
    return jsonify({"message": "Student pass prediction API. POST to /predict."})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    # Both inputs are required
    for field in ("study_hours", "attendance_percentage"):
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    # Must be numbers
    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({"error": "study_hours and attendance_percentage must be numbers."}), 400

    # Validation
    if study_hours < 0:
        return jsonify({"error": "study_hours cannot be negative."}), 400
    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100."}), 400

    X = pd.DataFrame(
        {"study_hours": [study_hours], "attendance_percentage": [attendance]}
    )
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])

    return jsonify(
        {
            "study_hours": study_hours,
            "attendance_percentage": attendance,
            "prediction": "PASS" if pred == 1 else "FAIL",
            "pass_probability": round(prob, 4),
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
