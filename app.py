from pathlib import Path
import math
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = Path(__file__).resolve().parent / "models" / "student_pass_model.joblib"
model = joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return jsonify({"message": "Student Pass Prediction API is running"})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "A JSON object is required"}), 400

    try:
        hours = data["study_hours"]
        attendance = data["attendance_percentage"]

        if isinstance(hours, bool) or isinstance(attendance, bool):
            raise ValueError

        hours = float(hours)
        attendance = float(attendance)

        if not math.isfinite(hours) or not math.isfinite(attendance):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": "Provide valid numeric study_hours and attendance_percentage"
        }), 400

    if hours < 0:
        return jsonify({"error": "Study hours cannot be negative"}), 400

    if not 0 <= attendance <= 100:
        return jsonify({
            "error": "Attendance percentage must be between 0 and 100"
        }), 400

    features = pd.DataFrame(
        [[hours, attendance]],
        columns=["study_hours", "attendance_percentage"]
    )
    prediction = int(model.predict(features)[0])

    return jsonify({
        "study_hours": hours,
        "attendance_percentage": attendance,
        "prediction": prediction,
        "prediction_label": "Pass" if prediction == 1 else "Fail"
    })


if __name__ == "__main__":
    app.run(debug=False, port=5000)