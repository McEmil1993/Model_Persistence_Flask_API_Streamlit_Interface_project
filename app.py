"""Exercise 2: Flask API with study_hours and attendance_percentage."""
import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)
model = joblib.load("student_pass_model_attendance.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True) or {}

    try:
        study_hours = float(body["study_hours"])
        attendance = float(body["attendance_percentage"])
    except (KeyError, TypeError, ValueError):
        return jsonify(error="study_hours and attendance_percentage are required numbers"), 400

    if study_hours < 0:
        return jsonify(error="study_hours must be 0 or greater"), 400
    if not 0 <= attendance <= 100:
        return jsonify(error="attendance_percentage must be between 0 and 100"), 400

    X = pd.DataFrame([[study_hours, attendance]],
                     columns=["study_hours", "attendance_percentage"])
    pred = int(model.predict(X)[0])

    return jsonify(
        study_hours=study_hours,
        attendance_percentage=attendance,
        prediction="Pass" if pred == 1 else "Fail",
    )


if __name__ == "__main__":
    app.run(debug=True)
