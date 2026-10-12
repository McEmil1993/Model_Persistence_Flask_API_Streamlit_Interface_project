from flask import Flask, request, jsonify
import joblib
import pandas as pd

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
        return jsonify(error="study_hours cannot be negative"), 400
    if not 0 <= attendance <= 100:
        return jsonify(error="attendance_percentage must be between 0 and 100"), 400

    features = pd.DataFrame({
        "study_hours": [study_hours],
        "attendance_percentage": [attendance],
    })
    pred = int(model.predict(features)[0])

    return jsonify(
        study_hours=study_hours,
        attendance_percentage=attendance,
        prediction=pred,
        result="Pass" if pred == 1 else "Fail",
    )

if __name__ == "__main__":
    app.run(debug=True)