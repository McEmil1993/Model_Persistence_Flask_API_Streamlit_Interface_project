"""Exercise 2: Flask API accepting study_hours and attendance_percentage."""
import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)
model = joblib.load("student_pass_model.joblib")  # loaded once, not retrained


def to_number(value):
    """Return a float, or None if the value is missing / not numeric."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Request body must be JSON."), 400

    study_hours = to_number(data.get("study_hours"))
    attendance = to_number(data.get("attendance_percentage"))

    if study_hours is None:
        return jsonify(error="study_hours is required and must be a number."), 400
    if attendance is None:
        return jsonify(error="attendance_percentage is required and must be a number."), 400
    if study_hours < 0:
        return jsonify(error="study_hours cannot be negative."), 400
    if not 0 <= attendance <= 100:
        return jsonify(error="attendance_percentage must be between 0 and 100."), 400

    features = pd.DataFrame(
        {"study_hours": [study_hours], "attendance_percentage": [attendance]}
    )
    result = int(model.predict(features)[0])

    return jsonify(
        study_hours=study_hours,
        attendance_percentage=attendance,
        prediction="Pass" if result == 1 else "Fail",
    )


if __name__ == "__main__":
    app.run(debug=True)
