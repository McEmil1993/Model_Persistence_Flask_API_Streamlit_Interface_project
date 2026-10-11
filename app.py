import joblib
import pandas as pd
from flask import Flask, jsonify, request

FEATURES = ["study_hours", "attendance_percentage"]

app = Flask(__name__)
model = joblib.load("student_pass_model.joblib")


def get_number(data, key):
    if key not in data or data[key] is None:
        return None, f"'{key}' is required."
    try:
        return float(data[key]), None
    except (TypeError, ValueError):
        return None, f"'{key}' must be a number."


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Request body must be valid JSON."), 400

    hours, err = get_number(data, "study_hours")
    if err:
        return jsonify(error=err), 400

    attendance, err = get_number(data, "attendance_percentage")
    if err:
        return jsonify(error=err), 400

    if hours < 0:
        return jsonify(error="'study_hours' must be 0 or greater."), 400
    if not 0 <= attendance <= 100:
        return jsonify(error="'attendance_percentage' must be between 0 and 100."), 400

    row = pd.DataFrame([[hours, attendance]], columns=FEATURES)
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1])

    return jsonify(
        study_hours=hours,
        attendance_percentage=attendance,
        prediction=prediction,
        result="pass" if prediction == 1 else "fail",
        pass_probability=round(probability, 4),
    )


if __name__ == "__main__":
    app.run(debug=True)