"""Exercise 2 - Flask API accepting study_hours and attendance_percentage."""
import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)
model = joblib.load("student_pass_model.joblib")   # loaded once at startup


def parse_number(name):
    """Read a numeric query/JSON value. Returns (value, error_message)."""
    payload = request.get_json(silent=True) or {}
    raw = request.args.get(name, payload.get(name))
    if raw is None:
        return None, f"Missing required input: {name}"
    try:
        return float(raw), None
    except (TypeError, ValueError):
        return None, f"{name} must be a number"


@app.route("/")
def home():
    return jsonify(message="Student pass predictor. Use /predict?study_hours=4&attendance_percentage=80")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    study_hours, err = parse_number("study_hours")
    if err:
        return jsonify(error=err), 400

    attendance, err = parse_number("attendance_percentage")
    if err:
        return jsonify(error=err), 400

    if study_hours < 0:
        return jsonify(error="study_hours cannot be negative"), 400
    if not 0 <= attendance <= 100:
        return jsonify(error="attendance_percentage must be between 0 and 100"), 400

    features = pd.DataFrame(
        {"study_hours": [study_hours], "attendance_percentage": [attendance]}
    )
    pred = int(model.predict(features)[0])
    prob = float(model.predict_proba(features)[0][1])

    return jsonify(
        study_hours=study_hours,
        attendance_percentage=attendance,
        prediction=pred,
        result="Pass" if pred == 1 else "Fail",
        pass_probability=round(prob, 3),
    )


if __name__ == "__main__":
    app.run(debug=True)