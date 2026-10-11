"""Exercise 2 - Flask API accepting study_hours and attendance_percentage."""
import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)
model = joblib.load("student_pass_model_attendance.joblib")  # loaded once at startup


def parse_number(value, name):
    """Return (float, None) or (None, error message)."""
    if value is None or value == "":
        return None, f"'{name}' is required."
    try:
        return float(value), None
    except (TypeError, ValueError):
        return None, f"'{name}' must be a number."


@app.route("/")
def home():
    return jsonify(message="Student pass predictor. Use POST /predict.")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    data = request.get_json(silent=True) or request.values
    hours, err = parse_number(data.get("study_hours"), "study_hours")
    if err:
        return jsonify(error=err), 400
    attendance, err = parse_number(
        data.get("attendance_percentage"), "attendance_percentage"
    )
    if err:
        return jsonify(error=err), 400

    if hours < 0:
        return jsonify(error="'study_hours' cannot be negative."), 400
    if not 0 <= attendance <= 100:
        return jsonify(error="'attendance_percentage' must be between 0 and 100."), 400

    features = [[hours, attendance]]
    pred = int(model.predict(features)[0])
    prob = float(model.predict_proba(features)[0][1])
    return jsonify(
        study_hours=hours,
        attendance_percentage=attendance,
        prediction="pass" if pred == 1 else "fail",
        pass_probability=round(prob, 3),
    )


if __name__ == "__main__":
    app.run(debug=True)
