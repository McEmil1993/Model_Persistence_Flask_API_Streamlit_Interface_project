"""Exercise 2 - Flask API accepting study hours and attendance percentage."""
import joblib
import pandas as pd
from flask import Flask, jsonify, request

MODEL_FILE = "student_pass_model_attendance.joblib"

app = Flask(__name__)
model = joblib.load(MODEL_FILE)  # loaded once at startup, never retrained


def parse_number(data, name):
    """Return (value, error_message). Accepts JSON body or query string."""
    raw = data.get(name)
    if raw is None or raw == "":
        return None, f"'{name}' is required."
    try:
        return float(raw), None
    except (TypeError, ValueError):
        return None, f"'{name}' must be a number."


@app.get("/")
def home():
    return jsonify(
        message="Student pass prediction API",
        usage="POST /predict with JSON {study_hours, attendance_percentage}",
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    data = request.get_json(silent=True) or request.values

    hours, err = parse_number(data, "study_hours")
    if err:
        return jsonify(error=err), 400
    attendance, err = parse_number(data, "attendance_percentage")
    if err:
        return jsonify(error=err), 400

    if hours < 0:
        return jsonify(error="'study_hours' must be 0 or greater."), 400
    if not 0 <= attendance <= 100:
        return (
            jsonify(error="'attendance_percentage' must be between 0 and 100."),
            400,
        )

    X = pd.DataFrame(
        {"study_hours": [hours], "attendance_percentage": [attendance]}
    )
    label = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])

    return jsonify(
        study_hours=hours,
        attendance_percentage=attendance,
        prediction="pass" if label == 1 else "fail",
        predicted_class=label,
        pass_probability=round(prob, 4),
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
