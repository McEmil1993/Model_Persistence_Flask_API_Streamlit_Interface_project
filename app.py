# Exercise 2
from pathlib import Path

import joblib
from flask import Flask, jsonify, request

MODEL_FILE = Path(__file__).resolve().parent / "student_pass_model_v2.joblib"

app = Flask(__name__)
model = joblib.load(MODEL_FILE)   # note: trained on BOTH study hours and attendance


def read_values():
    """Get the inputs from JSON (POST) or from the URL (GET)."""
    if request.method == "POST":
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return None
        return body
    return request.args.to_dict()


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Student Pass Prediction API is running"})


@app.route("/predict", methods=["GET", "POST"])
def predict():
    values = read_values()
    if not values:
        return jsonify({"error": "Please send study_hours and attendance_percentage"}), 400

    # Task 1: accept both study hours and attendance percentage
    missing = [k for k in ("study_hours", "attendance_percentage") if k not in values]
    if missing:
        return jsonify({"error": f"Missing field(s): {', '.join(missing)}"}), 400

    try:
        study_hours = float(values["study_hours"])
        attendance = float(values["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({"error": "Both values must be numbers"}), 400

    if study_hours < 0:
        return jsonify({"error": "study_hours cannot be negative"}), 400

    # Task 2: validate that attendance is between 0 and 100
    if attendance < 0 or attendance > 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100"}), 400

    # Task 3: return both values and the model prediction in JSON
    outcome = model.predict([[study_hours, attendance]])[0]
    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": "Pass" if outcome == 1 else "Fail",
    })


if __name__ == "__main__":
    app.run(debug=True)
