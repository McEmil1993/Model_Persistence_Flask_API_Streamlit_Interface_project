"""Exercise 2 - Flask API accepting study_hours and attendance_percentage."""
import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)
model = joblib.load("student_pass_model_v2.joblib")


@app.route("/")
def home():
    return jsonify({"message": "Student pass prediction API. POST to /predict"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Send a JSON body."}), 400

    # Check required fields
    for field in ("study_hours", "attendance_percentage"):
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    # Check they are numbers
    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({"error": "Both values must be numbers."}), 400

    # Validation rules
    if study_hours < 0:
        return jsonify({"error": "study_hours cannot be negative."}), 400
    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100."}), 400

    features = pd.DataFrame(
        [[study_hours, attendance]],
        columns=["study_hours", "attendance_percentage"],
    )
    pred = int(model.predict(features)[0])
    prob = float(model.predict_proba(features)[0][1])

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": "Pass" if pred == 1 else "Fail",
        "pass_probability": round(prob, 3),
    })


if __name__ == "__main__":
    app.run(debug=True)
