"""Exercise 2: Flask API with study_hours + attendance_percentage."""
import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)
model = joblib.load("student_pass_model_v2.joblib")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    payload = request.get_json(silent=True) or request.values

    try:
        study_hours = float(payload.get("study_hours"))
        attendance = float(payload.get("attendance_percentage"))
    except (TypeError, ValueError):
        return jsonify(error="study_hours and attendance_percentage are required numbers"), 400

    if study_hours < 0:
        return jsonify(error="study_hours must be 0 or more"), 400
    if not 0 <= attendance <= 100:
        return jsonify(error="attendance_percentage must be between 0 and 100"), 400

    X = pd.DataFrame([{"study_hours": study_hours, "attendance_percentage": attendance}])
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])

    return jsonify(
        study_hours=study_hours,
        attendance_percentage=attendance,
        prediction="pass" if prediction == 1 else "fail",
        pass_probability=round(probability, 3),
    )


if __name__ == "__main__":
    app.run(debug=True)