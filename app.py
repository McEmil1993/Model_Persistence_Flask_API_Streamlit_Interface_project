from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("student_pass_model_v2.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    try:
        hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (KeyError, TypeError, ValueError):
        return jsonify(error="study_hours and attendance_percentage are required numbers"), 400

    if not 0 <= attendance <= 100:
        return jsonify(error="attendance_percentage must be between 0 and 100"), 400
    if hours < 0:
        return jsonify(error="study_hours cannot be negative"), 400

    features = pd.DataFrame([[hours, attendance]],
                            columns=["study_hours", "attendance_percentage"])
    prediction = int(model.predict(features)[0])

    return jsonify(
        study_hours=hours,
        attendance_percentage=attendance,
        prediction=prediction,
        result="Pass" if prediction == 1 else "Fail",
    )

if __name__ == "__main__":
    app.run(debug=True)
