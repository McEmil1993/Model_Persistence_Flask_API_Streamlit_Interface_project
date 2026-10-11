from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("student_pass_model_v2.joblib")  # loaded once at startup


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    try:
        hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (KeyError, TypeError, ValueError):
        return jsonify(error="study_hours and attendance_percentage are required numbers"), 400

    if hours < 0:
        return jsonify(error="study_hours cannot be negative"), 400
    if not 0 <= attendance <= 100:
        return jsonify(error="attendance_percentage must be between 0 and 100"), 400

    X = pd.DataFrame([[hours, attendance]],
                     columns=["study_hours", "attendance_percentage"])
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])

    return jsonify(
        study_hours=hours,
        attendance_percentage=attendance,
        prediction="Pass" if pred == 1 else "Fail",
        probability_of_passing=round(prob, 2),
    )


if __name__ == "__main__":
    app.run(debug=True)