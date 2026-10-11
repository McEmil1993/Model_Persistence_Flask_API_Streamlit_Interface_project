from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("student_pass_model.joblib")
FEATURES = ["study_hours", "attendance_percentage"]


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Provide numeric 'study_hours' and 'attendance_percentage'."), 400

    if study_hours < 0:
        return jsonify(error="study_hours cannot be negative."), 400
    if not 0 <= attendance <= 100:
        return jsonify(error="attendance_percentage must be bet`ween 0 and 100."), 400

    X = pd.DataFrame([[study_hours, attendance]], columns=FEATURES)
    pred = int(model.predict(X)[0])

    return jsonify(
        study_hours=study_hours,
        attendance_percentage=attendance,
        prediction="Pass" if pred == 1 else "Fail",
    )


if __name__ == "__main__":
    app.run(debug=True)
