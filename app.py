import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)
model = joblib.load("student_pass_model.joblib")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    source = request.get_json(silent=True) or {} if request.method == "POST" else request.args
    try:
        study_hours = float(source.get("study_hours"))
        attendance = float(source.get("attendance_percentage"))
    except (TypeError, ValueError):
        return jsonify(error="study_hours and attendance_percentage must be numbers"), 400

    if study_hours < 0:
        return jsonify(error="study_hours cannot be negative"), 400
    if not 0 <= attendance <= 100:
        return jsonify(error="attendance_percentage must be between 0 and 100"), 400

    X = pd.DataFrame([[study_hours, attendance]],
                     columns=["study_hours", "attendance_percentage"])
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])
    return jsonify(
        study_hours=study_hours,
        attendance_percentage=attendance,
        prediction="Pass" if pred == 1 else "Fail",
        pass_probability=round(prob, 3),
    )


if __name__ == "__main__":
    app.run(debug=True)
