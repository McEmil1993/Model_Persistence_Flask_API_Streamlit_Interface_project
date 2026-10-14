import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)
model = joblib.load("student_pass_model_v2.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (KeyError, TypeError, ValueError):
        return jsonify(error="study_hours and attendance_percentage are required numbers"), 400

    if not 0 <= attendance <= 100:
        return jsonify(error="attendance_percentage must be between 0 and 100"), 400
    if study_hours < 0:
        return jsonify(error="study_hours must be 0 or more"), 400

    pred = int(model.predict([[study_hours, attendance]])[0])

    return jsonify(
        study_hours=study_hours,
        attendance_percentage=attendance,
        prediction="PASS" if pred == 1 else "FAIL",
    )


if __name__ == "__main__":
    app.run(port=5000)