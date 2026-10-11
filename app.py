import joblib
from flask import Flask, jsonify, request

MODEL_FILE = "student_pass_model_attendance.joblib"

app = Flask(__name__)
model = joblib.load(MODEL_FILE)


def parse_number(raw, name):
    if raw is None:
        raise ValueError(f"'{name}' is required")
    try:
        return float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"'{name}' must be a number")


@app.route("/")
def home():
    return jsonify(message="Student pass predictor. Use /predict with "
                           "study_hours and attendance_percentage.")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    data = request.get_json(silent=True) or request.args

    try:
        study_hours = parse_number(data.get("study_hours"), "study_hours")
        attendance = parse_number(data.get("attendance_percentage"),
                                  "attendance_percentage")
    except ValueError as e:
        return jsonify(error=str(e)), 400

    if study_hours < 0:
        return jsonify(error="'study_hours' must be 0 or greater"), 400
    if not 0 <= attendance <= 100:
        return jsonify(error="'attendance_percentage' must be between 0 and 100"), 400

    features = [[study_hours, attendance]]
    pred = int(model.predict(features)[0])
    prob = float(model.predict_proba(features)[0][1])

    return jsonify(
        study_hours=study_hours,
        attendance_percentage=attendance,
        prediction="pass" if pred == 1 else "fail",
        probability_of_passing=round(prob, 3),
    )


if __name__ == "__main__":
    app.run(debug=True)
