import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)
model = joblib.load("student_pass_model_attendance.joblib")  # loaded once at startup


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except KeyError as e:
        return jsonify(error=f"Missing field: {e.args[0]}"), 400
    except (TypeError, ValueError):
        return jsonify(error="study_hours and attendance_percentage must be numbers"), 400

    if study_hours < 0:
        return jsonify(error="study_hours must be 0 or greater"), 400
    if not 0 <= attendance <= 100:
        return jsonify(error="attendance_percentage must be between 0 and 100"), 400

    pred = int(model.predict([[study_hours, attendance]])[0])
    prob = float(model.predict_proba([[study_hours, attendance]])[0][1])
    return jsonify(
        study_hours=study_hours,
        attendance_percentage=attendance,
        prediction="Pass" if pred == 1 else "Fail",
        pass_probability=round(prob, 3),
    )


if __name__ == "__main__":
    app.run(debug=True)