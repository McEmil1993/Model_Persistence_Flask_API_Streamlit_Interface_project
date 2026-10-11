
from flask import Flask, request, jsonify
import joblib
import math

app = Flask(__name__)

model = joblib.load("student_pass_model.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Request body must contain valid JSON."
        }), 400

    if "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify({
            "error": "Both study_hours and attendance_percentage are required."
        }), 400

    study_hours = data["study_hours"]
    attendance = data["attendance_percentage"]

    if (
        isinstance(study_hours, bool)
        or not isinstance(study_hours, (int, float))
        or not math.isfinite(study_hours)
        or study_hours < 0
    ):
        return jsonify({
            "error": "Study hours must be a non-negative number."
        }), 400

    if (
        isinstance(attendance, bool)
        or not isinstance(attendance, (int, float))
        or not math.isfinite(attendance)
        or not 0 <= attendance <= 100
    ):
        return jsonify({
            "error": "Attendance percentage must be between 0 and 100."
        }), 400

    prediction = model.predict([[study_hours]])[0]

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": str(prediction)
    }), 200


if __name__ == "__main__":
    app.run(debug=False)
