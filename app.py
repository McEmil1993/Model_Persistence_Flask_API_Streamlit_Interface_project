
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model
model = joblib.load("student_pass_model.joblib")


@app.route("/predict", methods=["POST"])
def predict_student():
    data = request.get_json(silent=True) or {}

    try:
        study_time = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])

    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": "Provide valid study_hours and attendance_percentage"
        }), 400

    if study_time < 0 or not 0 <= attendance <= 100:
        return jsonify({
            "error": "Study hours must be non-negative and attendance must be 0-100"
        }), 400

    # Predict the student's result
    result = model.predict([[study_time]])[0]
    status = "Pass" if result == 1 else "Fail"

    return jsonify({
        "study_hours": study_time,
        "attendance_percentage": attendance,
        "prediction": status,
        "note": "Prediction currently uses study hours only."
    })


if __name__ == "__main__":
    app.run(debug=True)
