
from flask import Flask, request, jsonify
import joblib
import math

app = Flask(__name__)

# Load the trained model once when the app starts
model = joblib.load("student_pass_model.joblib")


@app.route("/")
def home():
    return jsonify({
        "message": "Student Pass Prediction API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Please send a valid JSON object."
        }), 400

    # Check that both inputs are provided
    if "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify({
            "error": "study_hours and attendance_percentage are required."
        }), 400

    # Validate numeric inputs
    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (ValueError, TypeError):
        return jsonify({
            "error": "Both inputs must be numbers."
        }), 400

    if not math.isfinite(study_hours) or study_hours < 0:
        return jsonify({
            "error": "study_hours must be a finite, non-negative number."
        }), 400

    if not math.isfinite(attendance) or not 0 <= attendance <= 100:
        return jsonify({
            "error": "attendance_percentage must be between 0 and 100."
        }), 400

    # The current model uses study hours only.
    prediction = int(model.predict([[study_hours]])[0])
    result = "Pass" if prediction == 1 else "Fail"

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": result,
        "note": "Attendance is validated but does not affect this prediction."
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
