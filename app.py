from flask import Flask, request, jsonify
import joblib
import math

app = Flask(__name__)

# Load the saved model once when the API starts
model = joblib.load("student_pass_model.joblib")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Student Pass Prediction API is running",
        "endpoint": "/predict",
        "method": "POST",
        "required_fields": [
            "study_hours",
            "attendance_percentage"
        ]
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Please send a valid JSON object."
        }), 400

    if "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify({
            "error": "Both study_hours and attendance_percentage are required."
        }), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({
            "error": "Both inputs must be numeric."
        }), 400

    if not math.isfinite(study_hours) or study_hours < 0:
        return jsonify({
            "error": "study_hours must be a finite number greater than or equal to 0."
        }), 400

    if not math.isfinite(attendance) or not 0 <= attendance <= 100:
        return jsonify({
            "error": "attendance_percentage must be between 0 and 100."
        }), 400

    # The current model predicts from study hours only
    prediction = int(model.predict([[study_hours]])[0])
    status = "Pass" if prediction == 1 else "Fail"

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": prediction,
        "status": status,
        "note": "Attendance is validated but does not affect this model's prediction."
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)