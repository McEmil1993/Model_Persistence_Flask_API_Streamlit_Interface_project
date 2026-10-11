# app.py
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("student_pass_model.joblib")


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Student Pass Prediction API",
        "usage": {
            "endpoint": "/predict",
            "method": "POST",
            "json_body": {"study_hours": 6, "attendance_percentage": 85},
            "example": 'curl -X POST http://127.0.0.1:5000/predict '
                       '-H "Content-Type: application/json" '
                       '-d \'{"study_hours": 6, "attendance_percentage": 85}\''
        }
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "not found",
        "hint": "POST JSON with study_hours and attendance_percentage to /predict"
    }), 404


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    study_hours = data.get("study_hours")
    attendance = data.get("attendance_percentage")

    # Validate study_hours
    if study_hours is None:
        return jsonify({"error": "study_hours is required"}), 400
    try:
        study_hours = float(study_hours)
    except (TypeError, ValueError):
        return jsonify({"error": "study_hours must be a number"}), 400
    if study_hours < 0:
        return jsonify({"error": "study_hours must be >= 0"}), 400

    # Validate attendance
    if attendance is None:
        return jsonify({"error": "attendance_percentage is required"}), 400
    try:
        attendance = float(attendance)
    except (TypeError, ValueError):
        return jsonify({"error": "attendance_percentage must be a number"}), 400
    if not (0 <= attendance <= 100):
        return jsonify({"error": "attendance_percentage must be between 0 and 100"}), 400

    prediction = model.predict([[study_hours, attendance]])[0]
    result = "Pass" if prediction == 1 else "Fail"

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": result
    })


if __name__ == "__main__":
    app.run(debug=True)
