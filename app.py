from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the newly trained model
model = joblib.load("student_pass_model.joblib")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Student Performance Prediction API is running."
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
            "error": "Please provide study_hours and attendance_percentage."
        }), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({
            "error": "Both values must be numbers."
        }), 400

    if study_hours < 0 or study_hours > 24:
        return jsonify({
            "error": "Study hours must be between 0 and 24."
        }), 400

    if attendance < 0 or attendance > 100:
        return jsonify({
            "error": "Attendance percentage must be between 0 and 100."
        }), 400

    student_data = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance_percentage": attendance
    }])

    prediction = model.predict(student_data)[0]

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": str(prediction)
    })


if __name__ == "__main__":
    app.run(debug=True)