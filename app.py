from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("student_pass_model.joblib")


@app.route("/")
def home():
    return jsonify({
        "message": "Student Performance Prediction API",
        "endpoint": "/predict",
        "method": "POST"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send a valid JSON object."}), 400

    if "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify({
            "error": "Both study_hours and attendance_percentage are required."
        }), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({
            "error": "Both inputs must be valid numbers."
        }), 400

    if not (0 <= attendance <= 100):
        return jsonify({
            "error": "Attendance must be between 0 and 100."
        }), 400

    if not (0 <= study_hours <= 24):
        return jsonify({
            "error": "Study hours must be between 0 and 24."
        }), 400

    input_data = pd.DataFrame({
        "study_hours": [study_hours]
    })

    prediction = model.predict(input_data)[0]

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": str(prediction),
        "note": "Attendance is validated but does not affect this model's prediction."
    })


if __name__ == "__main__":
    app.run(debug=True)
