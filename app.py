import joblib
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

# Load the trained model
model = joblib.load("student_pass_model.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No input JSON provided"}), 400

    study_hours = data.get("study_hours")
    attendance_percentage = data.get("attendance_percentage")

    # Check for missing values
    if study_hours is None or attendance_percentage is None:
        return (
            jsonify(
                {"error": "Both 'study_hours' and 'attendance_percentage' are required."}
            ),
            400,
        )

    # Validate numeric types and attendance range [0, 100]
    try:
        study_hours = float(study_hours)
        attendance_percentage = float(attendance_percentage)
    except (ValueError, TypeError):
        return jsonify({"error": "Inputs must be numeric values."}), 400

    if not (0 <= attendance_percentage <= 100):
        return (
            jsonify(
                {"error": "attendance_percentage must be between 0 and 100."}
            ),
            400,
        )

    if study_hours < 0:
        return jsonify({"error": "study_hours cannot be negative."}), 400

    # Note: Current model only uses study_hours
    input_features = np.array([[study_hours]])
    prediction = int(model.predict(input_features)[0])
    status = "Pass" if prediction == 1 else "Fail"

    return (
        jsonify(
            {
                "study_hours": study_hours,
                "attendance_percentage": attendance_percentage,
                "prediction": prediction,
                "status": status,
            }
        ),
        200,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)