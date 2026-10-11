
from flask import Flask, request, jsonify
import joblib
from pathlib import Path
import math

app = Flask(__name__)

# Load the saved machine learning model
model_path = (
    Path(__file__).resolve().parent
    / "student_pass_model.joblib"
)

model = joblib.load(model_path)


# Home endpoint
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Student Pass/Fail Prediction API is running"
    }), 200


# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    # Get JSON data from the request
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Please send valid JSON data"
        }), 400

    # Check required inputs
    if (
        "study_hours" not in data
        or "attendance_percentage" not in data
    ):
        return jsonify({
            "error": (
                "study_hours and attendance_percentage are required"
            )
        }), 400

    # Validate numeric inputs
    try:
        hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (ValueError, TypeError):
        return jsonify({
            "error": "Inputs must be numeric"
        }), 400

    # Reject NaN and infinity
    if not math.isfinite(hours) or not math.isfinite(attendance):
        return jsonify({
            "error": "Inputs must be finite numbers"
        }), 400

    # Validate study hours
    if hours < 0 or hours > 24:
        return jsonify({
            "error": "study_hours must be between 0 and 24"
        }), 400

    # Validate attendance percentage
    if attendance < 0 or attendance > 100:
        return jsonify({
            "error": (
                "attendance_percentage must be between 0 and 100"
            )
        }), 400

    # Prepare inputs using the same feature names as training
    import pandas as pd

    student_data = pd.DataFrame(
        [[hours, attendance]],
        columns=["study_hours", "attendance_percentage"]
    )

    # Generate prediction
    prediction = int(model.predict(student_data)[0])

    # Calculate estimated pass probability
    probability = float(
        model.predict_proba(student_data)[0][1]
    )

    # Return results as JSON
    return jsonify({
        "status": "success",
        "study_hours": hours,
        "attendance_percentage": attendance,
        "prediction": "Pass" if prediction == 1 else "Fail",
        "pass_probability": round(probability, 4)
    }), 200


# Run the Flask server
if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )