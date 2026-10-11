
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the saved model
MODEL_FILE = "student_pass_model.joblib"

try:
    model = joblib.load(MODEL_FILE)
except FileNotFoundError:
    model = None


@app.route("/predict", methods=["GET"])
def home():
    return jsonify({
        "message": "Student Pass Prediction API is running",
        "endpoint": "/predict",
        "method": "POST"
    })


@app.route("/predict", methods=["POST"])
def predict():
    # Check if the model exists
    if model is None:
        return jsonify({
            "error": "Model file not found. Run train_model.py first."
        }), 500

    # Get JSON data
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Request body must be valid JSON."
        }), 400

    # Check required inputs
    if "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify({
            "error": "Both study_hours and attendance_percentage are required."
        }), 400

    # Validate study hours
    try:
        study_hours = float(data["study_hours"])
    except (TypeError, ValueError):
        return jsonify({
            "error": "study_hours must be a valid number."
        }), 400

    if not 0 <= study_hours <= 24:
        return jsonify({
            "error": "study_hours must be between 0 and 24."
        }), 400

    # Validate attendance percentage
    try:
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({
            "error": "attendance_percentage must be a valid number."
        }), 400

    if not 0 <= attendance <= 100:
        return jsonify({
            "error": "attendance_percentage must be between 0 and 100."
        }), 400

    # The current model uses study_hours only.
    input_data = pd.DataFrame({
        "study_hours": [study_hours]
    })

    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])

    result = "Pass" if prediction == 1 else "Fail"

    # Return both input values and prediction
    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": result,
        "pass_probability": round(probability, 4),
        "note": (
            "The current model uses study_hours only. "
            "Attendance percentage is accepted but does not affect "
            "the prediction."
        )
    }), 200


if __name__ == "__main__":
    app.run(debug=True)
