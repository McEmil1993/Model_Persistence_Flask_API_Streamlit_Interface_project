"""A small Flask API that predicts whether a student will pass."""

import math
from pathlib import Path

import joblib
from flask import Flask, jsonify, request


# Create the Flask web application.
app = Flask(__name__)

# Load the saved model once when this application starts.
# The path is based on this file, so it works when launched from another folder.
MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")
model = joblib.load(MODEL_PATH)


def get_number(data, field_name):
    """Read one finite numeric field, or return an error message."""
    if field_name not in data:
        return None, f"Missing required field: {field_name}."

    value = data[field_name]
    # bool is excluded because True/False are not meaningful numeric inputs here.
    if isinstance(value, bool):
        return None, f"{field_name} must be a number."

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, f"{field_name} must be a number."

    if not math.isfinite(number):
        return None, f"{field_name} must be a finite number."

    return number, None


@app.post("/predict")
def predict():
    """Validate JSON input, then return a prediction from the loaded model."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Request body must be a JSON object."), 400

    study_hours, error = get_number(data, "study_hours")
    if error:
        return jsonify(error=error), 400

    attendance, error = get_number(data, "attendance_percentage")
    if error:
        return jsonify(error=error), 400

    # Check the allowed ranges before asking the model for a prediction.
    if study_hours < 0:
        return jsonify(error="study_hours cannot be negative."), 400
    if not 0 <= attendance <= 100:
        return jsonify(
            error="attendance_percentage must be between 0 and 100."
        ), 400

    # The model expects its two features in the same order as the training data.
    prediction = model.predict([[study_hours, attendance]])[0]
    result = "Pass" if prediction == 1 else "Fail"

    return jsonify(
        study_hours=study_hours,
        attendance_percentage=attendance,
        prediction=result,
    )


if __name__ == "__main__":
    # Start the development server when this file is run directly.
    app.run(host="127.0.0.1", port=5000, debug=False)
