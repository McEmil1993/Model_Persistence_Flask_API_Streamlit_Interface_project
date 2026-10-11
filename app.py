
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("student_pass_model.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    # Check if JSON data was provided
    if not isinstance(data, dict):
        return jsonify({
            "error": "Please provide valid JSON data"
        }), 400

    # Check required inputs
    if "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify({
            "error": "study_hours and attendance_percentage are required"
        }), 400

    # Validate numeric inputs
    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (ValueError, TypeError):
        return jsonify({
            "error": "Both inputs must be numbers"
        }), 400

    # Validate attendance
    if not 0 <= attendance <= 100:
        return jsonify({
            "error": "Attendance must be between 0 and 100"
        }), 400

    # Validate study hours
    if not 0 <= study_hours <= 24:
        return jsonify({
            "error": "Study hours must be between 0 and 24"
        }), 400

    # Current model uses study_hours only
    X = pd.DataFrame({
        "study_hours": [study_hours]
    })

    prediction = model.predict(X)[0]

    # Return JSON response
    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": str(prediction),
        "note": "Prediction currently uses study hours only."
    })


if __name__ == "__main__":
    app.run(debug=True)
