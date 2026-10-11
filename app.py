
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the saved model
model = joblib.load("student_pass_model.joblib")


@app.route("/")
def home():
    return jsonify({
        "message": "Student Pass Prediction API is running."
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
            "error": "Both study_hours and attendance_percentage are required."
        }), 400

    # Validate numeric inputs
    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({
            "error": "Study hours and attendance must be numbers."
        }), 400

    # Check input ranges
    if not 0 <= study_hours <= 24:
        return jsonify({
            "error": "Study hours must be between 0 and 24."
        }), 400

    if not 0 <= attendance <= 100:
        return jsonify({
            "error": "Attendance percentage must be between 0 and 100."
        }), 400

    # Predict using study hours only
    prediction = model.predict([[study_hours]])[0]

    # Return the inputs and prediction as JSON
    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": str(prediction)
    })


if __name__ == "__main__":
    app.run(debug=True)
