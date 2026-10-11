
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the saved model once when the application starts
model = joblib.load("models/student_pass_model.joblib")


@app.route("/", methods=["GET"])
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
        return jsonify({
            "error": "Request body must be a JSON object."
        }), 400

    # Check that both fields are provided
    if "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify({
            "error": "Provide study_hours and attendance_percentage."
        }), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({
            "error": "Both values must be numbers."
        }), 400

    # Validate the inputs
    if not 0 <= study_hours <= 24:
        return jsonify({
            "error": "Study hours must be between 0 and 24."
        }), 400

    if not 0 <= attendance <= 100:
        return jsonify({
            "error": "Attendance must be between 0 and 100."
        }), 400

    # The model currently uses study hours only
    prediction = model.predict([[study_hours]])[0]

    result = "Pass" if prediction == 1 else "Fail"

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": result,
        "note": "Prediction is based on study hours only."
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
