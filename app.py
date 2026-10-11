import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the saved model; do not train it again here.
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
        return jsonify({
            "error": "Please send valid JSON data."
        }), 400

    # Check required inputs
    if "study_hours" not in data:
        return jsonify({
            "error": "study_hours is required."
        }), 400

    if "attendance_percentage" not in data:
        return jsonify({
            "error": "attendance_percentage is required."
        }), 400

    # Validate numeric values
    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({
            "error": "Study hours and attendance must be numbers."
        }), 400

    if study_hours < 0:
        return jsonify({
            "error": "Study hours cannot be negative."
        }), 400

    if not 0 <= attendance <= 100:
        return jsonify({
            "error": "Attendance must be between 0 and 100."
        }), 400

    # The current model uses study hours only.
    prediction = model.predict([[study_hours]])[0]

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": prediction,
        "note": (
            "Prediction is based on study hours only. "
            "Attendance is returned but does not affect the prediction."
        )
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)