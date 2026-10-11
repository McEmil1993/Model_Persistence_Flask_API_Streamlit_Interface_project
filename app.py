
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the saved model without retraining
model = joblib.load("student_pass_model.joblib")


@app.route("/")
def home():
    return "Student Performance Prediction API is running!"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Please provide JSON data."}), 400

    if "study_hours" not in data:
        return jsonify({"error": "study_hours is required."}), 400

    if "attendance_percentage" not in data:
        return jsonify({
            "error": "attendance_percentage is required."
        }), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (ValueError, TypeError):
        return jsonify({
            "error": "Study hours and attendance must be numbers."
        }), 400

    if not 0 <= attendance <= 100:
        return jsonify({
            "error": "Attendance must be between 0 and 100."
        }), 400

    if not 0 <= study_hours <= 24:
        return jsonify({
            "error": "Study hours must be between 0 and 24."
        }), 400

    # This model currently uses study hours only.
    prediction = model.predict([[study_hours]])[0]
    result = "Pass" if prediction == 1 else "Fail"

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": result,
        "note": "Prediction is based on study hours only."
    })


if __name__ == "__main__":
    app.run(debug=True)
