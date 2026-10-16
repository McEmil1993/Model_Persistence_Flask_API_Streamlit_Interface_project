
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the saved machine learning model
model = joblib.load("student_pass_model.joblib")


@app.route("/")
def home():
    return "Student Pass Prediction API is running!"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({"error": "Please provide JSON data"}), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": "Provide valid study_hours and attendance_percentage"
        }), 400

    # Validate attendance percentage
    if not 0 <= attendance <= 100:
        return jsonify({
            "error": "Attendance must be between 0 and 100"
        }), 400

    if study_hours < 0:
        return jsonify({
            "error": "Study hours cannot be negative"
        }), 400

    # The current trained model uses study hours only
    prediction = model.predict([[study_hours]])[0]

    result = "PASS" if prediction == 1 else "FAIL"

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": result
    })


if __name__ == "__main__":
    app.run(debug=True)
