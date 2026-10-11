from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved model from Exercise 1
model = joblib.load("student_pass_model.joblib")

@app.route("/")
def home():
    return """
    <h1>Student Pass Prediction API</h1>
    <p>Your Flask API is running successfully!</p>
    <p>Use Postman to test the /predict endpoint.</p>
    """

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Request must contain JSON data"}), 400

    # Check required inputs
    if "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify({
            "error": "Both study_hours and attendance_percentage are required"
        }), 400

    # Validate input values
    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (ValueError, TypeError):
        return jsonify({
            "error": "Both inputs must be numbers"
        }), 400

    if not 0 <= attendance <= 100:
        return jsonify({
            "error": "Attendance must be between 0 and 100"
        }), 400

    if not 0 <= study_hours <= 24:
        return jsonify({
            "error": "Study hours must be between 0 and 24"
        }), 400

    # The current model uses study hours only
    prediction = model.predict(
        pd.DataFrame({"StudyHours": [study_hours]})
    )[0]

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": str(prediction)
    })


if __name__== "__main__":
    app.run(debug=True)