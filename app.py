"""Exercise 2 - Flask API with study_hours and attendance_percentage."""
import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)

# Model is loaded once when the server starts (no retraining).
model = joblib.load("student_pass_model_v2.joblib")


@app.route("/")
def home():
    return jsonify({"message": "Student pass prediction API. POST to /predict"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    # Both fields are required
    if "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify(
            {"error": "Both 'study_hours' and 'attendance_percentage' are required"}
        ), 400

    # Must be numbers
    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({"error": "Both values must be numbers"}), 400

    # Validation
    if study_hours < 0:
        return jsonify({"error": "study_hours cannot be negative"}), 400
    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100"}), 400

    pred = model.predict([[study_hours, attendance]])[0]
    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": "Pass" if pred == 1 else "Fail",
    })


if __name__ == "__main__":
    app.run(debug=True)
