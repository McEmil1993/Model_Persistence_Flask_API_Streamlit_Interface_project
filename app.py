# Exercise 2 - Flask API (study_hours + attendance_percentage)
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("student_pass_model_attendance.joblib")  # loaded once, not retrained


@app.route("/")
def home():
    return jsonify({"message": "Student Pass/Fail Prediction API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Send a JSON body."}), 400

    if "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify({"error": "Both study_hours and attendance_percentage are required."}), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({"error": "Both values must be numbers."}), 400

    if study_hours < 0:
        return jsonify({"error": "study_hours cannot be negative."}), 400
    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100."}), 400

    pred = int(model.predict([[study_hours, attendance]])[0])
    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": "Pass" if pred == 1 else "Fail",
    })


if __name__ == "__main__":
    app.run(debug=True)
