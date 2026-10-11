"""Exercise 2 - Flask API accepting study_hours AND attendance_percentage."""
import joblib
from flask import Flask, jsonify, request
from model_utils import LogisticModel  # needed so joblib can load the model

app = Flask(__name__)
model = joblib.load("student_pass_model_attendance.joblib")


@app.route("/")
def home():
    return jsonify({"message": "Student Pass Prediction API",
                    "usage": "/predict?study_hours=5&attendance_percentage=80"})


@app.route("/predict", methods=["GET", "POST"])
def predict():
    data = request.get_json(silent=True) or request.args
    try:
        study_hours = float(data.get("study_hours"))
        attendance = float(data.get("attendance_percentage"))
    except (TypeError, ValueError):
        return jsonify({"error": "study_hours and attendance_percentage are required numbers"}), 400

    if study_hours < 0:
        return jsonify({"error": "study_hours cannot be negative"}), 400
    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100"}), 400

    pred = int(model.predict([[study_hours, attendance]])[0])
    prob = float(model.predict_proba([[study_hours, attendance]])[0][1])
    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": "Pass" if pred == 1 else "Fail",
        "pass_probability": round(prob, 3),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
