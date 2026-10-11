<<<<<<< HEAD

from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("student_pass_model.joblib")
=======
﻿from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("student_pass_model.joblib")
print("Loaded model from student_pass_model.joblib")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "Student Pass Prediction API",
        "endpoint": "/predict",
        "method": "POST"
    })
>>>>>>> 8f9db9abf4e38999b89fc0670daf90d53af9bdd9


@app.route("/predict", methods=["POST"])
def predict():
<<<<<<< HEAD
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Please provide valid JSON"}), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": "Provide study_hours and attendance_percentage"
        }), 400

    if not 0 <= attendance <= 100:
        return jsonify({
            "error": "Attendance must be between 0 and 100"
        }), 400

    if study_hours < 0:
        return jsonify({
            "error": "Study hours cannot be negative"
        }), 400

    prediction = int(model.predict([[study_hours]])[0])
    result = "Pass" if prediction == 1 else "Fail"

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": result
    })


if __name__ == "__main__":
    app.run(debug=True)
=======
    try:
        data = request.get_json()
        if "study_hours" not in data:
            return jsonify({"error": "Missing study_hours"}), 400
        if "attendance_percentage" not in data:
            return jsonify({"error": "Missing attendance_percentage"}), 400

        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
        student_name = data.get("student_name", "Unknown Student")

        if not (0 <= study_hours <= 24):
            return jsonify({"error": "study_hours must be 0-24"}), 400
        if not (0 <= attendance <= 100):
            return jsonify({"error": "attendance_percentage must be 0-100"}), 400

        features = np.array([[study_hours, attendance]])
        prediction = int(model.predict(features)[0])
        probability = model.predict_proba(features)[0]
        confidence = round(float(max(probability)) * 100, 2)
        label = "Pass" if prediction == 1 else "Fail"

        return jsonify({
            "student_name": student_name,
            "study_hours": study_hours,
            "attendance_percentage": attendance,
            "prediction": label,
            "confidence": confidence
        }), 200
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid input. Numbers required."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
>>>>>>> 8f9db9abf4e38999b89fc0670daf90d53af9bdd9
