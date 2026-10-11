
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the saved model
model = joblib.load("student_pass_model.joblib")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Student Pass Prediction API is running!"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "error": "Please send JSON data."
        }), 400

    # Validate study hours
    study_hours = data.get("study_hours")

    if study_hours is None:
        return jsonify({
            "error": "study_hours is required."
        }), 400

    try:
        study_hours = float(study_hours)
    except (TypeError, ValueError):
        return jsonify({
            "error": "study_hours must be a number."
        }), 400

    if not 0 <= study_hours <= 24:
        return jsonify({
            "error": "study_hours must be between 0 and 24."
        }), 400

    # Validate attendance percentage
    attendance = data.get("attendance_percentage")

    if attendance is None:
        return jsonify({
            "error": "attendance_percentage is required."
        }), 400

    try:
        attendance = float(attendance)
    except (TypeError, ValueError):
        return jsonify({
            "error": "attendance_percentage must be a number."
        }), 400

    if not 0 <= attendance <= 100:
        return jsonify({
            "error": "attendance_percentage must be between 0 and 100."
        }), 400

    # Current model uses study hours only.
    prediction = model.predict([[study_hours]])[0]
    result = "Pass" if prediction == 1 else "Fail"

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": result,
        "note": "Attendance is validated but does not affect the prediction because the current model uses study hours only."
    })


if __name__ == "__main__":
    app.run(debug=True)
