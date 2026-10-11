
import joblib
from flask import Flask, request, jsonify

server = Flask(__name__)

# Load the trained model
classifier = joblib.load("student_pass_model.joblib")


@server.route("/predict", methods=["POST"])
def get_prediction():
    student_data = request.get_json(silent=True) or {}

    try:
        hours = float(student_data["study_hours"])
        attendance_rate = float(student_data["attendance_percentage"])

    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": "Provide valid study_hours and attendance_percentage"
        }), 400

    if hours < 0 or not 0 <= attendance_rate <= 100:
        return jsonify({
            "error": "Study hours must be non-negative and attendance must be 0-100"
        }), 400

    # Predict the student's result
    predicted_label = classifier.predict([[hours]])[0]
    student_status = "Pass" if predicted_label == 1 else "Fail"

    return jsonify({
        "study_hours": hours,
        "attendance_percentage": attendance_rate,
        "prediction": student_status,
        "note": "Prediction currently uses study hours only."
    })


if __name__ == "__main__":
    server.run(debug=True)