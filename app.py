
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the saved model without retraining
model = joblib.load("student_pass_model.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send a valid JSON object."}), 400

    study_hours = data.get("study_hours")
    attendance = data.get("attendance_percentage")

    # Validate study hours
    try:
        study_hours = float(study_hours)
        if not 0 <= study_hours <= 24:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({
            "error": "Study hours must be a number between 0 and 24."
        }), 400

    # Validate attendance
    try:
        attendance = float(attendance)
        if not 0 <= attendance <= 100:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({
            "error": "Attendance must be between 0 and 100."
        }), 400

    # Model currently uses study hours only
    prediction = int(model.predict([[study_hours]])[0])
    result = "Pass" if prediction == 1 else "Fail"

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": result
    })


if __name__ == "__main__":
    app.run(debug=True)
