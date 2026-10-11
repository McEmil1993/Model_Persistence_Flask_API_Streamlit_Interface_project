
from flask import Flask, request, jsonify
import joblib
import math

app = Flask(__name__)

# Load the trained model
model = joblib.load("student_pass_model.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Valid JSON object required"}), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (KeyError, ValueError, TypeError):
        return jsonify({
            "error": "Both inputs must be valid numbers"
        }), 400

    if not math.isfinite(study_hours) or study_hours < 0:
        return jsonify({
            "error": "Study hours must be a non-negative finite number"
        }), 400

    if not math.isfinite(attendance) or not 0 <= attendance <= 100:
        return jsonify({
            "error": "Attendance must be between 0 and 100"
        }), 400

    prediction = model.predict([[study_hours]])[0]
    result = "Pass" if prediction == 1 else "Fail"

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": result
    })

if __name__ == "__main__":
    app.run(debug=True)
