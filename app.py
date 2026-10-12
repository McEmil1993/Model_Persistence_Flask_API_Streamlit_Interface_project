from flask import Flask, request, jsonify
import joblib
import math

app = Flask(__name__)

# Load the saved model
model = joblib.load("student_pass_model.joblib")


@app.route("/")
def home():
    return jsonify({
        "message": "Student Pass Prediction API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Please provide valid JSON data"
        }), 400

    if "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify({
            "error": "study_hours and attendance_percentage are required"
        }), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (ValueError, TypeError):
        return jsonify({
            "error": "Both values must be numbers"
        }), 400

    if not math.isfinite(study_hours) or study_hours < 0:
        return jsonify({
            "error": "study_hours must be a finite, non-negative number"
        }), 400

    if not math.isfinite(attendance) or not 0 <= attendance <= 100:
        return jsonify({
            "error": "attendance_percentage must be between 0 and 100"
        }), 400

    # The trained model uses study hours only
    prediction = int(model.predict([[study_hours]])[0])
    result = "PASS" if prediction == 1 else "FAIL"

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": result
    })


if __name__ == "__main__":
    app.run(debug=True)
