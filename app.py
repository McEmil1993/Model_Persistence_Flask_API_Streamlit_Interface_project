
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

    if not isinstance(data, dict):
        return jsonify({"error": "Please send valid JSON data."}), 400

    if "study_hours" not in data:
        return jsonify({"error": "study_hours is required."}), 400

    if "attendance_percentage" not in data:
        return jsonify({"error": "attendance_percentage is required."}), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (ValueError, TypeError):
        return jsonify({"error": "Inputs must be numbers."}), 400

    if study_hours < 0:
        return jsonify({"error": "Study hours cannot be negative."}), 400

    if not 0 <= attendance <= 100:
        return jsonify({
            "error": "Attendance must be between 0 and 100."
        }), 400

    prediction = model.predict([[study_hours]])[0]

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": str(prediction),
        "note": "The current model uses study hours only. Attendance is validated but does not affect the prediction."
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
