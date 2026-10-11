from flask import Flask, request, jsonify
import joblib
import math

app = Flask(__name__)

# Load the model once when the API starts
model = joblib.load("student_pass_model.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Request body must contain valid JSON."}), 400

    # Check for required fields
    if "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify({"error": "Both study_hours and attendance_percentage are required."}), 400

    study_hours = data["study_hours"]
    attendance = data["attendance_percentage"]

    # Validate study_hours
    if (isinstance(study_hours, bool) or not isinstance(study_hours, (int, float)) 
        or not math.isfinite(study_hours) or study_hours < 0):
        return jsonify({"error": "Study hours must be a non-negative number."}), 400

    # Validate attendance_percentage
    if (isinstance(attendance, bool) or not isinstance(attendance, (int, float)) 
        or not math.isfinite(attendance) or not 0 <= attendance <= 100):
        return jsonify({"error": "Attendance percentage must be between 0 and 100."}), 400

    # Predict using ONLY study_hours (since that's what the model was trained on)
    prediction = model.predict([[study_hours]])[0]

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": str(prediction),
        "note": "Prediction is based on study hours only. Attendance is accepted but not yet used by the model."
    }), 200

if __name__ == "__main__":
    app.run(debug=True, port=5000)