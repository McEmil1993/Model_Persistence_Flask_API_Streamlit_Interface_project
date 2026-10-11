from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model at startup
model = joblib.load("student_pass_model.joblib")

@app.route("/")
def home():
    return jsonify({"message": "Student Pass Prediction API is running."})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    try:
        study_hours = float(data.get("study_hours"))
        attendance = float(data.get("attendance_percentage"))
    except (TypeError, ValueError):
        return jsonify({
            "error": "study_hours and attendance_percentage must be numbers."
        }), 400

    # Validation: attendance must be between 0 and 100
    if study_hours < 0:
        return jsonify({"error": "study_hours must be >= 0."}), 400
    if attendance < 0 or attendance > 100:
        return jsonify({
            "error": "attendance_percentage must be between 0 and 100."
        }), 400

    # Predict
    features = np.array([[study_hours, attendance]])
    prediction = int(model.predict(features)[0])
    result = "Pass" if prediction == 1 else "Fail"

    # Return BOTH inputs and prediction in JSON
    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": result
    })

if __name__ == "__main__":
    app.run(debug=True)