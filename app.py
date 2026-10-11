# app.py
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("student_pass_model.joblib")
print("✅ Model loaded for Flask API")

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Student Pass Predictor API is running!",
        "endpoint": "POST /predict",
        "example_body": {
            "study_hours": 6,
            "attendance_percentage": 85
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    if "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify({
            "error": "Missing fields. Required: study_hours, attendance_percentage"
        }), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (ValueError, TypeError):
        return jsonify({"error": "study_hours and attendance_percentage must be numbers"}), 400

    if study_hours < 0 or study_hours > 24:
        return jsonify({"error": "study_hours must be between 0 and 24"}), 400

    if attendance < 0 or attendance > 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100"}), 400

    prediction = int(model.predict([[study_hours]])[0])
    probability = float(model.predict_proba([[study_hours]])[0][1])

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": prediction,
        "result": "PASS" if prediction == 1 else "FAIL",
        "pass_probability": round(probability, 4)
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)