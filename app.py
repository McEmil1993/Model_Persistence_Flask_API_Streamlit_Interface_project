
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

print("Starting Student Pass Prediction API...", flush=True)

app = Flask(__name__)

MODEL_FILE = os.path.join(os.path.dirname(__file__), "student_pass_model.joblib")
model = joblib.load(MODEL_FILE)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Student Pass Prediction API is running!"
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Please send JSON data."}), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": "Provide numeric study_hours and attendance_percentage."
        }), 400

    if not 0 <= study_hours <= 24:
        return jsonify({"error": "Study hours must be between 0 and 24."}), 400

    if not 0 <= attendance <= 100:
        return jsonify({
            "error": "Attendance must be between 0 and 100."
        }), 400

    features = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance_percentage": attendance
    }])

    result = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])

    return jsonify({
        "prediction": "PASS" if result == 1 else "FAIL",
        "pass_probability": round(probability, 4)
    })

if __name__ == "__main__":
    print("Open http://127.0.0.1:5000 in your browser.", flush=True)
    app.run(host="127.0.0.1", port=5000, debug=False)
