import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"   # reduces memory use (fixes OpenBLAS error)

from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("student_pass_model_v2.joblib")   # loaded once, no retraining

@app.route("/")
def home():
    return jsonify({"message": "Student Pass Prediction API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    if "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify({"error": "study_hours and attendance_percentage are required"}), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({"error": "Both inputs must be numbers"}), 400

    if study_hours < 0:
        return jsonify({"error": "study_hours cannot be negative"}), 400
    if attendance < 0 or attendance > 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100"}), 400

    features = pd.DataFrame({
        "study_hours": [study_hours],
        "attendance_percentage": [attendance],
    })
    pred = int(model.predict(features)[0])
    prob = float(model.predict_proba(features)[0][1])

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": "Pass" if pred == 1 else "Fail",
        "pass_probability": round(prob, 2),
    })

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)