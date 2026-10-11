import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

# Load the trained model once when the server starts
model = joblib.load("student_pass_model_v2.joblib")


@app.route("/")
def home():
    return jsonify({
        "message": "Student Pass Prediction API is running",
        "usage": "POST or GET /predict with study_hours and attendance_percentage",
    })


@app.route("/predict", methods=["GET", "POST"])
def predict():
    # Accept JSON body (POST) or query parameters (GET)
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
    else:
        payload = request.args

    # 1. Check both inputs exist
    if "study_hours" not in payload or "attendance_percentage" not in payload:
        return jsonify({
            "error": "Both study_hours and attendance_percentage are required"
        }), 400

    # 2. Check both are numbers
    try:
        study_hours = float(payload["study_hours"])
        attendance = float(payload["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({"error": "Inputs must be numbers"}), 400

    # 3. Validate ranges
    if study_hours < 0:
        return jsonify({"error": "study_hours cannot be negative"}), 400
    if attendance < 0 or attendance > 100:
        return jsonify({
            "error": "attendance_percentage must be between 0 and 100"
        }), 400

    # 4. Predict
    sample = pd.DataFrame({
        "study_hours": [study_hours],
        "attendance_percentage": [attendance],
    })
    prediction = int(model.predict(sample)[0])
    probability = float(model.predict_proba(sample)[0][1])

    # 5. Return JSON
    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": "Pass" if prediction == 1 else "Fail",
        "pass_probability": round(probability, 3),
    })


if __name__ == "__main__":
    app.run(debug=True)