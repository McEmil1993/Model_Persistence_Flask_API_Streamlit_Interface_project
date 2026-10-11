import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = "student_pass_model.joblib"

# Load the model ONCE when the server starts.
# This is exactly why we did Exercise 1 — no retraining per request.
model = joblib.load(MODEL_PATH)

LABELS = {0: "Fail", 1: "Pass"}

# Feature order must match training: [study_hours, attendance_percentage]
FEATURE_ORDER = ["study_hours", "attendance_percentage"]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Student Pass Prediction API",
        "endpoint": "POST /predict",
        "required_fields": {
            "study_hours": "number >= 0",
            "attendance_percentage": "number between 0 and 100"
        },
        "example_request": {
            "study_hours": 5,
            "attendance_percentage": 85
        }
    })


@app.route("/predict", methods=["POST"])
def predict():
    # ---------- 1. Parse JSON ----------
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    # ---------- 2. Validate study_hours ----------
    if "study_hours" not in data:
        return jsonify({"error": "Missing required field: study_hours"}), 400

    try:
        study_hours = float(data["study_hours"])
    except (TypeError, ValueError):
        return jsonify({"error": "study_hours must be a number."}), 400

    if study_hours < 0:
        return jsonify({"error": "study_hours cannot be negative."}), 400

    # ---------- 3. Validate attendance_percentage ----------
    if "attendance_percentage" not in data:
        return jsonify({"error": "Missing required field: attendance_percentage"}), 400

    try:
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({"error": "attendance_percentage must be a number."}), 400

    if attendance < 0 or attendance > 100:
        return jsonify({
            "error": "attendance_percentage must be between 0 and 100.",
            "received": attendance
        }), 400

    # ---------- 4. Predict ----------
    features = np.array([[study_hours, attendance]], dtype=float)

    prediction_code = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    # model.classes_ is [0, 1], so index 0 = Fail, index 1 = Pass
    prob_fail = float(probabilities[0])
    prob_pass = float(probabilities[1])

    # ---------- 5. Return both inputs + prediction as JSON ----------
    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": LABELS[prediction_code],
        "prediction_code": prediction_code,
        "probability": {
            "Fail": round(prob_fail, 4),
            "Pass": round(prob_pass, 4)
        },
        "model_features_used": FEATURE_ORDER
    }), 200


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)