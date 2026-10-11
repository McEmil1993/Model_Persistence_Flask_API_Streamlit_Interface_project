from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_pass_model_v2.joblib"

app = Flask(__name__)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. Run 'py train_model_v2.py' first."
        )
    return joblib.load(MODEL_PATH)


model = load_model()


@app.route("/")
def home():
    return jsonify({"message": "Student Pass Prediction API is running. POST to /predict."})


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True)
    if body is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    for field in ("study_hours", "attendance_percentage"):
        if field not in body:
            return jsonify({"error": f"Missing field: {field}"}), 400

    try:
        study_hours = float(body["study_hours"])
        attendance = float(body["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({"error": "study_hours and attendance_percentage must be numbers."}), 400

    if study_hours < 0:
        return jsonify({"error": "study_hours cannot be negative."}), 400
    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100."}), 400

    sample = pd.DataFrame({
        "study_hours": [study_hours],
        "attendance_percentage": [attendance],
    })
    prediction = int(model.predict(sample)[0])
    probability = float(model.predict_proba(sample)[0][1])

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": "PASS" if prediction == 1 else "FAIL",
        "pass_probability": round(probability, 2),
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)