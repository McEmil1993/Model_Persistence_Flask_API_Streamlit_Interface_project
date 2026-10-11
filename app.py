"""Flask API that loads the persisted model and serves predictions."""
from pathlib import Path
import math

import joblib
import pandas as pd
from flask import Flask, jsonify, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_pass_model.joblib"

app = Flask(__name__)

# Load once when the API starts; do not retrain for every request.
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError as exc:
    raise RuntimeError(
        "The saved model was not found. Run `python train_model.py` first."
    ) from exc


def is_number(value) -> bool:
    """True for finite int/float values, but not booleans or numeric strings."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


@app.get("/")
def home():
    return jsonify({
        "service": "Student Pass Prediction API",
        "status": "running",
        "prediction_endpoint": "POST /predict",
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a valid JSON object in the request body."}), 400

    required_fields = ["student_name", "study_hours", "attendance_percentage"]
    missing = [field for field in required_fields if field not in data]
    if missing:
        return jsonify({
            "error": "Missing required field(s).",
            "missing_fields": missing,
        }), 400

    student_name = data["student_name"]
    study_hours = data["study_hours"]
    attendance = data["attendance_percentage"]

    if not isinstance(student_name, str) or not student_name.strip():
        return jsonify({"error": "student_name must be a non-empty string."}), 400

    if not is_number(study_hours):
        return jsonify({"error": "study_hours must be a finite number."}), 400
    if study_hours < 0 or study_hours > 24:
        return jsonify({"error": "study_hours must be between 0 and 24."}), 400

    if not is_number(attendance):
        return jsonify({"error": "attendance_percentage must be a finite number."}), 400
    if attendance < 0 or attendance > 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100."}), 400

    features = pd.DataFrame([{
        "study_hours": float(study_hours),
        "attendance_percentage": float(attendance),
    }])
    prediction = str(model.predict(features)[0])

    return jsonify({
        "student_name": student_name.strip(),
        "study_hours": float(study_hours),
        "attendance_percentage": float(attendance),
        "prediction": prediction,
        "message": f"The demonstration model predicts that the student may {prediction.lower()}.",
        "model_note": (
            "This result comes from a model trained on synthetic classroom data. "
            "It is for demonstration only and is not a real academic assessment."
        ),
    }), 200


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"error": "Endpoint not found. Use POST /predict."}), 404


if __name__ == "__main__":
    # Debug is off for safer default behavior.
    app.run(host="127.0.0.1", port=5000, debug=False)
