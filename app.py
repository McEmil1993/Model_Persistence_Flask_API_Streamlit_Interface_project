from pathlib import Path
import math
import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)
MODEL_PATH = Path(__file__).resolve().parent / "skilltrack_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError("Model file is missing. Run train_model.py first.")

model = joblib.load(MODEL_PATH)


def read_number(data, key):
    try:
        value = float(data[key])
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number.")
    if not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number.")
    return value


@app.get("/")
def home():
    return jsonify({
        "service": "SkillTrack ML",
        "endpoint": "POST /predict",
        "required_fields": ["study_hours", "attendance_percentage"]
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a valid JSON object."}), 400

    missing = [field for field in ("study_hours", "attendance_percentage") if field not in data]
    if missing:
        return jsonify({"error": "Missing required fields.", "fields": missing}), 400

    try:
        study_hours = read_number(data, "study_hours")
        attendance = read_number(data, "attendance_percentage")
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    if not 0 <= study_hours <= 24:
        return jsonify({"error": "study_hours must be between 0 and 24."}), 400
    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100."}), 400

    progress = str(model.predict([[study_hours, attendance]])[0])
    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "learning_progress": progress,
        "message": "Prediction generated from a small synthetic classroom dataset."
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
