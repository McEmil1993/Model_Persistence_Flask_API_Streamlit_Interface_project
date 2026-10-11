from pathlib import Path
import math
import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)
MODEL_FILE = Path(__file__).resolve().parent / "studymate_model.joblib"

if not MODEL_FILE.is_file():
    raise FileNotFoundError("Run train_model.py to create studymate_model.joblib.")

model = joblib.load(MODEL_FILE)


def read_number(data, field):
    try:
        value = float(data[field])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"{field} must be a number.")
    if not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number.")
    return value


@app.get("/")
def index():
    return jsonify({
        "app": "StudyMate AI",
        "endpoint": "POST /predict",
        "fields": ["study_hours", "attendance_percentage"]
    })


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send a JSON object."}), 400

    try:
        hours = read_number(payload, "study_hours")
        attendance = read_number(payload, "attendance_percentage")
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    if not 0 <= hours <= 24:
        return jsonify({"error": "study_hours must be between 0 and 24."}), 400
    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100."}), 400

    outcome = str(model.predict([[hours, attendance]])[0])
    return jsonify({
        "study_hours": hours,
        "attendance_percentage": attendance,
        "prediction": outcome,
        "notice": "This is a classroom demonstration trained on small synthetic data."
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
