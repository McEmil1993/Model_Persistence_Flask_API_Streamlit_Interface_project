from math import isfinite
from pathlib import Path

import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)
MODEL_PATH = Path(__file__).resolve().parent / "student_pass_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model file not found. Run 'python train_model.py' first."
    )

model = joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return jsonify({
        "message": "Student Pass/Fail Prediction API",
        "endpoint": "POST /predict",
        "required_fields": ["study_hours", "attendance_percentage"],
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send a valid JSON object."}), 400

    if "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify({
            "error": "Both study_hours and attendance_percentage are required."
        }), 400

    values = (data["study_hours"], data["attendance_percentage"])
    if any(isinstance(value, bool) for value in values):
        return jsonify({
            "error": "study_hours and attendance_percentage must be numbers."
        }), 400

    try:
        study_hours = float(values[0])
        attendance = float(values[1])
    except (TypeError, ValueError):
        return jsonify({
            "error": "study_hours and attendance_percentage must be numbers."
        }), 400

    if not isfinite(study_hours) or not isfinite(attendance):
        return jsonify({
            "error": "study_hours and attendance_percentage must be numbers."
        }), 400

    if not 0 <= study_hours <= 24:
        return jsonify({"error": "study_hours must be between 0 and 24."}), 400

    if not 0 <= attendance <= 100:
        return jsonify({
            "error": "attendance_percentage must be between 0 and 100."
        }), 400

    predicted_class = int(model.predict([[study_hours]])[0])
    prediction_label = "Pass" if predicted_class == 1 else "Fail"

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": prediction_label,
        "note": (
            "Prediction uses study_hours only; attendance is validated and "
            "returned but does not affect the current model."
        ),
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
