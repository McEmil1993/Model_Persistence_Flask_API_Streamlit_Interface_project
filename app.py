from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "student_pass_model.joblib"

app = Flask(__name__)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Trained model not found. Run 'python train_model.py' before starting the API."
    )

model = joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return jsonify({
        "message": "Student Pass Prediction API",
        "endpoint": "POST /predict",
        "expected_json": {
            "student_name": "Michael Felisilda Canja",
            "study_hours": 4,
            "attendance_percentage": 90
        }
    })


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send a valid JSON object in the request body."}), 400

    try:
        study_hours = float(payload.get("study_hours"))
        attendance = float(payload.get("attendance_percentage"))
    except (TypeError, ValueError):
        return jsonify({
            "error": "study_hours and attendance_percentage must be numeric."
        }), 400

    if study_hours < 0 or study_hours > 24:
        return jsonify({"error": "study_hours must be between 0 and 24."}), 400
    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100."}), 400

    student_name = str(payload.get("student_name", "Student")).strip() or "Student"
    prediction = int(model.predict(pd.DataFrame({"study_hours": [study_hours]}))[0])
    probability = float(model.predict_proba(pd.DataFrame({"study_hours": [study_hours]}))[0][1])

    return jsonify({
        "student_name": student_name,
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": "Pass" if prediction == 1 else "Fail",
        "prediction_class": prediction,
        "pass_probability_percent": round(probability * 100, 2),
        "note": "This model uses study hours only. Attendance is validated and returned, but does not affect the prediction."
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
