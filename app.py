"""
Exercise 2 — Flask API
POST /predict with JSON:
{
  "study_hours": 3.5,
  "attendance_percentage": 85
}

Note: the saved model uses study_hours only. Attendance is validated and returned,
but it does not affect the prediction until the model is retrained with attendance
as a feature.
"""
from pathlib import Path
from flask import Flask, jsonify, request
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_pass_model.joblib"

app = Flask(__name__)

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH.name} was not found. Run `python train_model.py` first."
        )
    return joblib.load(MODEL_PATH)

@app.get("/")
def home():
    return jsonify({
        "message": "Student Pass Prediction API",
        "endpoint": "POST /predict",
        "required_json": {
            "study_hours": "number",
            "attendance_percentage": "number from 0 to 100"
        }
    })

@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be valid JSON."}), 400

    study_hours = data.get("study_hours")
    attendance = data.get("attendance_percentage")

    try:
        study_hours = float(study_hours)
        attendance = float(attendance)
    except (TypeError, ValueError):
        return jsonify({
            "error": "study_hours and attendance_percentage must be numbers."
        }), 400

    if study_hours < 0:
        return jsonify({"error": "study_hours must be 0 or greater."}), 400
    if not 0 <= attendance <= 100:
        return jsonify({
            "error": "attendance_percentage must be between 0 and 100."
        }), 400

    model = load_model()
    prediction = int(model.predict([[study_hours]])[0])
    label = "Pass" if prediction == 1 else "Fail"

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": prediction,
        "result": label,
        "note": (
            "The current model uses study_hours only; attendance is returned "
            "but does not influence this prediction."
        )
    })

if __name__ == "__main__":
    # Local development server only.
    app.run(host="127.0.0.1", port=5000, debug=True)
