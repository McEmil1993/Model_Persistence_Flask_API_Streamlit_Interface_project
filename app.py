from pathlib import Path
import joblib
import numpy as np
from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "student_pass_model.joblib"
app = Flask(__name__)

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python train_model.py` first.")
    return joblib.load(MODEL_PATH)

@app.get("/")
def home():
    return jsonify({"service": "Student Performance Prediction API", "status": "ok", "endpoint": "POST /predict"})

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_exists": MODEL_PATH.exists()})

@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object in the request body."}), 400
    if "study_hours" not in data:
        return jsonify({"error": "study_hours is required."}), 400
    try:
        hours = float(data["study_hours"])
    except (TypeError, ValueError):
        return jsonify({"error": "study_hours must be a number."}), 400
    if not np.isfinite(hours) or not 0 <= hours <= 24:
        return jsonify({"error": "study_hours must be between 0 and 24."}), 400

    attendance = data.get("attendance_percentage")
    if attendance is not None:
        try:
            attendance = float(attendance)
        except (TypeError, ValueError):
            return jsonify({"error": "attendance_percentage must be a number between 0 and 100."}), 400
        if not np.isfinite(attendance) or not 0 <= attendance <= 100:
            return jsonify({"error": "attendance_percentage must be between 0 and 100."}), 400

    try:
        model = load_model()
        prediction = int(model.predict([[hours]])[0])
        label = "Pass" if prediction == 1 else "Fail"
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503

    response = {"study_hours": hours, "prediction": label,
                "model_features": ["study_hours"],
                "note": "Attendance is validated and returned but does not affect this model's prediction."}
    if attendance is not None:
        response["attendance_percentage"] = attendance
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
