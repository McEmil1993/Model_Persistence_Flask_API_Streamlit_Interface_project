from pathlib import Path
import joblib
from flask import Flask, jsonify, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_pass_model.joblib"

app = Flask(__name__)

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Run 'python train_model.py' before starting the API."
        )
    return joblib.load(MODEL_PATH)

@app.get("/")
def home():
    return jsonify({
        "message": "Student Pass Prediction API",
        "endpoint": "POST /predict",
        "json_example": {
            "study_hours": 5,
            "attendance_percentage": 90
        }
    })

@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be valid JSON."}), 400

    if "study_hours" not in payload or "attendance_percentage" not in payload:
        return jsonify({
            "error": "Both study_hours and attendance_percentage are required."
        }), 400

    try:
        study_hours = float(payload["study_hours"])
        attendance = float(payload["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({
            "error": "study_hours and attendance_percentage must be numeric."
        }), 400

    if study_hours < 0:
        return jsonify({"error": "study_hours must be 0 or greater."}), 400
    if not 0 <= attendance <= 100:
        return jsonify({
            "error": "attendance_percentage must be between 0 and 100."
        }), 400

    try:
        model = load_model()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500

    # Important: this model was trained using study_hours only.
    prediction = int(model.predict([[study_hours]])[0])
    label = "Pass" if prediction == 1 else "Needs Improvement"

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction_class": prediction,
        "prediction": label,
        "note": (
            "Attendance is validated and returned, but it does not affect the "
            "prediction because the current model uses study_hours only."
        )
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
