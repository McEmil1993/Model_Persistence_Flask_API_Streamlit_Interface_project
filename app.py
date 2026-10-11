from pathlib import Path
from flask import Flask, jsonify, request
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "student_pass_model.joblib"

app = Flask(__name__)

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run 'python train_model.py' first."
        )
    return joblib.load(MODEL_PATH)

model = load_model()

@app.get("/")
def home():
    return jsonify({
        "message": "Student Prediction API is running.",
        "endpoint": "POST /predict",
        "required_fields": ["study_hours", "attendance_percentage"]
    })

@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be valid JSON."}), 400

    if "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify({
            "error": "Both study_hours and attendance_percentage are required."
        }), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({"error": "Both inputs must be numeric values."}), 400

    if not 0 <= study_hours <= 24:
        return jsonify({"error": "study_hours must be between 0 and 24."}), 400
    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100."}), 400

    prediction = int(model.predict([[study_hours]])[0])
    probability = float(model.predict_proba([[study_hours]])[0][1])
    result = "Pass" if prediction == 1 else "Needs Improvement"

    return jsonify({
        "student_name": str(data.get("student_name", "Not provided")).strip() or "Not provided",
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": prediction,
        "result": result,
        "pass_probability": round(probability, 4),
        "note": "The current model uses study_hours only. Attendance is validated and returned, but does not affect the prediction."
    }), 200

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
