"""
Exercise 2 — Flask API
Run: python app.py
Test: curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\":6,\"attendance_percentage\":90}"
Note: attendance_percentage is validated and returned, but it does NOT affect
the prediction because this model was trained using study_hours only.
"""
from pathlib import Path
import joblib
from flask import Flask, jsonify, request

MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")
app = Flask(__name__)

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python train_model.py` first."
        )
    return joblib.load(MODEL_PATH)

model = load_model()

@app.get("/")
def home():
    return jsonify({
        "message": "Student Pass Prediction API",
        "endpoint": "POST /predict",
        "required_json": {
            "study_hours": "number, 0 or greater",
            "attendance_percentage": "number between 0 and 100"
        }
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
        return jsonify({"error": "Both inputs must be numeric."}), 400

    if study_hours < 0:
        return jsonify({"error": "study_hours must be 0 or greater."}), 400
    if not 0 <= attendance <= 100:
        return jsonify({
            "error": "attendance_percentage must be between 0 and 100."
        }), 400

    prediction = int(model.predict([[study_hours]])[0])
    label = "Pass" if prediction == 1 else "Fail"

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": prediction,
        "prediction_label": label,
        "note": "Attendance is validated and returned, but is not used by the current model."
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
