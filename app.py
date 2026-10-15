"""
Exercise 2 — Flask API
Run: python app.py
Test POST http://127.0.0.1:5000/predict with JSON:
{"study_hours": 5, "attendance_percentage": 90}
"""
from pathlib import Path
from flask import Flask, jsonify, request
import joblib

app = Flask(__name__)
MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Run 'python train_model.py' first."
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

    if "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify({
            "error": "Both study_hours and attendance_percentage are required."
        }), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({"error": "Both inputs must be numbers."}), 400

    if study_hours < 0:
        return jsonify({"error": "study_hours must be 0 or greater."}), 400
    if not 0 <= attendance <= 100:
        return jsonify({
            "error": "attendance_percentage must be between 0 and 100."
        }), 400

    try:
        model = load_model()
        prediction = int(model.predict([[study_hours]])[0])
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 500

    # Attendance is returned and validated, but does not affect the prediction
    # because the provided model is trained using study_hours only.
    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": prediction,
        "predicted_class": "Pass" if prediction == 1 else "Fail",
        "note": "Prediction currently uses study_hours only; attendance is not a model feature."
    })

if __name__ == "__main__":
    app.run(debug=True)
