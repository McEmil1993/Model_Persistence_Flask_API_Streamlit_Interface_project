"""Exercise 2: Flask API accepting study hours and attendance percentage."""
from pathlib import Path
from flask import Flask, jsonify, request
import joblib

app = Flask(__name__)
MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")

def get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model missing. Run 'python train_model.py' first.")
    return joblib.load(MODEL_PATH)

@app.get("/")
def home():
    return jsonify({
        "endpoint": "POST /predict",
        "message": "Student Pass Prediction API is running."
    })

@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a valid JSON object."}), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": "Provide numeric study_hours and attendance_percentage."
        }), 400

    if study_hours < 0:
        return jsonify({"error": "study_hours must be 0 or greater."}), 400
    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100."}), 400

    try:
        model = get_model()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500

    predicted = int(model.predict([[study_hours, attendance]])[0])
    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": predicted,
        "result": "Pass" if predicted == 1 else "Fail"
    })

if __name__ == "__main__":
    app.run(debug=True)
