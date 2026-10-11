
from pathlib import Path
import joblib
from flask import Flask, jsonify, request

MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")
if not MODEL_PATH.exists():
    raise FileNotFoundError("Model file not found. Run `python train_model.py` first.")

model = joblib.load(MODEL_PATH)
app = Flask(__name__)

@app.get("/")
def home():
    return jsonify({
        "message": "Student Pass Prediction API",
      
    })

@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be valid JSON."}), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": "Provide numeric study_hours and attendance_percentage."
        }), 400

    if not 0 <= study_hours <= 24:
        return jsonify({"error": "study_hours must be between 0 and 24."}), 400
    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100."}), 400

    prediction = int(model.predict([[study_hours]])[0])
    label = "Pass" if prediction == 1 else "Fail"

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": label,
    })

if __name__ == "__main__":
    app.run(debug=True)
