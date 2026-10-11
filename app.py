"""
Flask API for student pass/fail predictions.
Run: python app.py
Test with Postman or curl.exe.
"""
from flask import Flask, request, jsonify
import joblib
from pathlib import Path

app = Flask(__name__)
MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model file not found. Run 'python train_model.py' first."
    )

model = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Student Prediction API is running",
        "endpoint": "POST /predict",
        "example": {"study_hours": 5, "attendance_percentage": 90}
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object in the request body."}), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": "Provide numeric study_hours and attendance_percentage."
        }), 400

    if study_hours < 0:
        return jsonify({"error": "study_hours cannot be negative."}), 400
    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100."}), 400

    prediction_value = int(model.predict([[study_hours]])[0])
    prediction = "Pass" if prediction_value == 1 else "Fail"

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": prediction,
        "note": "This lesson model predicts using study_hours only; attendance is validated and returned but does not affect the prediction."
    })

if __name__ == "__main__":
    app.run(debug=True)
