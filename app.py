"""
Jay Mark Renos - Flask API
POST /predict with JSON:
{"study_hours": 5, "attendance_percentage": 90}
The model currently uses study_hours only; attendance is validated and returned
but does not affect the prediction unless the model is retrained with both features.
"""
from pathlib import Path
import joblib
from flask import Flask, jsonify, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_pass_model.joblib"

app = Flask(__name__)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Run 'python train_model.py' first."
        )
    return joblib.load(MODEL_PATH)


model = load_model()


@app.get("/")
def home():
    return jsonify({
        "message": "Student Pass Prediction API is running.",
        "endpoint": "POST /predict",
        "example": {"study_hours": 5, "attendance_percentage": 90}
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a valid JSON object in the request body."}), 400

    if "study_hours" not in data:
        return jsonify({"error": "study_hours is required."}), 400
    if "attendance_percentage" not in data:
        return jsonify({"error": "attendance_percentage is required."}), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({"error": "study_hours and attendance_percentage must be numbers."}), 400

    if study_hours < 0 or study_hours > 24:
        return jsonify({"error": "study_hours must be between 0 and 24."}), 400
    if attendance < 0 or attendance > 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100."}), 400

    prediction_number = int(model.predict([[study_hours]])[0])
    prediction = "Pass" if prediction_number == 1 else "Fail"
    pass_probability = float(model.predict_proba([[study_hours]])[0][1])

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": prediction,
        "pass_probability": round(pass_probability, 4),
        "note": "Prediction uses study_hours only. Attendance is validated and returned but is not used by this model."
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
