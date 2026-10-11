"""
Exercise 2 — Flask API
POST JSON to /predict with:
{"study_hours": 6, "attendance_percentage": 90}
Note: this provided model uses study hours only. Attendance is validated and
returned in the response, but does not affect the prediction unless the model
is updated and retrained with attendance as a feature.
"""
from pathlib import Path
from flask import Flask, jsonify, request
import joblib

app = Flask(__name__)
MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")


def get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Run `python train_model.py` first."
        )
    return joblib.load(MODEL_PATH)


model = get_model()


@app.get("/")
def home():
    return jsonify({
        "message": "Student Prediction API is running.",
        "usage": "POST /predict with study_hours and attendance_percentage."
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    if "study_hours" not in data:
        return jsonify({"error": "Missing required field: study_hours."}), 400
    if "attendance_percentage" not in data:
        return jsonify({"error": "Missing required field: attendance_percentage."}), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({"error": "study_hours and attendance_percentage must be numbers."}), 400

    if study_hours < 0:
        return jsonify({"error": "study_hours must be 0 or greater."}), 400
    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100."}), 400

    predicted_class = int(model.predict([[study_hours]])[0])
    label = "Pass" if predicted_class == 1 else "Fail"

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "predicted_class": predicted_class,
        "prediction": label,
        "note": "Prediction uses study_hours only; attendance is validated but is not a model feature."
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
