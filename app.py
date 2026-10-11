from pathlib import Path
import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)
MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model file not found. Run 'python train_model.py' first."
    )

model = joblib.load(MODEL_PATH)

@app.get("/")
def home():
    return jsonify({
        "message": "Student Pass Prediction API",
        "endpoint": "POST /predict",
        "required_json": {
            "study_hours": 5,
            "attendance_percentage": 90
        }
    })

@app.post("/predict")
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
        return jsonify({"error": "study_hours must be 0 or greater."}), 400
    if not 0 <= attendance <= 100:
        return jsonify({
            "error": "attendance_percentage must be between 0 and 100."
        }), 400

    # Important: this model was trained on study_hours only.
    # Attendance is validated and returned, but does not affect the prediction.
    prediction = str(model.predict([[study_hours]])[0])

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": prediction,
        "note": "Prediction uses study_hours only; attendance is not a model feature."
    })

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
