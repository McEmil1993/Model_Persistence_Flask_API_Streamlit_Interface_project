from pathlib import Path
import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)
MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

@app.get("/")
def home():
    return jsonify({
        "message": "Student Pass Prediction API",
        "usage": "POST JSON to /predict",
        "example": {"study_hours": 5, "attendance_percentage": 90}
    })

@app.post("/predict")
def predict():
    if model is None:
        return jsonify({"error": "Model file not found. Run train_model.py first."}), 500
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be valid JSON."}), 400
    try:
        hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Provide numeric study_hours and attendance_percentage."}), 400
    if hours < 0:
        return jsonify({"error": "study_hours must be 0 or greater."}), 400
    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100."}), 400

    prediction = int(model.predict([[hours]])[0])
    return jsonify({
        "study_hours": hours,
        "attendance_percentage": attendance,
        "prediction": prediction,
        "prediction_label": "Pass" if prediction else "Fail",
        "note": "This example model uses study hours only; attendance is validated but does not affect the prediction."
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
