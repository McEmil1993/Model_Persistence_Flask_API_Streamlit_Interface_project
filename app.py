import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

# Load the model once at startup (no retraining)
model = joblib.load("student_pass_model_v2.joblib")


@app.route("/")
def home():
    return jsonify({"message": "Student Pass Prediction API is running. POST to /predict."})


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True)
    if body is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    # Check both fields exist
    for field in ("study_hours", "attendance_percentage"):
        if field not in body:
            return jsonify({"error": f"Missing field: {field}"}), 400

    # Check both are numbers
    try:
        study_hours = float(body["study_hours"])
        attendance = float(body["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({"error": "study_hours and attendance_percentage must be numbers."}), 400

    # Validate ranges
    if study_hours < 0:
        return jsonify({"error": "study_hours cannot be negative."}), 400
    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100."}), 400

    # Predict
    sample = pd.DataFrame({
        "study_hours": [study_hours],
        "attendance_percentage": [attendance],
    })
    prediction = int(model.predict(sample)[0])
    probability = float(model.predict_proba(sample)[0][1])

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": "PASS" if prediction == 1 else "FAIL",
        "pass_probability": round(probability, 2),
    })


if __name__ == "__main__":
    app.run(debug=True)