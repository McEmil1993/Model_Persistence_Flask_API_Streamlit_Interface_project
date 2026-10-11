"""Exercise 2 - Flask API with study_hours and attendance_percentage."""
import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)
model = joblib.load("student_pass_model_attendance.joblib")


@app.route("/")
def home():
    return jsonify({"message": "Student Pass Prediction API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Send a JSON body"}), 400

    # Both inputs required
    for field in ("study_hours", "attendance_percentage"):
        if field not in data:
            return jsonify({"error": f"'{field}' is required"}), 400

    # Must be numbers (bool is a subclass of int, so reject it explicitly)
    try:
        if isinstance(data["study_hours"], bool) or isinstance(data["attendance_percentage"], bool):
            raise ValueError
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({"error": "study_hours and attendance_percentage must be numbers"}), 400

    # Validation
    if study_hours < 0:
        return jsonify({"error": "study_hours cannot be negative"}), 400
    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100"}), 400

    features = pd.DataFrame(
        {"study_hours": [study_hours], "attendance_percentage": [attendance]}
    )
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": prediction,
        "result": "Pass" if prediction == 1 else "Fail",
        "pass_probability": round(probability, 4),
    })


if __name__ == "__main__":
    app.run(debug=True)
