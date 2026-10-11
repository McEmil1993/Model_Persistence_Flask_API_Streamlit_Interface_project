"""Exercise 2 - Flask API that predicts pass/fail from study hours and attendance."""
import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)
model = joblib.load("student_pass_model.joblib")  # loaded once at startup


@app.route("/")
def home():
    return jsonify({"message": "Student pass prediction API. POST to /predict."})


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Send a JSON body."}), 400

    if "study_hours" not in body or "attendance_percentage" not in body:
        return jsonify({"error": "study_hours and attendance_percentage are required."}), 400

    try:
        study_hours = float(body["study_hours"])
        attendance = float(body["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({"error": "Both values must be numbers."}), 400

    if study_hours < 0:
        return jsonify({"error": "study_hours cannot be negative."}), 400
    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100."}), 400

    features = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance_percentage": attendance,
    }])

    features = features[model.feature_names_in_]
    
    pred = int(model.predict(features)[0])
    prob = float(model.predict_proba(features)[0][1])

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": "pass" if pred == 1 else "fail",
        "pass_probability": round(prob, 3),
    })


if __name__ == "__main__":
    app.run(debug=True)