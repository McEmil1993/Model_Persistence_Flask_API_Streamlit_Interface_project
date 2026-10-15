"""Exercise 2 - Flask API with study_hours and attendance_percentage."""
import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)
model = joblib.load("model_v2.pkl")  # retrained with both features


@app.route("/")
def home():
    return jsonify({"message": "Student pass prediction API. POST JSON to /predict."})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Send a JSON body."}), 400

    if "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify({"error": "study_hours and attendance_percentage are required."}), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({"error": "Both values must be numbers."}), 400

    if study_hours < 0:
        return jsonify({"error": "study_hours must be 0 or greater."}), 400
    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100."}), 400

    X = pd.DataFrame(
        [[study_hours, attendance]],
        columns=["study_hours", "attendance_percentage"],
    )
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": "Pass" if pred == 1 else "Fail",
        "predicted_class": pred,
        "pass_probability": round(prob, 3),
    })


if __name__ == "__main__":
    app.run(debug=True)
