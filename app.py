# app.py

from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("student_pass_model.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Kunin ang dalawang input
    study_hours = data.get("study_hours")
    attendance = data.get("attendance_percentage")

    # Validation
    if attendance is None or not (0 <= attendance <= 100):
        return jsonify({"error": "attendance_percentage must be between 0 and 100"}), 400

    # Predict
    prediction = model.predict(np.array([[study_hours, attendance]]))[0]
    result = "Pass" if prediction == 1 else "Fail"

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": result
    })

if __name__ == "__main__":
    app.run(debug=True)