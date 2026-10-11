from flask import Flask, request, jsonify
from joblib import load
import pandas as pd

app = Flask(__name__)
model = load("student_pass_model.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    study_hours = data.get("study_hours")
    attendance = data.get("attendance_percentage")

    if study_hours is None or attendance is None:
        return jsonify({"error": "study_hours and attendance_percentage are required"}), 400

    if not (0 <= attendance <= 100):
        return jsonify({"error": "attendance_percentage must be between 0 and 100"}), 400

    prediction = model.predict(pd.DataFrame({"StudyHours": [study_hours]}))[0]

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": prediction,
    })


if __name__ == "__main__":
    app.run(debug=True)
