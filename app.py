"""Exercise 2 - Flask API using study hours AND attendance."""
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)
model = joblib.load("student_pass_model_v2.joblib")   # loaded once at startup


@app.route("/predict", methods=["GET"])
def predict():
    hours = request.args.get("study_hours")
    attendance = request.args.get("attendance_percentage")

    # Check both inputs exist
    if hours is None or attendance is None:
        return jsonify(error="Please provide study_hours and attendance_percentage"), 400

    # Check they are numbers
    try:
        hours = float(hours)
        attendance = float(attendance)
    except ValueError:
        return jsonify(error="Both values must be numbers"), 400

    # Validate ranges
    if hours < 0:
        return jsonify(error="study_hours cannot be negative"), 400
    if not 0 <= attendance <= 100:
        return jsonify(error="attendance_percentage must be between 0 and 100"), 400

    X = pd.DataFrame({"study_hours": [hours], "attendance_percentage": [attendance]})
    prediction = int(model.predict(X)[0])

    return jsonify(
        study_hours=hours,
        attendance_percentage=attendance,
        prediction=prediction,
        result="Pass" if prediction == 1 else "Fail",
    )


if __name__ == "__main__":
    app.run(debug=True)
