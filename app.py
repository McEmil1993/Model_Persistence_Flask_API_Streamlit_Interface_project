import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)
model = joblib.load("student_pass_model.joblib")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    # Accept JSON body (POST) or query parameters (GET)
    data = request.get_json(silent=True) or request.args

    try:
        study_hours = float(data.get("study_hours"))
        attendance = float(data.get("attendance_percentage"))
    except (TypeError, ValueError):
        return jsonify({"error": "study_hours and attendance_percentage must be numbers"}), 400

    if study_hours < 0:
        return jsonify({"error": "study_hours cannot be negative"}), 400

    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100"}), 400

    sample = pd.DataFrame([[study_hours, attendance]],
                          columns=["study_hours", "attendance_percentage"])
    prediction = int(model.predict(sample)[0])

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": prediction,
        "result": "Pass" if prediction == 1 else "Fail",
    })


if __name__ == "__main__":
    app.run(debug=True)