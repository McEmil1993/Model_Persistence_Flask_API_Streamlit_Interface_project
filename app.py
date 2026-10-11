import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the model ONCE when the server starts (no retraining needed)
model = joblib.load("student_pass_model_v2.joblib")


@app.route("/")
def home():
    return jsonify({"message": "Student Pass Prediction API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True)
    if body is None:
        return jsonify({"error": "Send a JSON body"}), 400

    # Check the fields exist
    for field in ("study_hours", "attendance_percentage"):
        if field not in body:
            return jsonify({"error": f"Missing field: {field}"}), 400

    # Check they are numbers
    try:
        study_hours = float(body["study_hours"])
        attendance = float(body["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({"error": "study_hours and attendance_percentage must be numbers"}), 400

    # Validate ranges
    if study_hours < 0:
        return jsonify({"error": "study_hours must be 0 or more"}), 400
    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100"}), 400

    features = pd.DataFrame(
        [[study_hours, attendance]],
        columns=["study_hours", "attendance_percentage"],
    )
    result = int(model.predict(features)[0])

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": "Pass" if result == 1 else "Fail",
    })


if __name__ == "__main__":
    app.run(debug=True)