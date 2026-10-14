import joblib
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

# Load the trained model once when the app starts
model = joblib.load("student_pass_model_v2.joblib")


def parse_number(value, name):
    """Convert a value to float or raise a clear error."""
    if value is None:
        raise ValueError(f"'{name}' is required.")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{name}' must be a number.")


def make_prediction(data):
    study_hours = parse_number(data.get("study_hours"), "study_hours")
    attendance = parse_number(
        data.get("attendance_percentage"), "attendance_percentage"
    )

    if study_hours < 0:
        raise ValueError("'study_hours' cannot be negative.")
    if not 0 <= attendance <= 100:
        raise ValueError("'attendance_percentage' must be between 0 and 100.")

    features = np.array([[study_hours, attendance]])
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])

    return {
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": prediction,
        "result": "Pass" if prediction == 1 else "Fail",
        "pass_probability": round(probability, 4),
    }


@app.route("/")
def home():
    return jsonify({"message": "Student pass prediction API is running."})


@app.route("/predict", methods=["GET", "POST"])
def predict():
    try:
        if request.method == "POST":
            data = request.get_json(silent=True)
            if data is None:
                return jsonify({"error": "Send a valid JSON body."}), 400
        else:
            data = request.args  # query string for GET requests
        return jsonify(make_prediction(data))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400


if __name__ == "__main__":
    app.run(debug=True)