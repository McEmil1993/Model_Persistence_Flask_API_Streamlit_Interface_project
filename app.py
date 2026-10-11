from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model once at startup
model = joblib.load("student_pass_model.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    # Check that both fields are present
    if not data or "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify({
            "error": "Both 'study_hours' and 'attendance_percentage' are required."
        }), 400

    # Convert to numbers
    try:
        study_hours = float(data["study_hours"])
        attendance_percentage = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({"error": "Inputs must be numbers."}), 400

    # Validate ranges
    if study_hours < 0:
        return jsonify({"error": "study_hours must be 0 or greater."}), 400

    if not (0 <= attendance_percentage <= 100):
        return jsonify({
            "error": "attendance_percentage must be between 0 and 100."
        }), 400

    # Prediction (the model currently uses only study_hours)
    X_new = pd.DataFrame([[study_hours]], columns=["study_hours"])
    prediction = int(model.predict(X_new)[0])

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance_percentage,
        "prediction": "Pass" if prediction == 1 else "Fail",
        "prediction_value": prediction
    })


if __name__ == "__main__":
    app.run(debug=True)