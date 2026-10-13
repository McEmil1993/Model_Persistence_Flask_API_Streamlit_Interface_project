
from flask import Flask, request, jsonify
import joblib
import numpy as np
from pathlib import Path

# Create the Flask application FIRST
app = Flask(__name__)

# Load the trained model
MODEL_PATH = Path(__file__).resolve().parent / "student_pass_model.joblib"
model = joblib.load(MODEL_PATH)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Please provide valid JSON"}), 400

    if "study_hours" not in data or "attendance_percentage" not in data:
        return jsonify({"error": "Missing inputs"}), 400

    try:
        hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (ValueError, TypeError):
        return jsonify({"error": "Inputs must be numeric"}), 400

    if not np.isfinite(hours) or not np.isfinite(attendance):
        return jsonify({"error": "Inputs must be finite numbers"}), 400

    if not 0 <= hours <= 24:
        return jsonify({
            "error": "study_hours must be between 0 and 24"
        }), 400

    if not 0 <= attendance <= 100:
        return jsonify({
            "error": "attendance_percentage must be between 0 and 100"
        }), 400

    features = np.array([[hours, attendance]])
    prediction = int(model.predict(features)[0])

    result = {
        "status": "success",
        "study_hours": hours,
        "attendance_percentage": attendance,
        "prediction": "Pass" if prediction == 1 else "Fail"
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        classes = list(model.classes_)

        if 1 in classes:
            result["pass_probability"] = round(
                float(probabilities[classes.index(1)]), 4
            )

    return jsonify(result), 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)
