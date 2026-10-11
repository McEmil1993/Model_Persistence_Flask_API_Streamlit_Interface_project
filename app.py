"""Small Flask API that serves predictions from the saved model."""
from pathlib import Path
from numbers import Real

import joblib
from flask import Flask, jsonify, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_pass_model.joblib"
FEATURES = ["study_hours", "attendance_percentage"]

# Load once at startup instead of reloading on every request.
model = joblib.load(MODEL_PATH)
app = Flask(__name__)


def numeric_value(value):
    """Accept JSON numbers but reject booleans and numeric-looking strings."""
    return isinstance(value, Real) and not isinstance(value, bool)


@app.post("/predict")
def predict():
    # Require a JSON object with both named values.
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    missing = [key for key in FEATURES if key not in payload]
    if missing:
        return jsonify({"error": f"Missing required field(s): {', '.join(missing)}."}), 400

    study_hours = payload["study_hours"]
    attendance = payload["attendance_percentage"]
    if not numeric_value(study_hours) or not numeric_value(attendance):
        return jsonify({"error": "study_hours and attendance_percentage must be numeric values."}), 400
    if study_hours < 0:
        return jsonify({"error": "study_hours cannot be negative."}), 400
    if not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100."}), 400

    # Keep feature order consistent with training.
    prediction = int(model.predict([[float(study_hours), float(attendance)]])[0])
    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": prediction,
        "label": "Pass" if prediction == 1 else "Fail",
    })


if __name__ == "__main__":
    # Flask's local development server is suitable for this lab exercise.
    app.run(host="127.0.0.1", port=5000, debug=False)
