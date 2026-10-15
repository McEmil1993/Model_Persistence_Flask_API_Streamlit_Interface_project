"""Flask API — Exercise 2. POST /predict with study_hours + attendance_percentage."""
import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "student_pass_model.joblib")
FEATURES = ["study_hours", "attendance_percentage"]

app = Flask(__name__)
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Student Pass API. POST JSON to /predict",
                    "example": {"study_hours": 6.5, "attendance_percentage": 85.0}})


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model file student_pass_model.joblib not found. Run train_model.py first."}), 500
    data = request.get_json(silent=True) or {}

    # --- presence validation ---
    if "study_hours" not in data:
        return jsonify({"error": "Missing required field: 'study_hours' (number)."}), 400
    if "attendance_percentage" not in data:
        return jsonify({"error": "Missing required field: 'attendance_percentage' (number 0-100)."}), 400

    # --- type validation ---
    try:
        study_hours = float(data["study_hours"])
    except (TypeError, ValueError):
        return jsonify({"error": "'study_hours' must be a valid number."}), 400
    try:
        attendance = float(data["attendance_percentage"])
    except (TypeError, ValueError):
        return jsonify({"error": "'attendance_percentage' must be a valid number between 0 and 100 inclusive."}), 400

    # --- range validation (Exercise 2 requirement) ---
    if not (0 <= attendance <= 100):
        return jsonify({"error": f"'attendance_percentage' must be between 0 and 100 inclusive. Got {data['attendance_percentage']}."}), 400
    if study_hours < 0 or study_hours > 24:
        return jsonify({"error": f"'study_hours' must be between 0 and 24. Got {data['study_hours']}."}), 400

    X_new = pd.DataFrame([[study_hours, attendance]], columns=FEATURES)
    pred = int(model.predict(X_new)[0])
    label = "Pass" if pred == 1 else "Fail"
    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": label
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

# ---------------------------------------------------------------------------
# API TESTING DELIVERABLES (run Flask first:  python app.py)
# Valid request:
#   curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\": 6.5, \"attendance_percentage\": 85.0}"
# Invalid attendance (out of range):
#   curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\": 6.5, \"attendance_percentage\": 150}"
# Missing attendance:
#   curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\": 6.5}"
# Non-numeric attendance:
#   curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"study_hours\": 6.5, \"attendance_percentage\": \"high\"}"
# ---------------------------------------------------------------------------
