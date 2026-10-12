"""Exercise 2 — Flask API.

Accepts `study_hours` and `attendance_percentage`, validates attendance
is between 0 and 100, and returns both values plus the model prediction.

NOTE: The provided model (`student_pass_model.joblib`) was trained on
study hours ONLY (single feature). So attendance is accepted, validated,
and echoed back, but does not influence the prediction. To make attendance
influence predictions, retrain with 2 features, e.g.:

    X = [[hours, attendance], ...]
    model.fit(X, y)
    joblib.dump(model, "student_pass_model.joblib")

and then use `model.predict([[hours, attendance]])`.
"""
import joblib
from flask import Flask, jsonify, request

MODEL_FILENAME = "student_pass_model.joblib"

app = Flask(__name__)

try:
    model = joblib.load(MODEL_FILENAME)
except FileNotFoundError:
    model = None
    print(f"WARNING: {MODEL_FILENAME} not found. Run train_model.py first.")


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "Student Pass Prediction API",
        "endpoint": "/predict (POST)",
        "example_payload": {"study_hours": 6, "attendance_percentage": 85},
    })


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": f"Model file {MODEL_FILENAME} not found. Run train_model.py first."}), 500

    data = request.get_json(silent=True) or {}

    # Accept both snake_case names; attendance_percentage is required per the task.
    study_hours = data.get("study_hours")
    attendance = data.get("attendance_percentage")

    # --- Validation ---
    if study_hours is None:
        return jsonify({"error": "Missing required field: 'study_hours'."}), 400
    if attendance is None:
        return jsonify({"error": "Missing required field: 'attendance_percentage'."}), 400

    try:
        study_hours = float(study_hours)
    except (TypeError, ValueError):
        return jsonify({"error": "'study_hours' must be a number."}), 400

    try:
        attendance = float(attendance)
    except (TypeError, ValueError):
        return jsonify({"error": "'attendance_percentage' must be a number."}), 400

    if study_hours < 0 or study_hours > 24:
        return jsonify({"error": "'study_hours' must be between 0 and 24."}), 400

    if attendance < 0 or attendance > 100:
        return jsonify({"error": "'attendance_percentage' must be between 0 and 100."}), 400

    # --- Prediction (model uses study_hours only) ---
    pred = int(model.predict([[study_hours]])[0])
    proba = float(model.predict_proba([[study_hours]])[0].max())

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": pred,
        "result": "Pass" if pred == 1 else "Fail",
        "confidence": round(proba, 4),
        "note": "Prediction is based on study_hours only. Retrain with attendance as a second feature to let it influence predictions.",
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
