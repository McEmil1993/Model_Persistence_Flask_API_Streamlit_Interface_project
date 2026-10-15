"""
Exercise 2 — Flask API

Serves the trained model through a REST endpoint.
Accepts study_hours and attendance_percentage, validates inputs,
and returns the prediction as JSON.

Both features influence the prediction (the model was trained
on [study_hours, attendance_percentage]).
"""

import os

from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Resolve the model path relative to this file so the API works
# no matter which directory it is started from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "student_pass_model.joblib")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file '{MODEL_PATH}' not found. Run train_model.py first."
    )
model = joblib.load(MODEL_PATH)
N_FEATURES = getattr(model, "n_features_in_", 2)


@app.route("/", methods=["GET"])
def index():
    """Helpful landing page so opening the server URL in a browser is not a 404."""
    return jsonify({
        "message": "Student Pass/Fail Prediction API",
        "usage": "POST JSON to /predict with study_hours and attendance_percentage",
        "example": {"study_hours": 6, "attendance_percentage": 85},
        "endpoints": ["/", "/health", "/predict"],
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_features": int(N_FEATURES)})


@app.route("/predict", methods=["POST"])
def predict():
    """Predict pass/fail from JSON body.

    Expected JSON:
        {
            "study_hours": <float>,
            "attendance_percentage": <float 0-100>
        }
    """
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    # --- Validate study_hours ---
    study_hours = data.get("study_hours")
    if study_hours is None:
        return jsonify({"error": "'study_hours' is required."}), 400
    try:
        study_hours = float(study_hours)
    except (ValueError, TypeError):
        return jsonify({"error": "'study_hours' must be a number."}), 400
    if study_hours < 0:
        return jsonify({"error": "'study_hours' must be >= 0."}), 400

    # --- Validate attendance_percentage ---
    attendance = data.get("attendance_percentage")
    if attendance is None:
        return jsonify({"error": "'attendance_percentage' is required."}), 400
    try:
        attendance = float(attendance)
    except (ValueError, TypeError):
        return jsonify({"error": "'attendance_percentage' must be a number."}), 400
    if not (0 <= attendance <= 100):
        return jsonify({"error": "'attendance_percentage' must be between 0 and 100."}), 400

    # --- Predict ---
    # The current model is trained on [study_hours, attendance_percentage].
    # If an old 1-feature model file is still on disk, fall back to
    # study_hours only instead of crashing.
    if N_FEATURES == 1:
        features = [[study_hours]]
    else:
        features = [[study_hours, attendance]]

    prediction = int(model.predict(features)[0])
    label = "Pass" if prediction == 1 else "Fail"

    response = {
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": prediction,
        "result": label,
    }
    # Include the pass probability when the model supports it.
    if hasattr(model, "predict_proba"):
        try:
            response["probability_pass"] = round(float(model.predict_proba(features)[0][1]), 4)
        except Exception:
            pass
    return jsonify(response)


if __name__ == "__main__":
    # use_reloader=False avoids the watchdog restart loop seen with
    # debug=True on Windows, which made the server appear to hang/crash.
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
