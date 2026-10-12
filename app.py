import math
from pathlib import Path
from typing import Any

import joblib
from flask import Flask, jsonify, request

from student_model import StudentPassModel


MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")
app = Flask(__name__)


def load_model() -> StudentPassModel:
    if not MODEL_PATH.exists():
        raise RuntimeError("Model file not found. Run 'python train_model.py' first.")
    return joblib.load(MODEL_PATH)


model = load_model()


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@app.get("/health")
def health() -> Any:
    return jsonify({"status": "ok", "model_loaded": True})


@app.post("/predict")
def predict() -> Any:
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    study_hours = data.get("study_hours")
    attendance = data.get("attendance_percentage")

    if not is_number(study_hours) or not 0 <= study_hours <= 24:
        return jsonify({"error": "study_hours must be a number between 0 and 24."}), 400
    if not is_number(attendance) or not 0 <= attendance <= 100:
        return jsonify({"error": "attendance_percentage must be a number between 0 and 100."}), 400

    predicted_class = int(model.predict([[study_hours]])[0])
    return jsonify(
        {
            "study_hours": study_hours,
            "attendance_percentage": attendance,
            "predicted_class": predicted_class,
            "prediction": "Pass" if predicted_class == 1 else "Fail",
            "note": "Attendance is validated and returned, but this model predicts from study_hours only.",
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)