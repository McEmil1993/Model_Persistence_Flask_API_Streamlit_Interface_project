from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request
from werkzeug.exceptions import BadRequest

MODEL_FILE = Path(__file__).resolve().parent / "student_pass_model_v2.joblib"

app = Flask(__name__)
model = joblib.load(MODEL_FILE)


@dataclass
class StudentRecord:
    study_hours: float
    attendance_percentage: float

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or not data:
            raise BadRequest("Send study_hours and attendance_percentage")
        for field in ("study_hours", "attendance_percentage"):
            if field not in data:
                raise BadRequest(f"{field} is required")
        try:
            record = cls(float(data["study_hours"]), float(data["attendance_percentage"]))
        except (TypeError, ValueError):
            raise BadRequest("Both values must be numbers")

        if record.study_hours < 0:
            raise BadRequest("study_hours cannot be negative")
        if not (0 <= record.attendance_percentage <= 100):
            raise BadRequest("attendance_percentage must be between 0 and 100")
        return record


@app.errorhandler(BadRequest)
def handle_bad_request(error):
    return jsonify({"error": error.description}), 400


@app.route("/")
def home():
    return jsonify({"message": "Student Pass Prediction API is running"})


@app.route("/predict", methods=["GET", "POST"])
def predict():
    raw = request.get_json(silent=True) if request.method == "POST" else request.args.to_dict()
    record = StudentRecord.from_dict(raw)

    label = model.predict(np.array([[record.study_hours, record.attendance_percentage]]))[0]

    return jsonify({
        "study_hours": record.study_hours,
        "attendance_percentage": record.attendance_percentage,
        "prediction": "Pass" if label == 1 else "Fail",
    })


if __name__ == "__main__":
    app.run(debug=True)