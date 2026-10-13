from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request
from flask.views import MethodView

MODEL_FILE = Path(__file__).resolve().parent / "student_pass_model_v2.joblib"
pipeline = joblib.load(MODEL_FILE)    # trained on study hours + attendance

app = Flask(__name__)


class ValidationProblem(ValueError):
    pass


class StudentInput:
    """Reads and validates the two inputs."""

    def __init__(self, raw):
        if not isinstance(raw, dict) or not raw:
            raise ValidationProblem("Send study_hours and attendance_percentage")
        # Task 1: both values are accepted (and required)
        self.study_hours = self._number(raw, "study_hours")
        self.attendance = self._number(raw, "attendance_percentage")

        if self.study_hours < 0:
            raise ValidationProblem("study_hours cannot be negative")
        # Task 2: attendance must be between 0 and 100
        if not 0 <= self.attendance <= 100:
            raise ValidationProblem("attendance_percentage must be between 0 and 100")

    @staticmethod
    def _number(raw, key):
        if key not in raw:
            raise ValidationProblem(f"{key} is required")
        try:
            return float(raw[key])
        except (TypeError, ValueError):
            raise ValidationProblem("Both values must be numbers")


class PredictView(MethodView):
    def get(self):
        return self._respond(request.args.to_dict())

    def post(self):
        return self._respond(request.get_json(silent=True))

    def _respond(self, raw):
        try:
            student = StudentInput(raw)
        except ValidationProblem as problem:
            return jsonify({"error": str(problem)}), 400

        label = pipeline.predict(np.array([[student.study_hours, student.attendance]]))[0]
        # Task 3: return both values and the prediction as JSON
        return jsonify({
            "study_hours": student.study_hours,
            "attendance_percentage": student.attendance,
            "prediction": "Pass" if label == 1 else "Fail",
        })


@app.route("/")
def home():
    return jsonify({"message": "Student Pass Prediction API is running"})


app.add_url_rule("/predict", view_func=PredictView.as_view("predict"))

if __name__ == "__main__":
    app.run(debug=True)