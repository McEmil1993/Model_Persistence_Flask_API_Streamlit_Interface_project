from pathlib import Path
import math
import joblib
from flask import Flask, jsonify, request
from sklearn.tree import DecisionTreeClassifier

MODEL_PATH = Path(__file__).resolve().parent / "student_pass_model.joblib"

app = Flask(__name__)


def train_and_save_model():
    study_hours = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]

    results = [
        "Fail", "Fail", "Fail", "Fail", "Pass",
        "Pass", "Pass", "Pass", "Pass", "Pass"
    ]

    model = DecisionTreeClassifier(random_state=42)
    model.fit(study_hours, results)

    joblib.dump(model, MODEL_PATH)
    return model


def load_model():
    if not MODEL_PATH.exists():
        train_and_save_model()

    return joblib.load(MODEL_PATH)


model = load_model()


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Student Pass/Fail Prediction API",
        "endpoint": "/predict",
        "method": "POST",
        "example_json": {
            "study_hours": 5,
            "attendance_percentage": 90
        }
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": (
                "Send a JSON object containing study_hours "
                "and attendance_percentage."
            )
        }), 400

    if "study_hours" not in data:
        return jsonify({
            "error": "study_hours is required."
        }), 400

    if "attendance_percentage" not in data:
        return jsonify({
            "error": "attendance_percentage is required."
        }), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])

    except (TypeError, ValueError):
        return jsonify({
            "error": (
                "study_hours and attendance_percentage "
                "must be numeric."
            )
        }), 400

    if not math.isfinite(study_hours) or not math.isfinite(attendance):
        return jsonify({
            "error": "Values must be finite numbers."
        }), 400

    if study_hours < 0:
        return jsonify({
            "error": "study_hours cannot be negative."
        }), 400

    if not 0 <= attendance <= 100:
        return jsonify({
            "error": (
                "attendance_percentage must be between 0 and 100."
            )
        }), 400

    prediction = model.predict([[study_hours]])[0]

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": str(prediction),
        "note": (
            "This sample model uses study_hours only. "
            "Attendance is validated and returned, "
            "but does not affect the prediction."
        )
    }), 200


if __name__ == "__main__":
    print("Flask API: http://127.0.0.1:5000")
    print("Prediction endpoint: POST http://127.0.0.1:5000/predict")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
