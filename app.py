
"""Flask API for student pass predictions."""

from pathlib import Path

import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run python train_model.py first."
        )

    return joblib.load(MODEL_PATH)


# Homepage
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Student Pass Prediction API is running.",
        "endpoints": {
            "home": "GET /",
            "predict": "POST /predict"
        }
    })


# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True)

    if not isinstance(body, dict):
        return jsonify(
            error="Request body must be a JSON object."
        ), 400

    try:
        study_hours = float(body["study_hours"])
        attendance_percentage = float(
            body["attendance_percentage"]
        )
    except (KeyError, TypeError, ValueError):
        return jsonify(
            error=(
                "study_hours and attendance_percentage "
                "must be numbers."
            )
        ), 400

    if study_hours < 0:
        return jsonify(
            error="study_hours must be zero or greater."
        ), 400

    if not 0 <= attendance_percentage <= 100:
        return jsonify(
            error=(
                "attendance_percentage must be "
                "between 0 and 100."
            )
        ), 400

    try:
        model = load_model()

        # Use both inputs if the model was trained with two features.
        prediction = int(
            model.predict([
                [study_hours, attendance_percentage]
            ])[0]
        )

    except FileNotFoundError as error:
        return jsonify(error=str(error)), 500

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify(
            error=(
                "Unable to make a prediction. "
                "Check the trained model and its input features."
            )
        ), 500

    return jsonify(
        study_hours=study_hours,
        attendance_percentage=attendance_percentage,
        prediction=prediction,
        result="Pass" if prediction == 1 else "Needs more study"
    )


if __name__ == "__main__":
    app.run(debug=True)
