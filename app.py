"""Flask API for student pass predictions."""

from flask import Flask, jsonify, request

from train_model import load_model, prediction_label


app = Flask(__name__)
model = load_model()


def get_input_value(name: str):
    """Read a value from JSON, form data, or query parameters."""
    json_data = request.get_json(silent=True) or {}
    return json_data.get(name, request.form.get(name, request.args.get(name)))


@app.get("/")
def index():
    return jsonify(
        message="Student pass prediction API",
        endpoint="POST /predict",
        example={"study_hours": 6, "attendance_percentage": 85},
    )


@app.post("/predict")
def predict():
    study_hours_value = get_input_value("study_hours")
    attendance_value = get_input_value("attendance_percentage")

    if study_hours_value is None or attendance_value is None:
        return jsonify(error="study_hours and attendance_percentage are required."), 400

    try:
        study_hours = float(study_hours_value)
        attendance_percentage = float(attendance_value)
    except (TypeError, ValueError):
        return jsonify(error="study_hours and attendance_percentage must be numbers."), 400

    if study_hours < 0:
        return jsonify(error="study_hours must be 0 or greater."), 400
    if not 0 <= attendance_percentage <= 100:
        return jsonify(error="attendance_percentage must be between 0 and 100."), 400

    prediction = prediction_label(model, study_hours)
    return jsonify(
        study_hours=study_hours,
        attendance_percentage=attendance_percentage,
        prediction=prediction,
        note="This model currently uses study_hours only; attendance is validated and returned but does not affect the prediction.",
    )


if __name__ == "__main__":
    app.run(debug=True)
