"""Flask API for the student pass prediction model."""

from flask import Flask, jsonify, request

from train_model import load_model


app = Flask(__name__)
model = load_model()


@app.get("/")
def home():
    return jsonify({"message": "Student pass prediction API is running."})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object."}), 400

    try:
        study_hours = float(data["study_hours"])
        attendance_percentage = float(data["attendance_percentage"])
    except (KeyError, TypeError, ValueError):
        return jsonify(
            {
                "error": "study_hours and attendance_percentage must both be numbers."
            }
        ), 400

    if not 0 <= attendance_percentage <= 100:
        return jsonify({"error": "attendance_percentage must be between 0 and 100."}), 400
    if study_hours < 0:
        return jsonify({"error": "study_hours cannot be negative."}), 400

    prediction = int(model.predict([[study_hours]])[0])
    prediction_label = "Pass" if prediction == 1 else "Fail"

    return jsonify(
        {
            "study_hours": study_hours,
            "attendance_percentage": attendance_percentage,
            "prediction": prediction,
            "prediction_label": prediction_label,
            "note": "The current model uses study_hours only. Attendance is validated and returned but does not yet affect the prediction.",
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
