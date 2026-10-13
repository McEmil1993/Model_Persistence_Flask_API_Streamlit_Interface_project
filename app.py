
from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Load the saved model
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "student_pass_model.joblib"
)

model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Student Pass Prediction API is running!"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Please send valid JSON"}), 400

    study_hours = data.get("study_hours")
    attendance = data.get("attendance_percentage")

    # Validate study hours
    if (
        isinstance(study_hours, bool)
        or not isinstance(study_hours, (int, float))
        or not 0 <= study_hours <= 24
    ):
        return jsonify({
            "error": "Study hours must be between 0 and 24"
        }), 400

    # Validate attendance percentage
    if (
        isinstance(attendance, bool)
        or not isinstance(attendance, (int, float))
        or not 0 <= attendance <= 100
    ):
        return jsonify({
            "error": "Attendance must be between 0 and 100"
        }), 400

    # Model currently uses study hours only
    prediction = model.predict([[study_hours]])[0]

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": str(prediction),
        "note": "Prediction is based on study hours only."
    })


if __name__ == "__main__":
    app.run(debug=True)
