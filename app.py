
from flask import Flask, request, jsonify
import joblib
from pathlib import Path

app = Flask(__name__)

# Load the trained model
model_path = Path(__file__).parent / "student_pass_model.joblib"
model = joblib.load(model_path)


@app.route("/")
def home():
    return jsonify({
        "message": "Student Pass Prediction API is running."
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Request body must be valid JSON."
        }), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": (
                "Please provide numeric study_hours and "
                "attendance_percentage values."
            )
        }), 400

    if study_hours < 0:
        return jsonify({
            "error": "Study hours cannot be negative."
        }), 400

    if not 0 <= attendance <= 100:
        return jsonify({
            "error": "Attendance must be between 0 and 100."
        }), 400

    student_data = [[study_hours, attendance]]
    prediction = model.predict(student_data)[0]

    result = "Pass" if prediction == 1 else "Fail"

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": result
    })


if __name__ == "__main__":
    app.run(debug=True)
