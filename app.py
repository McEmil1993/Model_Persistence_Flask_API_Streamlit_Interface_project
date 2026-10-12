
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the saved model
model = joblib.load("student_pass_model.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read JSON request body
        data = request.get_json(force=True)
    except Exception:
        return jsonify({
            "error": "Invalid JSON. Please send valid JSON data."
        }), 400

    if not isinstance(data, dict):
        return jsonify({
            "error": "JSON data is required."
        }), 400

    # Validate required fields
    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": "Provide numeric study_hours and attendance_percentage."
        }), 400

    # Validate values
    if study_hours < 0:
        return jsonify({
            "error": "Study hours cannot be negative."
        }), 400

    if not 0 <= attendance <= 100:
        return jsonify({
            "error": "Attendance must be between 0 and 100."
        }), 400

    # Predict using study hours
    prediction = model.predict([[study_hours]])[0]

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": str(prediction),
        "note": "Attendance is validated but is not used by this model."
    })


if __name__ == "__main__":
    app.run(debug=True)
