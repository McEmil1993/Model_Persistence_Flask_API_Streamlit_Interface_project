
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the saved model
model = joblib.load("student_pass_model.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance_percentage"])
    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": "Provide valid study_hours and attendance_percentage."
        }), 400

    if not 0 <= attendance <= 100:
        return jsonify({
            "error": "Attendance must be between 0 and 100."
        }), 400

    prediction = model.predict([[study_hours]])[0]

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance,
        "prediction": str(prediction)
    })

if __name__ == "__main__":
    app.run(debug=True)
